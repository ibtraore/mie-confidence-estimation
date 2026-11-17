import math
import ast
import sys
import re
import json
import unicodedata
import pandas as pd
import numpy as np
from math import nan
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode

def get_field_spans(raw_text: str, fields: list[str]) -> dict:
    field_info = {}
    for f in fields:
        key_pattern = f'"{f}"'
        key_idx = raw_text.find(key_pattern)
        if key_idx == -1:
            continue
        colon_idx = raw_text.find(':', key_idx + len(key_pattern))
        if colon_idx == -1:
            continue
        val_start = colon_idx + 1
        while val_start < len(raw_text) and raw_text[val_start] in ' \n\t':
            val_start += 1
        field_info[f] = {"key_idx": key_idx, "val_start": val_start}

    ordered = sorted(field_info.items(), key=lambda kv: kv[1]["key_idx"])

    spans = {}
    obj_end = raw_text.rfind('}')
    for i, (f, info) in enumerate(ordered):
        start = info["val_start"]
        end = ordered[i + 1][1]["key_idx"] if i + 1 < len(ordered) else obj_end

        segment = raw_text[start:end].rstrip()
        segment = segment.rstrip(', \n\t')
        end = start + len(segment)

        spans[f] = (start, end)

    return spans

def build_full_text_with_offsets(logprobs: list):
    """
    Reconstructs the complete generated text (thinking + response) by concatenating the tokens in order. The offsets are exact by construction:
    no need for .find() anymore, thus eliminating ambiguity and synchronization issues.
    """
    full_text = ""
    token_spans = []
    for entry in logprobs:
        tok = entry["token"]
        start = len(full_text)
        full_text += tok
        end = len(full_text)
        token_spans.append((start, end, entry))
    return full_text, token_spans


def align_tokens_to_fields(logprobs: list, parsed: dict, field_names: list = None,
                            think_end_tag: str = "</thinking>"):
    """
    Aligns the tokens with the fields in the `parsed` dictionary, excluding the reasoning block (`<thinking>...</thinking>`) when searching for values.

    field_names: list of field names as they actually appear in `raw_text`, in the SAME ORDER as the keys in `parsed`.
    If None, the keys of `parsed` are used directly.
    """
    full_text, token_spans = build_full_text_with_offsets(logprobs)

    think_end_idx = full_text.rfind(think_end_tag)
    search_start = think_end_idx + len(think_end_tag) if think_end_idx != -1 else 0
    answer_text = full_text[search_start:]

    parsed_keys = list(parsed.keys())
    if field_names is None:
        field_names = parsed_keys
    elif len(field_names) != len(parsed_keys):
        raise ValueError(
            f"field_names ({len(field_names)} elements) must have the same "
            f"length as parsed ({len(parsed_keys)} elements)."
        )

    spans = {}
    for field, fname_in_text in zip(parsed_keys, field_names):
        value = parsed[field]
        span = find_value_span(answer_text, fname_in_text, value)
        if span:
            s, e = span
            spans[field] = (s + search_start, e + search_start) 

    result = {}
    for field, (s, e) in spans.items():
        result[field] = [
            entry for (ts, te, entry) in token_spans
            if ts >= s and te <= e
        ]
    return result


def to_raw_text_form(value_str: str) -> str:
    """
    Converts a decoded Python value (containing actual \n, \t, etc.) back into its literal JSON-escaped form, as it appears in `raw_text`.
    json.dumps('a\nb') -> '"a\\nb"'; the enclosing quotation marks are removed.
    """
    return json.dumps(value_str)[1:-1] 


def find_value_span(raw_text: str, field_name: str, parsed_value, debug: bool = False):
    if isinstance(parsed_value, list):
        value_str = str(parsed_value[0]) if parsed_value else None
    else:
        value_str = parsed_value

    field_pattern = rf"\b{re.escape(field_name)}\b"
    quote_open = r"[\"'\u201c\u2018]"
    quote_close = r"[\"'\u201d\u2019]"

    if value_str is None:
        near_pattern = re.compile(rf"{field_pattern}[^A-Za-z]{{0,20}}\bNone\b")
        m = near_pattern.search(raw_text)
        if m:
            none_match = re.search(r"\bNone\b", m.group(0))
            offset = m.start() + none_match.start()
            return (offset, offset + len("None"))
        m = re.search(r"\bNone\b", raw_text)
        return (m.start(), m.end()) if m else None

    raw_form = to_raw_text_form(str(value_str))
    escaped_value = re.escape(raw_form)

    m = re.search(
        rf"{field_pattern}.{{0,10}}?\[\s*{quote_open}\s*({escaped_value})\s*{quote_close}\s*\]",
        raw_text, re.DOTALL,
    )
    if m:
        if debug: print("[Cas 2 - list] OK")
        return m.start(1), m.end(1)

    m = re.search(
        rf"<\s*{field_pattern}\.\w+\s*:\s*{quote_open}\s*({escaped_value})\s*{quote_close}\s*>",
        raw_text,
    )
    if m:
        if debug: print("[Cas 3 - pydantic-like] OK")
        return m.start(1), m.end(1)

    m = re.search(
        rf"{quote_open}?{field_pattern}{quote_close}?\s*:\s*{quote_open}\s*({escaped_value})\s*{quote_close}",
        raw_text, re.DOTALL,
    )
    if m:
        if debug: print("[Cas 4 - JSON quoted] OK")
        return m.start(1), m.end(1)

    m = re.search(
        rf"{quote_open}?{field_pattern}{quote_close}?\s*[:=]\s*({escaped_value})(?=[\s,}}\]\n]|$)",
        raw_text, re.DOTALL,
    )
    if m:
        if debug: print("[Cas 5 - without quotation marks] OK")
        return m.start(1), m.end(1)

    m = re.search(escaped_value, raw_text, re.DOTALL)
    if m:
        if debug: print("[Cas 6 - fallback] OK")
        return m.start(), m.end()

    if debug:
        print(f"[TOTAL FAILED] field_name={field_name!r}")
        print(f"value_str (decoded)  = {value_str!r}")
        print(f"raw_form (decoded)  = {raw_form!r}")
        print(f"raw_text (200 car.) = {raw_text[:200]!r}")
    return None

# ══════════════════════════════════════════════════════════════════════════════
# UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════════
def uncertainty_metrics(tokens: list, k: int = 5):
    
    if not tokens:
        return None

    logps = [t["logprob"] for t in tokens]
    probs = [math.exp(lp) for lp in logps]
    n = len(logps)

    # --- Perplexity ---
    perplexity = math.exp(-sum(logps) / n)

    # --- Sequence Probability (SP) ---
    sp = math.exp(sum(logps))

    # --- Length-Normalized Sequence Probability (LNSP) ---
    lnsp = math.exp(sum(logps) / n)  # = 1 / perplexity

    # --- Minimum Token Probability (MTP) ---
    mtp = min(probs)

    # --- Probability Margin (PM) ---
    margins = []
    for t in tokens:
        alts = sorted(t.get("top_logprobs", []), key=lambda a: a["logprob"], reverse=True)
        if len(alts) >= 2:
            p_top1 = math.exp(alts[0]["logprob"])
            p_top2 = math.exp(alts[1]["logprob"])
            margins.append(p_top1 - p_top2)
        elif len(alts) == 1:
            margins.append(math.exp(alts[0]["logprob"]))
    pm = sum(margins) / len(margins) if margins else None

    # --- Negentropy@K, then ATN@K and MTN@K ---
    negentropies = []
    for t in tokens:
        alts = sorted(t.get("top_logprobs", []), key=lambda a: a["logprob"], reverse=True)[:k]
        k_eff = len(alts)
        if k_eff < 2:
            continue 
        raw_probs = [math.exp(a["logprob"]) for a in alts]
        total = sum(raw_probs)
        norm_probs = [p / total for p in raw_probs]
        h = -sum(p * math.log(p) for p in norm_probs if p > 0)
        h_max = math.log(k_eff)
        negentropy = 1 - (h / h_max) if h_max > 0 else 1.0
        negentropies.append(negentropy)

    atn_k = sum(negentropies) / len(negentropies) if negentropies else None
    mtn_k = min(negentropies) if negentropies else None

    return {
        "perplexity": perplexity,
        "SP": sp,
        "LNSP": lnsp,
        "MTP": mtp,
        "PM": pm,
        f"ATN@{k}": atn_k,
        f"MTN@{k}": mtn_k,
    }

def filter_structural(tokens):
    STRUCTURAL_TOKENS = {"[", "]", "<", ">", '"', ",", "{", "}", ":"}
    return [t for t in tokens if t["token"].strip() not in STRUCTURAL_TOKENS]



def safe_parse(x):
    k = 5
    if pd.isna(x):
        return {
            "perplexity": nan,
            "SP": nan,
            "LNSP": nan,
            "MTP": nan,
            "PM": nan,
            f"ATN@{k}": nan,
            f"MTN@{k}": nan,
        }
    try:
        if isinstance(x, dict):
            return x
        return ast.literal_eval(x)
    except (ValueError, SyntaxError, TypeError):
        return {
            "perplexity": nan,
            "SP": nan,
            "LNSP": nan,
            "MTP": nan,
            "PM": nan,
            f"ATN@{k}": nan,
            f"MTN@{k}": nan,
        }

def extract_unc(row):
    """Extract uncertainty scores from output"""
    unc_dict = safe_parse(row["value_uncertainty"])
    k = 5
    
    return {
        "perplexity": unc_dict["perplexity"],
        "SP": unc_dict["SP"],
        "LNSP": unc_dict["LNSP"],
        "MTP": unc_dict["MTP"],
        "PM": unc_dict["PM"],
        f"ATN@{k}": unc_dict[f"ATN@{k}"],
        f"MTN@{k}": unc_dict[f"MTN@{k}"],
    }

# ══════════════════════════════════════════════════════════════════════════════
# compute_f1_score
# ══════════════════════════════════════════════════════════════════════════════
def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    - Lowercase
    - Remove accents
    - Remove extra whitespace
    
    Args:
        text: Raw text string
    
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    tokens = [unidecode(t.strip()) for t in text.split()]
    
    # Remove extra whitespace
    text = " ".join(tokens)
    
    return text.strip()

def compute_f1_score(predictions, ground_truths, f1_threshold = 0.5) -> bool:
    """
    Compute token-level F1 score between predictions and ground truths.
    
    Returns boolean if the F1 score is greather than the threshold.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of acceptable answers
    
    Returns:
        True or False
    """

    if not isinstance(predictions, list):
        predictions = list(predictions)

    if not isinstance(ground_truths, list):
        ground_truths = list(ground_truths)

    if len(predictions) == 0 and len(ground_truths) == 0 :
        return True
    
    max_f1 = 0.0
    correct_words = 0

    for pred in predictions:
        pred_tokens = normalize_text(pred).split()
        
        for gt in ground_truths:
            gt_tokens = normalize_text(gt).split()
            
            if not gt_tokens:
                continue
            
            # Count common tokens
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_common = sum(common.values())
    
            if num_common == 0:
                continue
            
            precision = num_common / len(pred_tokens)
            recall = num_common / len(gt_tokens)
            
            f1 = 2 * precision * recall / (precision + recall)
            max_f1 = max(max_f1, f1)
            correct_words = correct_words + 1 if max_f1 >= f1_threshold else correct_words

    if correct_words == 0 :
        return False

    precision = correct_words / len(predictions)
    recall = correct_words / len(ground_truths)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1 >= f1_threshold

# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_confidence_analysis(df, output_dir="figures"):

    os.makedirs(output_dir, exist_ok=True)

    confidence_order = [
        "very low",
        "low",
        "high",
        "very high"
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.countplot(
        data=df,
        x="verbal_confidence",
        hue="correctness",
        order=confidence_order,
        palette={0: "tomato", 1: "seagreen"},
        ax=ax
    )
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontsize=10)

    plt.title("Distribution of answers by Pred confidence level")
    plt.xlabel("Pred Confidence level")
    plt.ylabel("Number of answers")
    plt.legend(title="Correctness", labels=["Incorrect (0)", "Correct (1)"])
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "verbal_confidence_vs_Correctness.png"), dpi=300)
    plt.close()