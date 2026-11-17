from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
import json
import os
import re
import yaml
from collections import defaultdict
import ast
from pathlib import Path
from config import Config
from utils import *
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, TypedDict, Literal
from sklearn.metrics import f1_score

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
FIELDS = ["route", "dosage", "frequency", "duration", "reason"]

class Validator(BaseModel):
    Response: str = Field(description="The Response.")
    Confidence: Literal["very high", "high", "low", "very low"] = Field(description="The confidence.")

def prepare_prompt(text: str) -> str:
    """
    Prepare a prompt for inference
    
    Args:
        text: Clinical text
    
    Returns:
        The formatted prompt
    """
    
    prompt = f"""
    ## <One-shot-example>
    Albuterol 2 puffs every 4-6 hours as needed. HOSPITAL COURSE: This is an 80-year-old female who was hospitalized about 2 months ago for chronic obstructive pulmonary disease exacerbation. At that time she was put on prednisone and antibiotics and seemed to get better. However,
    she was put on Augmentin ES and continued to have difficulty tasting food and felt that food tasted very salty. She had no appetite and she has continued to lose weight over the last 2 months.

    Label medications, ignoring allergies. Include dosage, route, frequency, duration, reason, if available.

    ## <One-shot-labels>
    "entities : \n"
        "	- group : 1 \n"
        "	MEDICATION : 'Albuterol' \n"
        " 	DOSE : '2 puffs' \n"
        " 	FREQUENCY : 'every 4-6 hours' \n"
        " 	DURATION : 'as needed' \n"
        " 	REASON : '' \n"
        " 	ROUTE : '' \n"
        "	- group : 2 \n"
        "	MEDICATION : 'prednisone' \n"
        " 	DOSE : '' \n"
        " 	FREQUENCY : '' \n"
        " 	DURATION : '2 months' \n"
        " 	REASON : '' \n"
        " 	ROUTE : '' \n"
        "	- group : 3 \n"
        "	MEDICATION : 'antibiotics' \n"
        " 	DOSE : '' \n"
        " 	FREQUENCY : '' \n"
        " 	DURATION : '2 months' \n"
        " 	REASON : '' \n"
        " 	ROUTE : '' \n"
        "	- group : 4 \n"
        "	MEDICATION : 'Augmentin ES' \n"
        " 	DOSE : ''  \n"
        " 	FREQUENCY : '' \ n "
        " 	DURATION : '2 months' \n"
        " 	REASON : '' \n"
        " 	ROUTE : '' \n"
    
    -----------------
    ## <Task>
    {text}

   Label medications, ignoring allergies. Include dosage, route, frequency, duration, reason, if available.
    -----------------
    ## <Output-format>
    Return your answer in a JSON object with the following fields:
    - Response: [Write your answer here. YAML template designed for representing a list of medications.]
    - Confidence: [Write your confidence score here. Provide one of the following confidence scores measuring your level of confidence regarding your answer: very high, high, low, very low.]
    """
    

    return prompt



def respond(df, df_example, config):
    columns = ["snippet", "annotations"]
    df = df.reset_index(drop=True)
    new_df = df[columns].copy()
    if isinstance(new_df, pd.Series):
        new_df = new_df.to_frame()

    new_df['pred_annotations'] = None
    new_df['verbal_confidence'] = None
    new_df['value_uncertainty'] = None
    
    llm = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            top_p=config.top_p,
            api_key=config.api_key,
            base_url=config.base_url,
            num_predict=config.max_tokens,
            max_retries=config.max_retries,
            timeout=config.timeout,
            logprobs=config.logprobs,
            top_logprobs=config.top_logprobs,
            reasoning=config.reasoning
    )

    system_prompt = (
        "You are a medical assistant with expertise in clinical document processing.\n\n"
        "- Your task is to tag groups of entities related to medications from the provided text. (Some groups may have overlap as two medications can be used for the same reason.)\n"
        # "- Each provided text contains AT LEAST ONE medication group. You must identify and extract it."
        "- Each medication is a SPECIFIC reference to the drug's name used by the patient. This should not contain other data already within a group.\n"
        "- The mode of administration, such as 'oral' or 'intravenous', should be tagged SEPARATELY from the medication's name.\n"
        "- Text should contain the EXACT characters in the raw input, including punctuation, and typos.\n"
        # "- Do not return an empty response."
    )


    for i in range(df.shape[0]):
        print(f"Processing row {i+1} / {len(df)}")
        row_values = new_df.iloc[i]
        prompt = prepare_prompt(text=row_values["snippet"])
        
        try:

            messages = [
                (
                    "system",
                    system_prompt,
                ),
                ("human", prompt),
            ]
            response = llm.with_structured_output(Validator, method="json_schema", include_raw=True).invoke(messages)
            raw_text = response["raw"].content

            if not raw_text :
                new_df.at[i, 'pred_annotations'] = None
                new_df.at[i, 'verbal_confidence'] = None
                new_df.at[i, 'value_uncertainty'] = None
            else:

                match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
                raw_text = match.group(1)
                parsed = json.loads(raw_text)
                logprobs = response["raw"].response_metadata["logprobs"]
                field_tokens = align_tokens_to_fields(logprobs, parsed, raw_text, field_names = ["Response", "Confidence"])
                
                if "Response" in field_tokens:
                    value_tokens = filter_structural(field_tokens["Response"])
                    metrics_value = uncertainty_metrics(value_tokens)
                else:
                    metrics_value = None
    
                new_df.at[i, 'pred_annotations'] = parsed["Response"]
                new_df.at[i, 'verbal_confidence'] = parsed["Confidence"]
                new_df.at[i, 'value_uncertainty'] = metrics_value
            
        except Exception as e:
            print(f"Error occurred at row {i+1}")
            raise
                

    return new_df



def clean_llm_text(raw_text: str) -> str:
    """Cleans an LLM output intended to be YAML but incorrectly formatted."""
    text = raw_text

    text = (text
            .replace("\u2018", "'").replace("\u2019", "'")   # ‘ ’
            .replace("\u201c", '"').replace("\u201d", '"'))  # “ ”

    text = re.sub(r'\\+\s*n', '\n', text)
    text = re.sub(r'\\+\s*t', '\t', text)

    text = re.sub(r'^\s*```(?:yaml)?\s*\n?', '', text.strip())
    text = re.sub(r'\n?\s*```\s*$', '', text.strip())

    text = re.sub(r'^\s*entities\s*:\s*\n', '', text, flags=re.IGNORECASE)

    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line.strip() != "")

    return text

def normalize_field_name(name: str) -> str:
    compact = re.sub(r'\s+', '', name).lower()
    for field in FIELDS:
        if compact == field:
            return field
    return compact


def parse_groups_regex(text: str) -> dict:
    """
    Tolerant fallback parser: does not rely on strictly valid YAML.
    Recognizes lines such as '- group 1 :' and 'field: "value"'.
    """
    groups = defaultdict(dict)
    current_idx = None

    group_re = re.compile(r'^-?\s*group\s*[:\s]?\s*(\d+)\s*:?\s*$', re.IGNORECASE)
    field_re = re.compile(r'^([A-Za-z][A-Za-z\s]*?)\s*:\s*(.*)$')

    for line in text.split("\n"):
        stripped = line.strip()

        m_group = group_re.match(stripped)
        if m_group:
            current_idx = int(m_group.group(1))
            continue

        m_field = field_re.match(stripped)
        if m_field and current_idx is not None:
            raw_field, raw_value = m_field.groups()
            field = normalize_field_name(raw_field)
            value = raw_value.strip().strip('"').strip("'")
            groups[current_idx][field] = value

    return dict(groups)


def load_groups(raw_text: str) -> dict:
    """
    Complete pipeline: cleaning -> YAML parsing attempt -> regex fallback if parsing fails.
    Returns a dictionary {idx: {field: value, ...}, ...}.
    """
    clean_text = clean_llm_text(raw_text)

    try:
        parsed = yaml.safe_load(clean_text)
        if isinstance(parsed, dict) and "entities" in parsed:
            parsed = parsed["entities"]
        if isinstance(parsed, list):
            groups = defaultdict(dict)
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                for group_label, fields in item.items():
                    m = re.search(r'(\d+)', str(group_label))
                    idx = int(m.group(1)) - 1 if m else i
                    if isinstance(fields, dict):
                        for f, v in fields.items():
                            groups[idx][normalize_field_name(f)] = v
            if groups:
                return dict(groups)
    except yaml.YAMLError:
        pass

    return parse_groups_regex(clean_text)


def normalize(value):
    return (value or "").strip().lower()


def count_fields(entry):
    """Number of non-empty fields in an entry (excluding 'medication')."""
    return sum(1 for f in FIELDS if normalize(entry.get(f)))

def entry_f1(gt_entry, pred_entry):
    tp, fp, fn = compare_entries(gt_entry, pred_entry)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, tp, fp, fn


def compare_entries(gt_entry, pred_entry):
    """Compares two entries (ground truth vs. prediction) field by field -> TP, FP, FN."""
    tp = fp = fn = 0
    for f in FIELDS:
        gt_val = normalize(gt_entry.get(f))
        pred_val = normalize(pred_entry.get(f))
        if gt_val and pred_val:
            if gt_val == pred_val:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif gt_val and not pred_val:
            fn += 1
        elif pred_val and not gt_val:
            fp += 1
    return tp, fp, fn

def horizontal_metric_phrase_level(documents):
    """
    documents: list of dictionaries, each containing:
        - "target_groups": dictionary {idx: {...}} (ground truth)
        - "pred_groups":   dictionary {idx: {...}} (model predictions)

    Returns (Recall, Precision, F1) aggregated across all documents.
    """
    total_tp, total_fp, total_fn = 0, 0, 0

    for doc in documents:
        target_groups = doc["target_groups"]
        pred_groups = doc["pred_groups"]

        gt_entries_map = defaultdict(list)
        model_entries_map = defaultdict(list)

        for entry in target_groups.values():
            key = normalize(entry.get("medication"))
            gt_entries_map[key].append(entry)

        for entry in pred_groups.values():
            key = normalize(entry.get("medication"))
            model_entries_map[key].append(entry)

        all_keys = set(gt_entries_map) | set(model_entries_map)

        for med_key in all_keys:
            med_gt_entries = gt_entries_map.get(med_key, [])
            med_model_entries = model_entries_map.get(med_key, [])

            if not med_gt_entries:
                for entry in med_model_entries:
                    total_fp += count_fields(entry)

            elif not med_model_entries:
                for entry in med_gt_entries:
                    total_fn += count_fields(entry)

            else:
                used_pred_idx = set()
                for gt_entry in med_gt_entries:
                    best_f1 = -1
                    best_metrics = (0, 0, count_fields(gt_entry))
                    best_idx = None
                    for i, pred_entry in enumerate(med_model_entries):
                        if i in used_pred_idx:
                            continue
                        f1, tp, fp, fn = entry_f1(gt_entry, pred_entry)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_metrics = (tp, fp, fn)
                            best_idx = i
                    if best_idx is not None:
                        used_pred_idx.add(best_idx)
                    tp, fp, fn = best_metrics
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

                for i, pred_entry in enumerate(med_model_entries):
                    if i not in used_pred_idx:
                        total_fp += count_fields(pred_entry)

    R = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    P = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    F1 = 2 / ((1 / P) + (1 / R)) if P > 0 and R > 0 else 0.0

    return R, P, F1


def get_correctness(data):

    rows = []

    for index, row in data.iterrows():

        target = row["annotations"]
        pred = row["pred_annotations"]

        if isinstance(target, str):
            target = ast.literal_eval(target)

        target_groups = defaultdict(dict)

        if isinstance(pred, str):
            for start, end, label, value in target:
                m = re.match(r"([A-Za-z]+)(\d+)", label)
                field, idx = m.group(1).lower(), int(m.group(2))
                target_groups[idx][field] = value
    
            pred_groups = load_groups(pred)
    
            documents = [
                {
                    "target_groups": target_groups, 
                    "pred_groups": pred_groups,   
                },
            ]
            
            R, P, F1 = horizontal_metric_phrase_level(documents)
        else:
            F1 = 0

        correctness = 1 if F1 >= 0.5 else 0
        unc = extract_unc(row)

        rows.append({
            "snippet": row["snippet"], 
            "gt_annotations": row["annotations"], 
            "pred_annotations": row["pred_annotations"],
            'verbal_confidence': row["verbal_confidence"],
            "correctness": correctness,
            "perplexity": unc["perplexity"],
            "SP": unc["SP"],
            "LNSP": unc["LNSP"],
            "MTP": unc["MTP"],
            "PM": unc["PM"],
            "ATN@5": unc["ATN@5"],
            "MTN@5": unc["MTN@5"]
        })

    return pd.DataFrame(rows)
    

def main():
    task = "medication_attribute_extraction"
    models = ["model_name"]
    main_dir = os.path.dirname(ROOT_DIR)
    
    # output_dir = Path(main_dir + "/data/baselines/" + task + "/with_reasoning")
    output_dir = Path(main_dir + "/data/baselines/" + task)
    data_file = main_dir + "/data/baselines/medication_attr_test.csv"
    example_file = main_dir + "/data/baselines/medication_attr_dev.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_file)
    df_example = pd.read_csv(example_file)
    df = pd.concat([df, df_example], ignore_index=True)

    for model_name in models:
        output_excel = main_dir + "/data/baselines/" + task + f"/results_{model_name}_{task}.xlsx"
        file_name = main_dir + "/data/baselines/" + task + "/correctness_" + model_name + "_" + task + ".xlsx"
        config = Config(model_name=model_name, dataset_name=task)
        new_df = pd.read_excel(output_excel)
        completed_data = get_correctness(new_df)

        with pd.ExcelWriter(file_name) as writer:
            completed_data.to_excel(writer, sheet_name='correctness', index=False)

    print("Done !")

if __name__ == "__main__":
    main()