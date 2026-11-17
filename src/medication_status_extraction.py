from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
import json
import os
import re
import ast
import math
from pathlib import Path
from config import Config
from utils import *
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, TypedDict, Literal
from sklearn.metrics import f1_score

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 


class Validator(BaseModel):
    Response: str = Field(description="The Response.")
    Confidence: Literal["very high", "high", "low", "very low"] = Field(description="The confidence.")

def prepare_prompt(text: str, status: str) -> str:
    """
    Prepare a prompt for inference
    
    Args:
        text: Clinical text
        status: The status of medication to be identified
    
    Returns:
        The formatted prompt
    """
    
    prompt = f"""
    ## Examples

    ## Example 1
    [...] start her on Ativan 1 mg p.o. q. 8 hours and use Ativan 1 mg IV q. 4 hours p.r.n. for agitation. I will also start her on Inderal LA 60 mg p.o. q.d. for essential tremors. She does not want to take Celexa, and I will put her back on Lexapro 2 mg p.o.    q.d. I will discontinue Esmolol.

    Create a list of active medications [...] 
    ['Inderal LA (active)', 'Lexapro (active)', 'Esmolol (active)']

    ## Example 2
    [...] start her on Ativan 1 mg p.o. q. 8 hours and use Ativan 1 mg IV q. 4 hours p.r.n. for agitation. I will also start her on Inderal LA 60 mg p.o. q.d. for essential tremors. She does not want to take Celexa, and I will put her back on Lexapro 2 mg p.o.    q.d. I will discontinue Esmolol.

    Create a list of discontinued medications [...] 
    ['Ativan (discontinued)']

    ## Example 2
    
    -----------------
    {text}

    Create a list of {status} medications [...] 
    -----------------
    Output format: 
    Return strictly a JSON object with the following fields:
    - Response: [Write your answer here. Return strictly a correct list of {status} medications, if no {status} medications, return an empty list. Do not include any additional text, formatting, or explanation.]
    - Confidence: [Write your confidence score here. Provide one of the following confidence scores measuring your level of confidence regarding your answer: very high, high, low, very low.]

    """
    

    return prompt



def respond(df, df_example, config):
    # columns = ["snippet", "active_medications", "discontinued_medications", "neither_medications"]
    rows = []
    
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

    example_row = df_example.iloc[0]

    for i in range(df.shape[0]):
        print(f"Processing row {i+1} / {len(df)}")
        row_values = df.iloc[i]
        for categorie in ["active", "discontinued", "neither"]:
            prompt = prepare_prompt(text=row_values["snippet"], status=categorie)
            
            try:
  
                messages = [
                    (
                        "system",
                        "You are a helpful assistant in Medication Status Extraction. Think step by step before answering. Write your reasoning inside <thinking> tags before giving the final answer.",
                    ),
                    ("human", prompt),
                ]
                response = llm.with_structured_output(Validator, method="json_schema", include_raw=True).invoke(messages)
                raw_text = response["raw"].content
                if not raw_text :
                    rows.append({
                        "snippet": row_values["snippet"], 
                        "categorie": categorie, 
                        "gt_medications": row_values[f"{categorie}_medications"], 
                        "pred_medications": None,
                        'verbal_confidence': None,
                        "value_uncertainty": None,
                    })
                else:

                    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
                    raw_text = match.group(1)
                    parsed = json.loads(raw_text)
                    logprobs = response["raw"].response_metadata["logprobs"]
                    field_tokens = align_tokens_to_fields(logprobs, parsed, field_names=["Response", "Confidence"])

                    if "Response" in field_tokens:
                        value_tokens = filter_structural(field_tokens["Response"])
                        metrics_value = uncertainty_metrics(value_tokens)
                    else:
                        metrics_value = None
    
                    medications = extract_answer(preprocess_llm_output(parsed["Response"]))
    
                    rows.append({
                        "snippet": row_values["snippet"], 
                        "categorie": categorie, 
                        "gt_medications": row_values[f"{categorie}_medications"], 
                        "pred_medications": medications,
                        'verbal_confidence': parsed["Confidence"],
                        "value_uncertainty": metrics_value,
                    })
                
            except Exception as e:
                print(f"Error occurred at row {i+1}")
                raise
                

    return  pd.DataFrame(rows)


def preprocess_llm_output(text):
    """
    Cleans and parses an LLM output intended to represent a Python list,
    even if the format is imperfect (':' prefix, missing brackets, etc.).

    Returns a list of strings, or [] if nothing usable can be extracted.
    """
    if not isinstance(text, str):
        return []
    
    text = text.strip()
    
    if not text:
        return []
    
    idx_start = text.find("[")
    if idx_start != -1:
        text = text[idx_start:]
    
    if text.startswith("[") and not text.rstrip().endswith("]"):
        text = text.rstrip().rstrip(",") + "]"
    
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return [str(x).strip() for x in result if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    
    matches = re.findall(r"""['"]([^'"]+)['"]""", text)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    
    return []


def parse_status(item):
    match = re.match(r"(.+)\s\((\w+)\)", item)
    if match:
        return match.group(1).strip(), match.group(2).lower()
    return item, None
    
def extract_answer(pred_str):
    if isinstance(pred_str, str):
        pred_str = ast.literal_eval(pred_str)
    
    medications = []

    for item in pred_str:
        name, status = parse_status(item)
        medications.append(name)
    
    return medications


def get_correctness(data):

    rows = []

    for index, row in data.iterrows():

        target = row["gt_medications"]
        pred = row["pred_medications"]

        if isinstance(target, str):
            target = ast.literal_eval(target)

        if isinstance(pred, str):
            pred = ast.literal_eval(pred)
        else:
            pred = []

        correctness_bool = compute_f1_score(pred, target)
        correctness = 1 if correctness_bool == True else 0

        unc = extract_unc(row)

        rows.append({
            "snippet": row["snippet"], 
            "categorie": row["categorie"], 
            "gt_medications": row["gt_medications"], 
            "pred_medications": row["pred_medications"],
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

    return pd.DataFrame(rows, columns=[
        "snippet", "categorie", "gt_medications", "pred_medications",
        "verbal_confidence", "correctness", "perplexity","SP", "LNSP", "MTP", "PM", "ATN@5", "MTN@5",
    ])


def main():
    task = "medication_status_extraction"
    models = ["model_name"]
    main_dir = os.path.dirname(ROOT_DIR)
    
    output_dir = Path(main_dir + "/data/baselines/" + task + "/with_reasoning")

    data_file = main_dir + "/data/baselines/medication_status_test.csv"
    example_file = main_dir + "/data/baselines/medication_status_dev.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_file)
    df_example = pd.read_csv(example_file)
    df = pd.concat([df, df_example], ignore_index=True)

    for model_name in models:
        output_excel = main_dir + "/data/baselines/" + task + f"/with_reasoning/results_{model_name}_{task}.xlsx"
        file_name = main_dir + "/data/baselines/" + task + "/with_reasoning/correctness_" + model_name + "_" + task + ".xlsx"
        config = Config(model_name=model_name, dataset_name=task)

        new_df = pd.read_excel(output_excel)
        completed_data = get_correctness(new_df)

        with pd.ExcelWriter(file_name) as writer:
            completed_data.to_excel(writer, sheet_name='correctness', index=False)

    print("Done !")

if __name__ == "__main__":
    main()