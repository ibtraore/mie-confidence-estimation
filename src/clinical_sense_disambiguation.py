from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
import json
import os
import csv
from pathlib import Path
from config import Config
from utils import *
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, TypedDict, Literal

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

class Validator(BaseModel):
    Response: str = Field(description="The Response.")
    Confidence: Literal["very high", "high", "low", "very low"] = Field(description="The confidence.")

def prepare_prompt(text: str, abbreviation: str, meanings: List[str]) -> str:
    """
    Prepare a prompt for inference
    
    Args:
        text: Clinical text
        question: Question to be processed
    
    Returns:
        The formatted prompt
    """
    
    prompt = f"""
    {text}

    Expand the abbreviation: {abbreviation}

    -----------------
    List of possible meanings:
    {meanings}
    
    -----------------
    Output format: 
    Return strictly a JSON object with the following fields:
    - Response: [Write your answer here. Return only an exact value from the list of possible meanings above.]
    - Confidence: [Write your confidence score here. Provide one of the following confidence scores measuring your level of confidence regarding your answer: very high, high, low, very low.]

    """
    

    return prompt



def respond(df, config, main_dir, output_csv, csv_exists):
    BATCH_SIZE = 100
    buffer_rows = []
    columns = ["sf", "target_lf", "sf_rep", "start_idx", "end_idx", "section", "context"]
    df = df.reset_index(drop=True)
    new_df = df[columns].copy()
    if isinstance(new_df, pd.Series):
        new_df = new_df.to_frame()

    new_df['pred_lf'] = None
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
            # reasoning=config.reasoning
    )

    fieldnames = ["sf", "target_lf", "sf_rep", "start_idx", "end_idx", "section", "context", "pred_lf", "verbal_confidence", "value_uncertainty"]
    all_meanings = json.load(open(main_dir + "/data/baselines/sf_lf_map.json", encoding="UTF-8"))
    file_mode = 'a' if csv_exists else 'w'
    with open(output_csv, mode=file_mode, newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        if not csv_exists:
            writer.writeheader()

        for i in df.index:
            print(f"Processing row {i+1} / {len(df)}")
            row_values = new_df.iloc[i]
            meanings = all_meanings[row_values["sf"]]
            prompt = prepare_prompt(text=row_values["context"], abbreviation=row_values["sf"], meanings=meanings)
            
            try:

                messages = [
                    (
                        "system",
                        "You are a helpful assistant in Clinical Sense Disambiguation.",
                    ),
                    ("human", prompt),
                ]
                response = llm.with_structured_output(Validator, method="json_schema", include_raw=True).invoke(messages)
                raw_text = response["raw"].content
                print("raw_text =============== ", raw_text)
                parsed = json.loads(raw_text)
                logprobs = response["raw"].response_metadata["logprobs"]
                field_tokens = align_tokens_to_fields(logprobs, parsed, field_names=["Response", "Confidence"])
                value_tokens = filter_structural(field_tokens["Response"])
                metrics_value = uncertainty_metrics(value_tokens)
                print("confidence =============== ", metrics_value)
    
                new_df.loc[i, ['pred_lf', 'verbal_confidence', 'value_uncertainty']] = [
                    parsed["Response"],
                    parsed["Confidence"],
                    metrics_value
                ]
    
                buffer_rows.append({
                        "sf": row_values["sf"],
                        "target_lf": row_values["target_lf"],
                        "sf_rep": row_values["sf_rep"],
                        "start_idx": row_values["start_idx"],
                        "end_idx": row_values["end_idx"],
                        "section": row_values["section"],
                        "context": row_values["context"],
                        "pred_lf": parsed["Response"],
                        "verbal_confidence": parsed["Confidence"],
                        "value_uncertainty": metrics_value
                })
    
                if len(buffer_rows) >= BATCH_SIZE:
                    writer.writerows(buffer_rows)
                    buffer_rows.clear()
                
            except Exception as e:
                print(f"Error occurred at row {i+1}")
                raise
        if buffer_rows:
            writer.writerows(buffer_rows)
                
    return new_df


def get_correctness(data):

    rows = []

    for index, row in data.iterrows():
        target_lf = row["target_lf"]
        pred_lf = row["pred_lf"]

        if isinstance(target_lf, str):
            target_lf = [target_lf]

        if isinstance(pred_lf, str):
            pred_lf = [pred_lf]

        correctness_bool = compute_f1_score(pred_lf, target_lf)
        correctness = 1 if correctness_bool == True else 0

        unc = extract_unc(row)

        rows.append({
            "sf": row["sf"], 
            "context": row["context"], 
            "target_lf": row["target_lf"], 
            "pred_lf": row["pred_lf"],
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
        "sf", "context", "target_lf", "pred_lf", "verbal_confidence", "correctness", "perplexity",
        "SP", "LNSP", "MTP", "PM", "ATN@5", "MTN@5",
    ])


def main():
    task = "clinical_sense_disambiguation"
    models = ["model_name"]
    main_dir = os.path.dirname(ROOT_DIR)
    output_dir = Path(main_dir + "/data/baselines/" + task)
    data_file = main_dir + "/data/baselines/clinical_sense_disambiguation.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_file)
    new_df = pd.DataFrame()

    for model_name in models:
        output_csv = main_dir + "/data/baselines/" + task + f"/results_{model_name}_{task}.csv"
        file_name = main_dir + "/data/baselines/" + task + "/correctness_" + model_name + "_" + task + ".xlsx"

        if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
             new_df = pd.read_csv(output_csv, delimiter=";")

        df = df[len(new_df):]
        csv_exists = len(new_df) > 0

        if len(new_df) != len(df):
            config = Config(model_name=model_name, dataset_name=task)
            new_df = respond(df, config, main_dir, output_csv, csv_exists)
       
        completed_data = get_correctness(new_df)
        with pd.ExcelWriter(file_name) as writer:
            completed_data.to_excel(writer, sheet_name='correctness', index=False)

    print("Done !")


if __name__ == "__main__":
    main()