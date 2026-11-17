from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
import json
import os
import re
import ast
from pathlib import Path
from config import Config
from utils import *
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, TypedDict, Literal

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 


class Validator(BaseModel):
    Response: str = Field(description="The Response.")
    Confidence: Literal["very high", "high", "low", "very low"] = Field(description="The confidence.")

def prepare_prompt(text: str, pronoun: str, pronoun_sentence:str, example_text: str, example_pronoun: str, example_antecedent: str) -> str:
    """
    Prepares a prompt for inference.

    Args:
        text: Clinical text
        pronoun: Pronoun to be processed
        pronoun_sentence: Sentence containing the pronoun
        example_text: Example text
        example_pronoun: Example pronoun
        example_antecedent: Example antecedent

    Returns:
        The formatted prompt
    """
    
    prompt = f"""
    ## Examples
    {example_text}

    {example_pronoun} refers to {example_antecedent}
    
    -----------------
    ## Task
    {text}

    What does {pronoun} in the sentence {pronoun_sentence} refer to?
    -----------------
    Output format: 
    Return strictly a JSON object with the following fields:
    - Response: [Write your answer here. Return strictly the exact value. Do not include any additional text, formatting, or explanation.]
    - Confidence: [Write your confidence score here. Provide one of the following confidence scores measuring your level of confidence regarding your answer: very high, high, low, very low.]

    """
    

    return prompt



def respond(df, df_example, config):
    columns = ["snippet", "pronoun", "antecedents", "pronoun_sentence"]
    
    df = df.reset_index(drop=True)
    new_df = df[columns].copy()
    if isinstance(new_df, pd.Series):
        new_df = new_df.to_frame()

    new_df['pred_antecedents'] = None
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

    example_row = df_example.iloc[0]

    for i in range(df.shape[0]):
        print(f"Processing row {i+1} / {len(df)}")
        row_values = new_df.iloc[i]
        prompt = prepare_prompt(text=row_values["snippet"], 
                                pronoun=row_values["pronoun"],
                                pronoun_sentence=row_values["pronoun_sentence"],
                                example_text=example_row["snippet"], 
                                example_pronoun=example_row["pronoun"], 
                                example_antecedent=example_row["pronoun_sentence"]
                               )
        
        try:

            messages = [
                (
                    "system",
                    "You are a helpful assistant in Coreference Resolution. Think step by step before answering. Write your reasoning inside <thinking> tags before giving the final answer.",
                ),
                ("human", prompt),
            ]
            response = llm.with_structured_output(Validator, method="json_schema", include_raw=True).invoke(messages)
            raw_text = response["raw"].content

            if not raw_text :
                new_df.loc[i, ['pred_antecedents', 'verbal_confidence', 'value_uncertainty']] = [
                    None,
                    None,
                    None
                ]
            else :

                match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
                raw_text = match.group(1)
                parsed = json.loads(raw_text)
                logprobs = response["raw"].response_metadata["logprobs"]
                field_tokens = align_tokens_to_fields(logprobs, parsed, field_names=["Response", "Confidence"])
                
                value_tokens = filter_structural(field_tokens["Response"])
                metrics_value = uncertainty_metrics(value_tokens)
    
                new_df.loc[i, ['pred_antecedents', 'verbal_confidence', 'value_uncertainty']] = [
                    parsed["Response"],
                    parsed["Confidence"],
                    metrics_value
                ]
            
        except Exception as e:
            print(f"Error occurred at row {i+1}")
            raise
                

    return new_df
    
def get_correctness(data):

    rows = []

    for index, row in data.iterrows():
        target = row["antecedents"]
        pred = row["pred_antecedents"]

        if isinstance(target, str):
            target = ast.literal_eval(target)

        if isinstance(pred, str):
            pred = [pred]

        correctness_bool = compute_f1_score(pred, target, f1_threshold = 0.75)
        correctness = 1 if correctness_bool == True else 0

        unc = extract_unc(row)

        rows.append({
            "snippet": row["snippet"], 
            "pronoun": row["pronoun"], 
            "pronoun_sentence": row["pronoun_sentence"], 
            "antecedents": row["antecedents"], 
            "pred_antecedents": pred,
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
        "snippet", "pronoun", "pronoun_sentence", "antecedents", "pred_antecedents", "verbal_confidence", "correctness", "perplexity",
        "SP", "LNSP", "MTP", "PM", "ATN@5", "MTN@5",
    ])

def main():
    task = "coreference_resolution"
    models = ["model_name"]
    main_dir = os.path.dirname(ROOT_DIR)
    
    output_dir = Path(main_dir + "/data/baselines/" + task + "/with_reasoning")
    data_file =  main_dir + "/data/baselines/coreference_test.csv"
    example_file =  main_dir + "/data/baselines/coreference_dev.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    df_example = pd.read_csv(example_file)
    df = pd.concat([df, df_example], ignore_index=True)

    for model_name in models:
        output_excel = main_dir + "/data/baselines/" + task + f"/with_reasoning/results_{model_name}_{task}.xlsx"
        file_name = main_dir + "/data/baselines/" + task + "/with_reasoning/correctness_" + model_name + "_" + task + ".xlsx"
        config = Config(model_name=model_name, dataset_name=task)
        new_df = respond(df, df_example, config)
        new_df.to_excel(output_excel, index=False)
        completed_data = get_correctness(new_df)
        with pd.ExcelWriter(file_name) as writer:
            completed_data.to_excel(writer, sheet_name='correctness', index=False)

    print("Done !")

if __name__ == "__main__":
    main()