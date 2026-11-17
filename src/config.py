from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    model_name: str = ""
    temperature: float = 1.0
    top_p: float = 1.0
    api_key=""
    base_url=""
    max_retries=2
    timeout=300
    logprobs=True
    top_logprobs=5
    reasoning=True
    max_tokens: int = 2048
    # Dataset Related Options
    dataset_name: str = ""
    output_dir = ""