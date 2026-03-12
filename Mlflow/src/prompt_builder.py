from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class PromptConfig:
    name: str
    system_prompt: str
    user_template: str
    temperature: float
    max_tokens: int
    top_p: float
    few_shot_examples: List[Dict[str, Any]]


def load_prompt_config(filename: str) -> PromptConfig:
    path = CONFIG_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PromptConfig(
        name=str(raw.get("strategy")),
        system_prompt=str(raw.get("system_prompt", "")),
        user_template=str(raw.get("user_template", "")),
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 256)),
        top_p=float(raw.get("top_p", 1.0)),
        few_shot_examples=list(raw.get("few_shot_examples", [])),
    )


def build_messages(example: Dict[str, Any], cfg: PromptConfig) -> List[Dict[str, str]]:
    """
    Build chat messages for a single MedQA example given a prompt config.
    """
    messages: List[Dict[str, str]] = []

    if cfg.system_prompt:
        messages.append({"role": "system", "content": cfg.system_prompt})

    # Few-shot exemplars, if any
    for ex in cfg.few_shot_examples:
        q = ex.get("question", "")
        opts = ex.get("options", {})
        answer = ex.get("answer", "")
        text = (
            f"Example question: {q}\n"
            f"A) {opts.get('A', '')}\n"
            f"B) {opts.get('B', '')}\n"
            f"C) {opts.get('C', '')}\n"
            f"D) {opts.get('D', '')}\n"
            f"Correct answer: {answer}"
        )
        messages.append({"role": "user", "content": text})

    # Actual question
    filled = cfg.user_template.format(
        question=example["question"],
        option_a=example["option_a"],
        option_b=example["option_b"],
        option_c=example["option_c"],
        option_d=example["option_d"],
    )
    messages.append({"role": "user", "content": filled})
    return messages

