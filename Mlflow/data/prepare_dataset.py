import json
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "medqa_eval_10.jsonl"


def normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw MedQA example into a flat dict we can use everywhere.

    Expected MedQA fields (from GBaker/MedQA-USMLE-4-options):
      - question: str
      - options: dict-like with keys 'A', 'B', 'C', 'D'
      - answer: could be index, letter, or full text
    """
    options = example.get("options") or {}

    raw_answer = example.get("answer", "")
    # Map integer index -> letter if needed
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    if raw_answer in answer_map:
        answer_letter = answer_map[raw_answer]
    else:
        raw_str = str(raw_answer).strip()
        upper = raw_str.upper()
        if upper in {"A", "B", "C", "D"}:
            answer_letter = upper
        else:
            # Answer is likely the full text. Try to match it to one of the options.
            answer_letter = ""
            for letter, opt_text in options.items():
                if str(opt_text).strip() == raw_str:
                    answer_letter = letter
                    break

    return {
        "question": example.get("question", "").strip(),
        "option_a": options.get("A", "").strip(),
        "option_b": options.get("B", "").strip(),
        "option_c": options.get("C", "").strip(),
        "option_d": options.get("D", "").strip(),
        "answer": answer_letter,
    }


def main(sample_size: int = 10, seed: int = 42) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MedQA-USMLE-4-options from HuggingFace...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

    if sample_size and sample_size < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample_size))

    print(f"Sampling {len(ds)} examples and normalizing...")
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ex in ds:
            norm = normalize_example(ex)
            json.dump(norm, f)
            f.write("\n")

    print(f"Wrote {len(ds)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

