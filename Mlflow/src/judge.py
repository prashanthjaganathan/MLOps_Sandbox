from pathlib import Path
from typing import Any, Dict, List

import json

import mlflow
import pandas as pd

from .providers import ProviderName, call_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_judge_prompt(row: Dict[str, Any]) -> str:
    question = row["question"]
    options = {
        "A": row["option_a"],
        "B": row["option_b"],
        "C": row["option_c"],
        "D": row["option_d"],
    }
    reasoning = row["raw_text"]
    answer = row["answer"]

    opts_text = "\n".join(f"{k}) {v}" for k, v in options.items())

    return (
        "You are a senior physician evaluating a student's explanation on a USMLE-style question.\n"
        "Question:\n"
        f"{question}\n\n"
        "Options:\n"
        f"{opts_text}\n\n"
        f"Correct answer: {answer}\n\n"
        "Student's reasoning and final answer:\n"
        f"{reasoning}\n\n"
        "Rate on a 1-5 scale:\n"
        "1) Faithfulness: Does the reasoning correctly use clinical facts and support the final answer?\n"
        "2) Clinical reasoning: Are the diagnostic/therapeutic steps logical and medically sound?\n\n"
        "Respond strictly in JSON with keys 'faithfulness' and 'clinical_reasoning', each an integer 1-5."
    )


def parse_judge_scores(text: str) -> Dict[str, float]:
    try:
        data = json.loads(text)
        return {
            "faithfulness": float(data.get("faithfulness", 0.0)),
            "clinical_reasoning": float(data.get("clinical_reasoning", 0.0)),
        }
    except Exception:  # noqa: BLE001
        return {"faithfulness": 0.0, "clinical_reasoning": 0.0}


def score_run_with_judge(
    run_id: str,
    provider: ProviderName,
    model_id: str,
    judge_provider: ProviderName,
    judge_model_id: str,
) -> None:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # Find eval_results.csv in artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts" / run_id
    eval_path = artifacts_dir / "eval_results.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"{eval_path} not found for run {run_id}")

    df = pd.read_csv(eval_path)
    scores: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        prompt = build_judge_prompt(row.to_dict())
        resp = call_model(
            provider=judge_provider,
            model=judge_model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        judge_scores = parse_judge_scores(resp["text"])
        scores.append(judge_scores)

    if scores:
        avg_faith = sum(s["faithfulness"] for s in scores) / len(scores)
        avg_clin = sum(s["clinical_reasoning"] for s in scores) / len(scores)
    else:
        avg_faith = 0.0
        avg_clin = 0.0

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("judge_faithfulness_score", avg_faith)
        mlflow.log_metric("judge_reasoning_score", avg_clin)


def main() -> None:
    """
    Iterate over all runs in the medqa-bench experiment and add judge scores
    for runs that don't have them yet.
    """
    experiment_name = "medqa-bench"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment {experiment_name} not found")

    runs = mlflow.search_runs(experiment.experiment_id)
    for _, row in runs.iterrows():
        run_id = row["run_id"]
        if not pd.isna(row.get("metrics.judge_faithfulness_score", float("nan"))):
            continue

        provider = row["params.provider"]
        model_id = row["params.model_id"]
        print(f"Scoring run {run_id} ({provider} / {model_id}) with judge...")
        score_run_with_judge(
            run_id=run_id,
            provider=provider,  # type: ignore[arg-type]
            model_id=model_id,
            judge_provider="openai",
            judge_model_id="gpt-4o-mini",
        )


if __name__ == "__main__":
    main()

