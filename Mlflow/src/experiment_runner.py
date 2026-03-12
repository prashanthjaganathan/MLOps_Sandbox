import json
from pathlib import Path
from statistics import mean
import time
from typing import Any, Dict, Iterable, List, Tuple
from tenacity import RetryError

import mlflow
import pandas as pd
import yaml

from .prompt_builder import PromptConfig, build_messages, load_prompt_config
from .providers import ProviderName, call_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "medqa_eval_10.jsonl"
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_dataset() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python -m data.prepare_dataset` first."
        )
    records: List[Dict[str, Any]] = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def iter_models_from_yaml() -> Iterable[Tuple[ProviderName, str]]:
    cfg_path = CONFIG_DIR / "models.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    providers = raw.get("providers", {})
    for provider_name, p_cfg in providers.items():
        models = p_cfg.get("models", [])
        for m in models:
            yield provider_name, m["id"]


def run_single(
    provider: ProviderName,
    model_id: str,
    prompt_cfg: PromptConfig,
    dataset: List[Dict[str, Any]],
    experiment_name: str = "medqa-bench",
) -> None:
    mlflow.set_experiment(experiment_name)

    run_name = f"{provider}_{model_id}_{prompt_cfg.name}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Params
        mlflow.log_param("provider", provider)
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("strategy_name", prompt_cfg.name)
        mlflow.log_param("temperature", prompt_cfg.temperature)
        mlflow.log_param("max_tokens", prompt_cfg.max_tokens)
        mlflow.log_param("top_p", prompt_cfg.top_p)

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        total = len(dataset)
        print(
            f"[{provider} / {model_id} / {prompt_cfg.name}] "
            f"Evaluating {total} questions...",
            flush=True,
        )

        for idx, ex in enumerate(dataset):
            messages = build_messages(ex, prompt_cfg)
            try:
                print('Example ', idx + 1, 'of', total)
                resp = call_model(
                    provider=provider,
                    model=model_id,
                    messages=messages,
                    temperature=prompt_cfg.temperature,
                    top_p=prompt_cfg.top_p,
                )
                time.sleep(5)
                pred_text = (resp["text"] or "").strip()
                pred_letter = parse_answer_letter(pred_text, prompt_cfg.name)
                correct = str(ex["answer"]).strip().upper()
                is_correct = pred_letter == correct

                results.append(
                    {
                        "idx": idx,
                        "question": ex["question"],
                        "option_a": ex["option_a"],
                        "option_b": ex["option_b"],
                        "option_c": ex["option_c"],
                        "option_d": ex["option_d"],
                        "answer": correct,
                        "predicted_letter": pred_letter,
                        "raw_text": pred_text,
                        "latency_ms": resp["latency_ms"],
                        "provider": provider,
                        "model_id": model_id,
                        "strategy_name": prompt_cfg.name,
                    }
                )
            except Exception as e:  # noqa: BLE001
                underlying = e
                try:
                    # If this is a RetryError from tenacity, grab the last exception it saw
                    from tenacity import RetryError
                    if isinstance(e, RetryError) and e.last_attempt is not None:
                        underlying = e.last_attempt.exception()
                except Exception:
                    pass
                print(
                    f"[ERROR] Provider={provider}, model={model_id}, idx={idx}: {repr(underlying)}",
                    flush=True,
                )
                errors.append(
                    {
                        "idx": idx,
                        "error": repr(underlying),
                        "example": ex,
                    }
                )

            # Simple progress indicator every 10 questions (and at the end)
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(
                    f"[{provider} / {model_id} / {prompt_cfg.name}] "
                    f"Completed {idx + 1}/{total} questions",
                    flush=True,
                )

        # Aggregate metrics
        if results:
            acc = mean(1.0 if r["predicted_letter"] == r["answer"] else 0.0 for r in results)
            latencies = [r["latency_ms"] for r in results]
            avg_latency = mean(latencies)
            p95_latency = percentile(latencies, 95)
        else:
            acc = 0.0
            avg_latency = 0.0
            p95_latency = 0.0

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("avg_latency_ms", avg_latency)
        mlflow.log_metric("p95_latency_ms", p95_latency)
        mlflow.set_tag("provider", provider)
        mlflow.set_tag("strategy_name", prompt_cfg.name)

        # Artifacts
        artifacts_dir = PROJECT_ROOT / "artifacts" / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        results_path = artifacts_dir / "eval_results.csv"
        pd.DataFrame(results).to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path))

        errors_path = artifacts_dir / "errors.json"
        with errors_path.open("w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        mlflow.log_artifact(str(errors_path))

        # Also log the prompt config used
        cfg_dump_path = artifacts_dir / "prompt_config.yaml"
        with cfg_dump_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(prompt_cfg.__dict__, f)
        mlflow.log_artifact(str(cfg_dump_path))


def parse_answer_letter(text: str, strategy_name: str) -> str:
    text = text.strip()
    if not text:
        return ""

    # Chain-of-thought style often ends with "Final Answer: X"
    lowered = text.lower()
    if "final answer" in lowered:
        # Take last character after colon or in string
        after = lowered.split("final answer")[-1]
        for ch in reversed(after):
            if ch.upper() in {"A", "B", "C", "D"}:
                return ch.upper()

    # Otherwise, take first valid letter we see
    for ch in text:
        if ch.upper() in {"A", "B", "C", "D"}:
            return ch.upper()
    return ""


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int(round((p / 100.0) * (len(values_sorted) - 1)))
    return float(values_sorted[k])


def main() -> None:
    dataset = load_dataset()
    prompt_files = [
        "zero_shot_direct.yaml",
        "chain_of_thought.yaml",
        "few_shot_3.yaml",
    ]

    for provider, model_id in iter_models_from_yaml():
        for cfg_file in prompt_files:
            cfg = load_prompt_config(cfg_file)
            print(f"Running {provider} / {model_id} / {cfg.name}")
            run_single(provider, model_id, cfg, dataset)


if __name__ == "__main__":
    main()

