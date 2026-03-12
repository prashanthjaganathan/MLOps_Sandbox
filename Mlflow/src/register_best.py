from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.pyfunc import PythonModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


class MedQAModel(PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:  # type: ignore[name-defined]
        import yaml  # local import to ensure availability at runtime

        cfg_path = context.artifacts["config"]
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def predict(self, context, model_input: pd.DataFrame):  # type: ignore[override]
        """
        model_input columns:
          - question
          - option_a
          - option_b
          - option_c
          - option_d
        """
        from .prompt_builder import PromptConfig, build_messages
        from .providers import call_model

        cfg_dict: Dict[str, Any] = self.config
        prompt_cfg = PromptConfig(
            name=cfg_dict["strategy_name"],
            system_prompt=cfg_dict["system_prompt"],
            user_template=cfg_dict["user_template"],
            temperature=float(cfg_dict["temperature"]),
            max_tokens=int(cfg_dict["max_tokens"]),
            top_p=float(cfg_dict["top_p"]),
            few_shot_examples=list(cfg_dict.get("few_shot_examples", [])),
        )

        provider = cfg_dict["provider"]
        model_id = cfg_dict["model_id"]

        outputs = []
        for _, row in model_input.iterrows():
            ex = {
                "question": row["question"],
                "option_a": row["option_a"],
                "option_b": row["option_b"],
                "option_c": row["option_c"],
                "option_d": row["option_d"],
            }
            messages = build_messages(ex, prompt_cfg)
            resp = call_model(
                provider=provider, model=model_id, messages=messages, temperature=prompt_cfg.temperature
            )
            outputs.append(resp["text"])
        return np.array(outputs)


def main() -> None:
    experiment_name = "medqa-bench"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment {experiment_name} not found")

    runs = mlflow.search_runs(experiment.experiment_id)
    if runs.empty:
        raise RuntimeError("No runs to select from.")

    # Compute composite score
    acc = runs["metrics.accuracy"].fillna(0.0)
    lat = runs["metrics.avg_latency_ms"].fillna(acc.max() or 1.0)
    judge = runs.get("metrics.judge_faithfulness_score", pd.Series([0.0] * len(runs))).fillna(0.0)

    # Normalize latency so lower is better
    lat_norm = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
    score = 0.6 * acc + 0.2 * (1 - lat_norm) + 0.2 * (judge / 5.0)

    best_idx = int(score.idxmax())
    best_run = runs.loc[best_idx]
    best_run_id = best_run["run_id"]
    print(f"Best run: {best_run_id}")

    # Build config artifact from best run + prompt config artifact we logged
    artifacts_dir = PROJECT_ROOT / "artifacts" / best_run_id
    prompt_cfg_path = artifacts_dir / "prompt_config.yaml"
    with prompt_cfg_path.open("r", encoding="utf-8") as f:
        prompt_cfg_raw = yaml.safe_load(f)

    best_cfg: Dict[str, Any] = {
        "provider": best_run["params.provider"],
        "model_id": best_run["params.model_id"],
        "strategy_name": best_run["params.strategy_name"],
        "temperature": best_run["params.temperature"],
        "max_tokens": best_run["params.max_tokens"],
        "top_p": best_run["params.top_p"],
        "system_prompt": prompt_cfg_raw["system_prompt"],
        "user_template": prompt_cfg_raw["user_template"],
        "few_shot_examples": prompt_cfg_raw.get("few_shot_examples", []),
    }

    cfg_out_dir = PROJECT_ROOT / "registry_artifacts"
    cfg_out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_out_dir / "winning_config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="register_best", tags={"stage": "register"}) as run:
        mlflow.pyfunc.log_model(
            "medqa_model",
            python_model=MedQAModel(),
            artifacts={"config": str(cfg_path)},
            pip_requirements=[
                "mlflow",
                "openai",
                "anthropic",
                "google-genai",
                "groq",
                "pyyaml",
                "pandas",
                "numpy",
                "tenacity",
            ],
        )
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/medqa_model"

    print(f"Logged PyFunc model at {model_uri}")
    result = mlflow.register_model(model_uri, "medqa_best")
    print(f"Registered model name=medqa_best, version={result.version}")


if __name__ == "__main__":
    main()

