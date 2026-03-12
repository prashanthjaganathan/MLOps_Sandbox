from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    experiment_name = "medqa-bench"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment {experiment_name} not found")

    runs = mlflow.search_runs(experiment.experiment_id)
    if runs.empty:
        print("No runs found.")
        return

    df = pd.DataFrame(
        {
            "provider": runs["params.provider"],
            "model_id": runs["params.model_id"],
            "strategy": runs["params.strategy_name"],
            "accuracy": runs["metrics.accuracy"],
            "latency_p50": runs["metrics.avg_latency_ms"],
            "latency_p95": runs["metrics.p95_latency_ms"],
            "judge_faithfulness": runs.get("metrics.judge_faithfulness_score"),
        }
    )

    out_dir = PROJECT_ROOT / "analysis_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy vs latency
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="latency_p50",
        y="accuracy",
        hue="provider",
        style="strategy",
        s=80,
    )
    plt.xlabel("Average latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Latency")
    plt.tight_layout()
    acc_lat_path = out_dir / "accuracy_vs_latency.png"
    plt.savefig(acc_lat_path)
    plt.close()

    # Strategy comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df,
        x="strategy",
        y="accuracy",
        hue="provider",
    )
    plt.ylabel("Accuracy")
    plt.title("Strategy comparison")
    plt.tight_layout()
    strat_path = out_dir / "strategy_comparison.png"
    plt.savefig(strat_path)
    plt.close()

    # Log a parent summary run
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="comparison_summary") as run:
        mlflow.log_artifact(str(acc_lat_path))
        mlflow.log_artifact(str(strat_path))
        df.to_csv(out_dir / "runs_summary.csv", index=False)
        mlflow.log_artifact(str(out_dir / "runs_summary.csv"))

    print(f"Wrote analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()

