"""
Small helper to demonstrate loading the registered MedQA model locally.

Example:
    python -m src.serve_model
"""

import pandas as pd

import mlflow


def main() -> None:
    model_uri = "models:/medqa_best/production"
    model = mlflow.pyfunc.load_model(model_uri)

    df = pd.DataFrame(
        [
            {
                "question": "A 65-year-old man presents with chest pain radiating to his left arm...",
                "option_a": "Stable angina",
                "option_b": "Myocardial infarction",
                "option_c": "Pericarditis",
                "option_d": "Costochondritis",
            }
        ]
    )
    preds = model.predict(df)
    print(preds)


if __name__ == "__main__":
    main()

