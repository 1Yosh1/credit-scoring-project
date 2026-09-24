"""Reproducible training pipeline for the credit scoring model.

Reproduces the original notebook pipeline (SMOTE balancing, XGBoost) over the
eight applicant-facing features the serving form can actually collect, with
these additions:

- an explicit stratified train/test split with a fixed seed,
- a fitted StandardScaler exported for serving,
- metrics + parameters exported to models/metrics.json,
- optional MLflow tracking (skipped gracefully if no server is available).

Usage:
    python scripts/train.py --data-path data/application_train.csv
    python scripts/train.py --data-path data/application_train.csv --sample-frac 0.2  # quick run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Allow running as `python scripts/train.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_DIR  # noqa: E402
from src.features import MODEL_FEATURES  # noqa: E402

TARGET = "TARGET"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CALIB_SIZE = 0.25  # fraction of the training split held out for calibration

MODEL_VERSION = "1.1.0"


def load_data(path: Path, sample_frac: float | None) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}")

    if TARGET not in df.columns:
        raise SystemExit(f"Column '{TARGET}' not found -- is this application_train.csv?")

    # Aggregate the three bureau scores into one feature (mean of available
    # scores). At serving time the form's single score *is* this aggregate.
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

    # Financial ratios -- computed here so training and serving share formulas.
    df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    if sample_frac and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=RANDOM_STATE)
        print(f"Sampled down to: {df.shape}")

    return df


def balance_with_smote(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("imbalanced-learn not installed -- training on imbalanced data.")
        return X_train, y_train

    print("Applying SMOTE ... (this can take a few minutes on the full dataset)")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"Resampled: {X_train.shape} -> {X_res.shape}")
    return X_res, y_res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/application_train.csv"))
    parser.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Optional fraction of rows to sample for quick experiments")
    parser.add_argument("--skip-smote", action="store_true", help="Train on the imbalanced data")
    parser.add_argument("--target-flag-rate", type=float, default=0.25,
                        help="Fraction of applicants the calibrated threshold should flag (default 0.25)")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking server URI (e.g. http://127.0.0.1:5000)")
    parser.add_argument("--experiment-name", default="Credit_Risk_Scoring")
    args = parser.parse_args()

    df = load_data(args.data_path, args.sample_frac)
    X = df[MODEL_FEATURES].copy()
    # Median-fill the few gaps (12 rows missing AMT_ANNUITY; ~20% missing all
    # bureau scores) so "missing" doesn't read as an extreme value. The serving
    # form always supplies complete input, so this path is training-only.
    X = X.fillna(X.median())
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Carve a calibration set out of the training data (the model never fits
    # on it) so the serving threshold is chosen out-of-sample, not in-sample.
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=CALIB_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )
    print(
        f"Fit: {X_fit.shape} | Calib: {X_calib.shape} | Test: {X_test.shape}"
        f" | Default rate: {y.mean():.1%}"
    )

    # Standardize continuous features; the fitted scaler is exported so serving
    # applies an identical transform (see src/features.py).
    print("Fitting StandardScaler ...")
    scaler = StandardScaler()
    X_fit_scaled = pd.DataFrame(scaler.fit_transform(X_fit), columns=X_fit.columns)
    X_calib_scaled = pd.DataFrame(scaler.transform(X_calib), columns=X_calib.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    if not args.skip_smote:
        X_train_final, y_train_final = balance_with_smote(X_fit_scaled, y_fit)
    else:
        X_train_final, y_train_final = X_fit_scaled, y_fit

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    print("Training XGBoost ...")
    model = XGBClassifier(**params)
    model.fit(X_train_final, y_train_final)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    # Calibrate the serving threshold on the held-out calibration split: pick
    # the cutoff that flags the riskiest `--target-flag-rate` of applicants.
    # This makes the review-queue size a business input instead of a magic
    # constant (SMOTE compresses raw probabilities, so a fixed 0.20 cutoff is
    # not transferable between models).
    val_prob = model.predict_proba(X_calib_scaled)[:, 1]
    threshold = float(np.quantile(val_prob, 1 - args.target_flag_rate))

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
    }

    # Business metrics on the untouched test set, evaluated exactly the way
    # the API deploys the model.
    flagged = y_prob > threshold
    metrics["threshold"] = threshold
    metrics["target_flag_rate"] = args.target_flag_rate
    metrics["flag_rate"] = float(flagged.mean())
    metrics["flagged_recall"] = float(recall_score(y_test, flagged))
    metrics["flagged_precision"] = float(precision_score(y_test, flagged))
    print(
        f"Calibrated threshold {threshold:.3f}: flags {metrics['flag_rate']:.1%} of test applicants, "
        f"catches {metrics['flagged_recall']:.1%} of defaults "
        f"({metrics['flagged_precision']:.1%} of flags are true defaults)"
    )
    print(f"Test metrics: {metrics}")

    # --- Export artifacts for serving ---
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "credit_model.pkl")
    joblib.dump(list(X.columns), out_dir / "model_columns.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_training_rows": int(len(X_train_final)),
        "n_test_rows": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "balanced_with_smote": not args.skip_smote,
        "params": params,
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metadata, indent=2))
    print(f"Artifacts written to {out_dir}/")

    # --- Optional MLflow tracking (never fails the run) ---
    if args.tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(args.tracking_uri)
            mlflow.set_experiment(args.experiment_name)
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_param("smote", not args.skip_smote)
                mlflow.xgboost.log_model(model, name="model")
            print(f"Logged to MLflow at {args.tracking_uri}")
        except Exception as exc:  # noqa: BLE001
            print(f"MLflow logging skipped ({exc}); artifacts are saved locally.")

    print("Done.")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"Completed in {time.time() - started:.0f}s")
