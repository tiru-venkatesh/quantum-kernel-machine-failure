#!/usr/bin/env python3
"""Reproducible Qiskit quantum-kernel experiment on the UCI AI4I 2020 dataset.

This script downloads the dataset through ucimlrepo unless --csv is supplied.
It uses a case-control subset, a fixed stratified holdout, and a Qiskit
StatevectorSampler with finite shots. It does not use physical quantum hardware.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
DATASET_ID = 601
DATASET_DOI = "10.24432/C5HS5C"
DATASET_URL = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+"


def norm_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_column(df, expected):
    """Find a dataset column, allowing optional units and minor naming differences."""
    import re

    def normalize(name):
        name = str(name).strip().lower()
        name = re.sub(r"\[[^\]]*\]", "", name)
        name = re.sub(r"[^a-z0-9]+", "", name)
        return name

    expected_normalized = normalize(expected)
    columns = list(df.columns)

    # Prefer an exact normalized match.
    for column in columns:
        if normalize(column) == expected_normalized:
            return column

    # Fall back to a unique partial match.
    matches = [
        column for column in columns
        if expected_normalized in normalize(column)
        or normalize(column) in expected_normalized
    ]

    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Required column '{expected}' not found. "
        f"Available columns: {columns}. Matches: {matches}"
    )

def load_data(csv_path: str | None) -> tuple[pd.DataFrame, np.ndarray, str]:
    if csv_path:
        raw = pd.read_csv(csv_path)
        source = f"Local CSV: {Path(csv_path).resolve()}"
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise RuntimeError(
                "ucimlrepo is not installed. Run: pip install -r requirements.txt"
            ) from exc
        print("Downloading UCI dataset 601 via ucimlrepo ...", flush=True)
        dataset = fetch_ucirepo(id=DATASET_ID)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()
        raw = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
        source = f"UCI Machine Learning Repository, dataset {DATASET_ID}"

    air = pd.to_numeric(raw[find_column(raw, "Air temperature [K]")], errors="coerce")
    process = pd.to_numeric(raw[find_column(raw, "Process temperature [K]")], errors="coerce")
    speed = pd.to_numeric(raw[find_column(raw, "Rotational speed [rpm]")], errors="coerce")
    torque = pd.to_numeric(raw[find_column(raw, "Torque [Nm]")], errors="coerce")
    wear = pd.to_numeric(raw[find_column(raw, "Tool wear [min]")], errors="coerce")
    target = pd.to_numeric(raw[find_column(raw, "Machine failure")], errors="coerce")

    X = pd.DataFrame({
        "temperature_difference_K": process - air,
        "rotational_speed_rpm": speed,
        "torque_Nm": torque,
        "tool_wear_min": wear,
    })
    y = target
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Expected binary machine-failure labels 0/1; got {np.unique(y)}")
    return X, y, source


def case_control_sample(X: pd.DataFrame, y: np.ndarray, per_class: int, seed: int):
    """Select equal numbers of failure and non-failure rows for a controlled study."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if min(len(pos), len(neg)) < per_class:
        raise ValueError(
            f"Only {len(pos)} failures and {len(neg)} non-failures; reduce --per-class."
        )
    chosen_pos = rng.choice(pos, size=per_class, replace=False)
    chosen_neg = rng.choice(neg, size=per_class, replace=False)
    indices = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(indices)
    return X.iloc[indices].reset_index(drop=True), y[indices]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "failure_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "failure_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "failure_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist(),
    }


def to_quantum_angles(X_scaled: np.ndarray) -> np.ndarray:
    """Bound standardized features smoothly to (0, pi) for rotation angles."""
    z = np.clip(np.asarray(X_scaled, dtype=float), -8.0, 8.0)
    return np.pi / (1.0 + np.exp(-z))


def save_plots(results_dir: Path, model_results: dict, y_test: np.ndarray, predictions: dict,
               train_kernel: np.ndarray):
    names = list(model_results)
    f1s = [model_results[n]["failure_f1"] for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(names, f1s)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Failure-class F1 score")
    ax.set_title("Held-out test comparison")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(predictions), figsize=(4 * len(predictions), 3.8))
    if len(predictions) == 1:
        axes = [axes]
    for ax, (name, pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Greys")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["Normal", "Failure"])
        ax.set_yticks([0, 1], ["Normal", "Failure"])
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(train_kernel, cmap="viridis", aspect="auto")
    ax.set_title("Training quantum-kernel matrix")
    ax.set_xlabel("Training sample index")
    ax.set_ylabel("Training sample index")
    fig.colorbar(im, ax=ax, label="Estimated fidelity")
    fig.tight_layout()
    fig.savefig(results_dir / "quantum_kernel_matrix.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None, help="Local ai4i2020.csv; omit to fetch from UCI.")
    parser.add_argument("--per-class", type=int, default=120,
                        help="Number of failure and non-failure records in the case-control sample (default: 120 each).")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--shots", type=int, default=1024, help="Finite shots per quantum fidelity estimate.")
    parser.add_argument("--reps", type=int, default=2, help="ZZ feature-map repetitions.")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.per_class < 20:
        parser.error("--per-class should be at least 20 for a meaningful small demonstration.")
    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test-size must be between 0.1 and 0.4.")
    if args.shots < 128:
        parser.error("--shots must be at least 128.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    X_all, y_all, source = load_data(args.csv)
    X, y = case_control_sample(X_all, y_all, args.per_class, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Fit every transform on the training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_q = to_quantum_angles(X_train_scaled)
    X_test_q = to_quantum_angles(X_test_scaled)

    # Classical models use exactly the same four selected input features.
    classical_models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        ]),
        "RBF-SVM": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=args.seed, n_jobs=-1,
        ),
    }

    model_results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    score_vectors: dict[str, np.ndarray] = {}
    classical_runtime = {}
    for name, model in classical_models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)
        pred = model.predict(X_test)
        model_results[name] = metrics(y_test, pred, scores)
        predictions[name] = pred
        score_vectors[name] = np.asarray(scores)
        classical_runtime[name] = {"fit_seconds": fit_s}

    # Qiskit: an explicit quantum circuit feature map and finite-shot statevector sampler.
    try:
        from qiskit.circuit.library import zz_feature_map
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
    except ImportError as exc:
        raise RuntimeError(
            "Qiskit dependencies are missing. Install them with: pip install -r requirements.txt"
        ) from exc

    feature_map = zz_feature_map(
        feature_dimension=X_train_q.shape[1], reps=args.reps, entanglement="linear"
    )
    sampler = StatevectorSampler(default_shots=args.shots, seed=args.seed)
    fidelity = ComputeUncompute(sampler=sampler)
    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map, fidelity=fidelity, enforce_psd=True
    )

    print(f"Evaluating Qiskit training kernel for {len(X_train_q)} training samples ...", flush=True)
    q0 = time.perf_counter()
    K_train = quantum_kernel.evaluate(x_vec=X_train_q)
    train_kernel_seconds = time.perf_counter() - q0
    print(f"Evaluating Qiskit test-to-training kernel for {len(X_test_q)} test samples ...", flush=True)
    q1 = time.perf_counter()
    K_test = quantum_kernel.evaluate(x_vec=X_test_q, y_vec=X_train_q)
    test_kernel_seconds = time.perf_counter() - q1

    qsvc = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
    q2 = time.perf_counter()
    qsvc.fit(K_train, y_train)
    q_fit_seconds = time.perf_counter() - q2
    q_scores = qsvc.decision_function(K_test)
    q_pred = qsvc.predict(K_test)
    model_results["Qiskit Quantum Kernel SVM"] = metrics(y_test, q_pred, q_scores)
    predictions["Qiskit Quantum Kernel SVM"] = q_pred
    score_vectors["Qiskit Quantum Kernel SVM"] = q_scores

    # Save only the sampled analysis subset, not the full source file.
    sampled = X.copy()
    sampled["machine_failure"] = y
    sampled.to_csv(output / "case_control_sample.csv", index=False)
    pd.DataFrame({"actual": y_test, **{f"pred_{k.lower().replace(' ', '_')}": v for k, v in predictions.items()}}).to_csv(
        output / "test_predictions.csv", index=False
    )
    pd.DataFrame.from_dict(model_results, orient="index").to_csv(output / "model_metrics.csv")
    (output / "classification_reports.txt").write_text(
        "\n\n".join(
            f"{name}\n{classification_report(y_test, pred, target_names=['Normal', 'Failure'], zero_division=0)}"
            for name, pred in predictions.items()
        ), encoding="utf-8"
    )
    (output / "feature_map_circuit.txt").write_text(str(feature_map.draw(output="text")), encoding="utf-8")

    try:
        from qiskit import qpy
        with (output / "zz_feature_map.qpy").open("wb") as f:
            qpy.dump([feature_map], f)
    except Exception as exc:
        (output / "qpy_export_note.txt").write_text(f"Optional QPY export skipped: {exc}\n", encoding="utf-8")

    save_plots(output, model_results, y_test, predictions, K_train)

    import qiskit
    import qiskit_machine_learning
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "dataset_source": source,
        "dataset_id": DATASET_ID,
        "dataset_doi": DATASET_DOI,
        "dataset_license": "CC BY 4.0; attribution required",
        "uci_dataset_is_synthetic": True,
        "uci_dataset_rows_loaded": int(len(X_all)),
        "uci_failure_count_loaded": int((y_all == 1).sum()),
        "sample_design": "Case-control: equal number of failure and non-failure rows sampled without replacement",
        "per_class_sampled": int(args.per_class),
        "sampled_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "selected_features": list(X.columns),
        "excluded_features": ["UID", "Product ID", "Type", "individual failure-mode labels"],
        "target": "Machine failure",
        "split_seed": args.seed,
        "test_size": args.test_size,
        "feature_map": "Qiskit ZZFeatureMap, linear entanglement",
        "qubits": int(X_train_q.shape[1]),
        "feature_map_reps": args.reps,
        "sampler": "Qiskit StatevectorSampler (finite-shot simulation)",
        "shots_per_fidelity_estimate": args.shots,
        "physical_qpu_used": False,
        "train_kernel_matrix_seconds": train_kernel_seconds,
        "test_kernel_matrix_seconds": test_kernel_seconds,
        "quantum_svm_fit_seconds": q_fit_seconds,
        "classical_fit_seconds": classical_runtime,
        "software": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "qiskit": qiskit.__version__, "qiskit_machine_learning": qiskit_machine_learning.__version__,
        },
        "results": model_results,
        "interpretation_warning": (
            "Metrics are measured on a balanced case-control subset, not at the original 3.39% failure prevalence. "
            "Precision and accuracy therefore should not be treated as deployment estimates. "
            "The UCI dataset is synthetic; this is not field-validated industrial evidence. "
            "No quantum advantage or hardware advantage is claimed."
        ),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_lines = [
        "QML experiment run completed",
        f"Dataset: {source}",
        f"UCI rows loaded: {len(X_all)}; failures: {(y_all == 1).sum()}",
        f"Case-control subset: {len(X)} rows ({args.per_class} per class)",
        f"Train/test: {len(X_train)}/{len(X_test)}",
        f"Quantum: {args.reps}-repetition ZZFeatureMap, {X_train_q.shape[1]} qubits, {args.shots} shots",
        "Physical quantum processor used: NO",
        "",
    ]
    for name, row in model_results.items():
        summary_lines.append(
            f"{name}: accuracy={row['accuracy']:.3f}, balanced_accuracy={row['balanced_accuracy']:.3f}, "
            f"failure_precision={row['failure_precision']:.3f}, failure_recall={row['failure_recall']:.3f}, "
            f"failure_F1={row['failure_f1']:.3f}, ROC_AUC={row['roc_auc']:.3f}, AP={row['average_precision']:.3f}"
        )
    summary_lines += [
        "",
        "Caution: case-control sampling changes the original prevalence; precision/accuracy are not deployment estimates.",
        "Caution: UCI identifies AI4I 2020 as a synthetic dataset, not direct factory measurements.",
        "Caution: this run uses a simulator, not physical quantum hardware; no quantum advantage is established.",
    ]
    (output / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nSaved results to: {output.resolve()}")


if __name__ == "__main__":
    main()
