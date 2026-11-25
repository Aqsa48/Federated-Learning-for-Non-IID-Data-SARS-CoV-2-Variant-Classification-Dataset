"""Dataset checks and an explicitly centralized, reproducible reference baseline.

This does not reproduce a federated-learning experiment or provide privacy guarantees.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = {"alpha", "beta", "delta", "omicron"}


def validate(frame):
    if not {"SequenceID", "Variant"}.issubset(frame.columns):
        raise ValueError("Each client needs SequenceID and Variant columns")
    if frame.empty or frame.SequenceID.isna().any() or frame.SequenceID.astype(str).str.strip().eq("").any():
        raise ValueError("Client must contain rows with nonempty SequenceID values")
    if frame.SequenceID.duplicated().any():
        raise ValueError("SequenceID must be unique within each client")
    frame = frame.copy()
    if frame.Variant.isna().any():
        raise ValueError("Variant labels cannot be missing")
    frame["Variant"] = frame.Variant.astype(str).str.strip().str.lower()
    if not set(frame.Variant).issubset(LABELS):
        raise ValueError("Variant must be alpha, beta, delta, or omicron")
    features = frame.drop(columns=["SequenceID", "Variant"])
    if features.empty or len(features.columns) == 0:
        raise ValueError("No numeric features found")
    for name in features:
        try:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric feature: {name}; remove metadata columns before training") from exc
        if np.isinf(frame[name].to_numpy(dtype=float)).any() or frame[name].isna().all():
            raise ValueError(f"Feature {name} has infinity or no observed values")
    return frame


def demo_clients(seed=42):
    rng = np.random.default_rng(seed)
    labels = sorted(LABELS)
    clients = {}
    for client in range(4):
        # Deliberately different class distributions between synthetic clients.
        y = rng.choice(4, 120, p=np.roll([.55, .25, .15, .05], client))
        x = np.eye(4)[y] * 4 + rng.normal(0, .5, (120, 4))
        frame = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(4)])
        frame["SequenceID"] = [f"synthetic-{client}-{i}" for i in range(120)]
        frame["Variant"] = [labels[i] for i in y]
        clients[f"synthetic-client-{client+1}"] = frame
    return clients


def load_clients(folder):
    paths = sorted([*Path(folder).glob("*.csv"), *Path(folder).glob("*.csv.zip")])
    if not paths:
        raise ValueError(f"No CSV or single-CSV ZIP files in {folder}")
    clients = {}
    for path in paths:
        name = path.name.removesuffix(".zip").removesuffix(".csv")
        if name in clients:
            raise ValueError(f"Duplicate client file: {name}; keep CSV or ZIP, not both")
        clients[name] = pd.read_csv(path)
    return clients


def run(clients, seed=42, validate_only=False):
    clients = {name: validate(frame) for name, frame in clients.items()}
    if not clients:
        raise ValueError("At least one client is required")
    features = sorted(set(next(iter(clients.values())).columns) - {"SequenceID", "Variant"})
    for name, frame in clients.items():
        if sorted(set(frame.columns) - {"SequenceID", "Variant"}) != features:
            raise ValueError(f"Feature schema differs for {name}")
    report = {"method": "centralized-logistic-regression", "seed": seed,
              "features": features, "clients": {name: {"rows": len(f), "class_counts": f.Variant.value_counts().to_dict(),
              "missing_feature_cells": int(f[features].isna().sum().sum())} for name, f in clients.items()}}
    if validate_only:
        return report
    # Keep identical feature rows in the same partition, even across clients.
    # Sequence IDs are client-local identifiers and never enter the model.
    pooled = pd.concat([frame.assign(_client=name) for name, frame in clients.items()], ignore_index=True)
    groups = pd.util.hash_pandas_object(pooled[features], index=False)
    split = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    train, test = next(split.split(pooled, pooled.Variant, groups))
    if len(set(pooled.iloc[train].Variant)) < 2:
        raise ValueError("Training split needs at least two classes; use a larger dataset")
    unseen = set(pooled.iloc[test].Variant) - set(pooled.iloc[train].Variant)
    if unseen:
        raise ValueError(f"Test labels absent from training split: {sorted(unseen)}")
    missing_train = [f for f in features if pooled.iloc[train][f].isna().all()]
    if missing_train:
        raise ValueError(f"Features entirely missing in training split: {missing_train}")
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(pooled.iloc[train][features], pooled.iloc[train].Variant)
    predicted = model.predict(pooled.iloc[test][features])
    report.update({"train_rows": len(train), "test_rows": len(test),
                   "split": "pooled 75/25 grouped by identical feature rows; not temporal or geographic holdout",
                   "accuracy": float(accuracy_score(pooled.iloc[test].Variant, predicted)),
                   "macro_f1": float(f1_score(pooled.iloc[test].Variant, predicted, average="macro", zero_division=0)),
                   "test_class_counts": pooled.iloc[test].Variant.value_counts().to_dict(),
                   "per_client_test": {}})
    for name in clients:
        mask = pooled.iloc[test]._client.to_numpy() == name
        if mask.any():
            report["per_client_test"][name] = {"rows": int(mask.sum()), "accuracy": float(accuracy_score(pooled.iloc[test].Variant.to_numpy()[mask], predicted[mask]))}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use generated data, never research data")
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("baseline-report.json"))
    args = parser.parse_args(argv)
    try:
        report = run(demo_clients(args.seed) if args.demo else load_clients(args.data_dir), args.seed, args.validate_only)
        report["data"] = "synthetic demonstration; not research results" if args.demo else "user-supplied feature data"
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"baseline: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
