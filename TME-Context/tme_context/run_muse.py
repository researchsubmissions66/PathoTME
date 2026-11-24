"""Train the BRCA CLIP-RN50 TME adapter on real features and a native MUSE checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .core import FoldPreprocessor, TMEContext, load_panel
from .data import fit_fold_preprocessor, load_tme, read_csv
from .muse_model import FrozenMUSE


LABELS = {"IDC": 0, "ILC": 1}


def _metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    if (probabilities.ndim != 2 or probabilities.shape != (len(labels), 2)
            or not len(labels) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5)):
        raise ValueError("invalid binary class probabilities")
    predicted = probabilities.argmax(axis=1)
    recalls, f1s = [], []
    for cls in (0, 1):
        tp = int(((predicted == cls) & (labels == cls)).sum())
        fp = int(((predicted == cls) & (labels != cls)).sum())
        fn = int(((predicted != cls) & (labels == cls)).sum())
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean() * abs((predicted[mask] == labels[mask]).mean()
                                           - confidence[mask].mean()))
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    auroc = None
    if n_pos and n_neg:
        scores = probabilities[:, 1]
        order = np.argsort(scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.float64)
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and scores[order[end]] == scores[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        auroc = float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {"accuracy": float((predicted == labels).mean()),
            "balanced_accuracy": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)),
            "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels],
                                         1e-12, 1.0)).mean()),
            "ece": ece, "per_class_recall": {"0": recalls[0], "1": recalls[1]},
            "auroc_ovr": auroc}


def metric_bundle(predictions: list[dict]) -> dict:
    """Match the study's slide and patient aggregation using mean probabilities."""
    probabilities = np.asarray([[float(row["probability_IDC"]), float(row["probability_ILC"])]
                                for row in predictions], dtype=np.float64)
    labels = np.asarray([LABELS[row["label"]] for row in predictions], dtype=np.int64)
    patient: dict[str, list[int]] = {}
    for index, row in enumerate(predictions):
        patient.setdefault(row["case_id"], []).append(index)
    patient_probabilities, patient_labels = [], []
    for indices in patient.values():
        if len(set(labels[indices])) != 1:
            raise ValueError("one patient has conflicting labels")
        patient_probabilities.append(probabilities[indices].mean(axis=0))
        patient_labels.append(int(labels[indices[0]]))
    return {"slide_metrics": _metrics(probabilities, labels),
            "patient_metrics": _metrics(np.asarray(patient_probabilities),
                                        np.asarray(patient_labels, dtype=np.int64))}


def read_bag(path: Path, device: torch.device) -> torch.Tensor:
    with h5py.File(path, "r") as handle:
        if "features" not in handle or "coords" not in handle:
            raise ValueError(f"missing features or coordinates in {path}")
        feature, coords = handle["features"], handle["coords"]
        if (feature.ndim != 2 or feature.shape[0] < 1 or feature.shape[1] != 1024
                or coords.shape != (feature.shape[0], 2)
                or feature.attrs.get("encoder") != "clip_rn50"
                or float(coords.attrs.get("target_magnification", -1)) != 10
                or int(coords.attrs.get("patch_size", -1)) != 224):
            raise ValueError(f"BRCA CLIP-RN50 feature contract mismatch: {path}")
        array = feature[:]
    bag = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    if bag.ndim != 2 or bag.shape[0] < 1 or bag.shape[1] != 1024 or not torch.isfinite(bag).all():
        raise ValueError(f"invalid 1024-wide CLIP-RN50 feature bag: {path}")
    return bag


def evaluate(rows, paths, tme, preprocessor, native, conditioner, device):
    conditioner.eval()
    losses, correct, predictions = [], 0, []
    with torch.no_grad():
        for row in rows:
            slide = row["slide_id"]
            target = LABELS[row["label"]]
            bag = read_bag(paths[slide], device)
            values = preprocessor(tme[slide].unsqueeze(0).to(device))
            logits = native(bag, conditioner, values)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"nonfinite MUSE logits for slide {slide!r}")
            label = torch.tensor([target], device=device)
            losses.append(float(F.cross_entropy(logits, label)))
            probability = logits.softmax(dim=-1)[0].cpu().tolist()
            predicted = int(probability[1] > probability[0])
            correct += predicted == target
            predictions.append({"slide_id": slide, "case_id": row["case_id"],
                                "label": row["label"], "probability_IDC": probability[0],
                                "probability_ILC": probability[1]})
    return {"loss": sum(losses) / len(losses), "error": 1 - correct / len(rows)}, predictions


def train(args) -> None:
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("--output-dir must be outside the TME-Context source folder")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if args.fold not in range(5):
        raise ValueError("the bundled BRCA protocol has folds 0–4")
    config_path = (args.native_config or args.native_checkpoint.expanduser().parent / "config.json")
    if not config_path.is_file():
        raise FileNotFoundError("native MUSE config.json is required; pass --native-config")
    config = json.loads(config_path.read_text())
    expected = {"task": "brca", "method": "muse", "backbone": "clip-rn50",
                "classnames": ["invasive ductal carcinoma", "invasive lobular carcinoma"],
                "label_dict": LABELS, "feature_dim": 1024, "embed_dim": 1024, "shots": 16}
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if mismatches:
        raise ValueError(f"native checkpoint config differs from BRCA MUSE/CLIP-RN50: {mismatches}")
    checkpoint_name = args.native_checkpoint.name
    if checkpoint_name.startswith("fold") and checkpoint_name.endswith("_best.pt"):
        recorded_fold = checkpoint_name.removeprefix("fold").removesuffix("_best.pt")
        if recorded_fold.isdigit() and int(recorded_fold) != args.fold:
            raise ValueError("native checkpoint filename and requested fold disagree")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + args.fold)

    panel = load_panel("morph64")
    fold_dir = root / "splits/brca/16shot" / f"fold{args.fold}"
    rows = {phase: read_csv(fold_dir / f"{phase}.csv")
            for phase in ("train", "val", "test")}
    tme = load_tme(args.tme_csv, panel)
    preprocessor = fit_fold_preprocessor(rows, tme, panel).to(device)
    paths = {}
    for phase_rows in rows.values():
        for row in phase_rows:
            if row["label"] not in LABELS:
                raise ValueError(f"unsupported BRCA class: {row['label']!r}")
            name = row.get("feature_file", "")
            if not name.endswith(".h5") or Path(name).name != name:
                raise ValueError(f"invalid feature filename for {row['slide_id']}")
            path = args.feature_root.expanduser() / name
            if not path.is_file():
                raise FileNotFoundError(f"feature file missing for {row['slide_id']}: {path}")
            paths[row["slide_id"]] = path
    native = FrozenMUSE.from_checkpoint(args.native_checkpoint, device)
    conditioner = TMEContext(panel, model_dim=1024, hidden_dim=128,
                             heads=4, dropout=0.1, initial_gate=0.1,
                             single_scale=True).to(device)
    optimizer = torch.optim.Adam(conditioner.parameters(), lr=1e-4, weight_decay=1e-5)
    best_error, best_epoch, stale, best_state = float("inf"), -1, 0, None
    history = []
    for epoch in range(200):
        conditioner.train()
        order = rows["train"].copy()
        random.shuffle(order)
        train_loss = 0.0
        for row in order:
            slide = row["slide_id"]
            bag = read_bag(paths[slide], device)
            values = preprocessor(tme[slide].unsqueeze(0).to(device))
            target = torch.tensor([LABELS[row["label"]]], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = native(bag, conditioner, values)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"nonfinite MUSE logits for slide {slide!r}")
            loss = F.cross_entropy(logits, target)
            loss.backward()
            if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in conditioner.parameters() if parameter.requires_grad):
                raise RuntimeError("nonfinite or missing conditioner gradient")
            optimizer.step()
            train_loss += float(loss.detach())
        val, _ = evaluate(rows["val"], paths, tme, preprocessor, native, conditioner, device)
        if val["error"] < best_error:
            best_error, best_epoch, stale = val["error"], epoch, 0
            best_state = copy.deepcopy(conditioner.state_dict())
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": train_loss / len(order),
                        "val_loss": val["loss"], "val_error": val["error"]})
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} "
              f"val_error={val['error']:.6f}", flush=True)
        if epoch > 80 and stale >= 20:
            break

    if best_state is None:
        raise RuntimeError("no finite validation selection")
    conditioner.load_state_dict(best_state)
    test, predictions = evaluate(rows["test"], paths, tme, preprocessor,
                                 native, conditioner, device)
    output.mkdir(parents=True)
    torch.save({"fold": args.fold, "panel": panel.name,
                "preprocessor": preprocessor.state_dict(),
                "conditioner": conditioner.state_dict()}, output / "adapter.pt")
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(predictions)
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(history)
    (output / "summary.json").write_text(json.dumps({
        "fold": args.fold, "arm": "actual", "best_epoch": best_epoch,
        "validation_error": best_error, "test_error": test["error"],
        "test_loss": test["loss"], "test_slides": len(predictions),
        "metrics": metric_bundle(predictions)}, indent=2) + "\n")
    print(f"completed fold{args.fold}; outputs: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True,
                        help="directory containing the bundled split filenames")
    parser.add_argument("--tme-csv", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True,
                        help="trained native MUSE CLIP-RN50 state_dict")
    parser.add_argument("--native-config", type=Path,
                        help="native config.json; defaults to checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory outside this source folder")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
