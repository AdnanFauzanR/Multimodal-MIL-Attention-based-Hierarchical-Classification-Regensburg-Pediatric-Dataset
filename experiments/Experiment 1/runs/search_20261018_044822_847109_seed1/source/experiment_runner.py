"""Antrean serial Experiment 1, pemilihan macro-F1, dan laporan pengujian."""
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

from multimodal_mil_model import (
    CLASS_NAMES, ModelConfig, MultimodalMILModel, OptimizerConfig,
    build_optimizer, collate_bags, count_parameters, labels_to_class,
)
from training import binary_metrics, compute_metrics, fit, probability_metrics, run_epoch, set_seed


AUROC_DEFINITION = "macro_one_vs_rest_three_classes"
MONITORS = {"val_f1", "val_auroc", "val_diagnosis_auroc", "val_severity_auroc"}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True).encode()).hexdigest()


def make_scenarios():
    regimes = [
        ("scratch", False, False, 1.0),
        ("pretrained_trainable", True, False, 0.1),
        ("pretrained_frozen", True, True, 0.1),
    ]
    profiles = [
        ("P1", 1e-4, 0.4, 1e-3),
        ("P2", 1e-3, 0.2, 1e-4),
        ("P3", 1e-5, 0.7, 1e-3),
        ("P4", 1e-4, 0.7, 1e-4),
    ]
    rows = []
    for regime, pretrained, frozen, multiplier in regimes:
        for optimizer in ["adam", "adamw", "sgd"]:
            for profile, lr, dropout, decay in profiles:
                number = len(rows) + 1
                rows.append({
                    "scenario_id": f"S{number:02d}_{regime}_{optimizer}_{profile}",
                    "regime": regime, "pretrained": pretrained, "freeze_backbone": frozen,
                    "optimizer": optimizer, "profile": profile, "lr": lr,
                    "dropout_rate": dropout, "weight_decay": decay,
                    "backbone_lr_multiplier": multiplier,
                    "effective_backbone_lr": None if frozen else lr * multiplier,
                })
    return rows


def collate_with_ids(items):
    batch = collate_bags(items)
    batch["patient_id"] = torch.tensor([int(x["patient_id"]) for x in items])
    return batch


def binary_report(target, probability, threshold=0.5):
    target, probability = np.asarray(target, dtype=int), np.asarray(probability, dtype=float)
    result = binary_metrics(target, probability >= threshold)
    both = np.unique(target).size == 2
    result.update(
        auroc=float(roc_auc_score(target, probability)) if both else None,
        auprc=float(average_precision_score(target, probability)) if both else None,
        brier=float(np.mean((target - probability) ** 2)) if len(target) else None,
        confusion_matrix=confusion_matrix(target, probability >= threshold, labels=[0, 1]).tolist(),
    )
    return result


@torch.no_grad()
def evaluate_and_save(model, loader, device, directory, split_name, checkpoint_path, threshold=0.5):
    directory, device = Path(directory), torch.device(device)
    model.eval()
    rows = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, enabled=False):
            output = model(batch["images"].float(), batch["image_mask"], batch["tabular"].float())
            probabilities = model.predict_proba(output)
            p = output["logit_diagnosis"].float().sigmoid()
            q = output["logit_complication"].float().sigmoid()
            predicted = model.predict_classes(output, threshold=threshold)
            target = labels_to_class(batch["y_appendicitis"], batch["y_complicated"])
        for i, pid in enumerate(batch["patient_id"]):
            rows.append({
                "patient_id": int(pid), "n_images": int(batch["image_mask"][i].sum()),
                "y_true": int(target[i]), "y_appendicitis": int(batch["y_appendicitis"][i]),
                "y_complicated": int(batch["y_complicated"][i]),
                "p_diagnosis": float(p[i]), "q_complication": float(q[i]),
                "prob_no": float(probabilities[i, 0]),
                "prob_uncomplicated": float(probabilities[i, 1]),
                "prob_complicated": float(probabilities[i, 2]),
                "y_pred": int(predicted[i]), "y_pred_argmax": int(probabilities[i].argmax()),
            })
    predictions = pd.DataFrame(rows)
    if len(predictions) != len(loader.dataset) or predictions.patient_id.duplicated().any():
        raise ValueError("Prediksi tidak cocok dengan pasien pada dataset evaluasi")
    probability_columns = ["prob_no", "prob_uncomplicated", "prob_complicated"]
    auc = probability_metrics(predictions.y_true, predictions[probability_columns],
                              predictions.p_diagnosis, predictions.q_complication)
    overall = compute_metrics(predictions.y_true.to_numpy(), predictions.y_pred.to_numpy())
    argmax = compute_metrics(predictions.y_true.to_numpy(), predictions.y_pred_argmax.to_numpy())
    app = predictions.y_appendicitis.eq(1)
    result = {
        "split": split_name, "patients": len(predictions), "threshold": threshold,
        "auroc_definition": AUROC_DEFINITION, "class_order": list(CLASS_NAMES),
        **auc,
        "overall": {k: v for k, v in overall.items() if k != "per_class"},
        "argmax": {k: v for k, v in argmax.items() if k != "per_class"},
        "diagnosis": binary_report(predictions.y_appendicitis, predictions.p_diagnosis, threshold),
        "severity_direct_true_appendicitis": binary_report(
            predictions.loc[app, "y_complicated"], predictions.loc[app, "q_complication"], threshold),
        "severity_after_gating_true_appendicitis": binary_metrics(
            predictions.loc[app, "y_complicated"].to_numpy(), predictions.loc[app, "y_pred"].eq(2).to_numpy()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    predictions.to_csv(directory / f"{split_name}_predictions.csv", index=False)
    overall["per_class"].to_csv(directory / f"{split_name}_per_class.csv")
    pd.DataFrame(overall["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        directory / f"{split_name}_confusion_matrix.csv")
    write_json(directory / f"{split_name}_metrics.json", result)
    return clean_json(result), predictions


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_history_plots(history, directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    # Render file tanpa jendela GUI atau mengubah backend notebook.
    fig = Figure(figsize=(15, 7))
    FigureCanvasAgg(fig)
    for axis, key in zip(fig.subplots(2, 3).ravel(),
                         ["loss", "accuracy", "precision", "sensitivity", "specificity", "f1"]):
        for split in ["train", "val"]:
            axis.plot(history.index, history[f"{split}_{key}"], label=split)
        axis.axvline(history.attrs["best_epoch"], linestyle="--", color="gray")
        axis.set(xlabel="Epoch", title=key)
        axis.legend()
    fig.tight_layout()
    fig.savefig(Path(directory) / "training_curves.png", dpi=140, bbox_inches="tight")
    fig = Figure(figsize=(15, 4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    for axis, key, title in zip(axes, ["auroc", "diagnosis_auroc", "severity_auroc"],
                               ["Macro AUROC OvR", "Diagnosis AUROC", "Conditional severity AUROC"]):
        for split in ["train", "val"]:
            axis.plot(history.index, history[f"{split}_{key}"], label=split)
        axis.axvline(history.attrs["best_epoch"], linestyle="--", color="gray")
        axis.set(xlabel="Epoch", ylabel="AUROC", title=title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(Path(directory) / "auroc_curves.png", dpi=140)


def run_trial(directory, scenario, model_config, optimizer_config, train_dataset,
              val_dataset, train_eval_dataset, settings, shared_artifacts):
    """Satu model GPU per trial; tidak mengembalikan model atau optimizer."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model = optimizer = None
    start = time.perf_counter()
    device = torch.device(settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    trial_config = {
        "scenario": scenario, "model": model_config.to_dict(), "optimizer": asdict(optimizer_config),
        "training": settings, "shared_artifacts": shared_artifacts,
        "auroc_definition": AUROC_DEFINITION,
    }
    write_json(directory / "config.json", trial_config)
    write_json(directory / "status.json", {"status": "training"})
    checkpoint_path, best_loss_path = directory / "best_model.pt", directory / "best_loss.pt"
    tracker = {"epoch": 0, "loss": float("inf")}

    def epoch_runner(model, loader, device, optimizer=None, scaler=None, use_amp=False):
        if optimizer is not None:
            tracker["epoch_start"] = time.perf_counter()
        result = run_epoch(model, loader, device, optimizer, scaler,
                           use_amp=bool(use_amp and optimizer is not None))
        if not np.isfinite(result[0]["loss"]):
            raise FloatingPointError("Loss tidak finite")
        if optimizer is None:
            tracker["epoch"] += 1
            if result[0]["loss"] < tracker["loss"]:
                tracker["loss"] = result[0]["loss"]
                torch.save({"epoch": tracker["epoch"], "monitor": "val_loss", "value": tracker["loss"],
                            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            "config": model.cfg.to_dict()}, best_loss_path)
            row = {"epoch": tracker["epoch"], "seconds": time.perf_counter()-tracker["epoch_start"]}
            for split, values in [("train", tracker["train_result"]), ("val", result)]:
                row.update({f"{split}_{k}": v for k, v in values[0].items()})
                metrics = compute_metrics(values[1], values[2])
                row.update({f"{split}_{k}": metrics[k] for k in
                            ["accuracy", "precision", "sensitivity", "specificity", "f1"]})
            pd.DataFrame([row]).to_csv(directory / "history_partial.csv", mode="a", index=False,
                                      header=not (directory / "history_partial.csv").exists())
        else:
            tracker["train_result"] = result
        return result

    try:
        set_seed(settings["seed"])
        model = MultimodalMILModel(model_config).to(device)
        model.train()
        optimizer = build_optimizer(model, optimizer_config)
        trial_config["parameters"] = count_parameters(model)
        trial_config["optimizer_groups"] = [
            {"name": g["name"], "lr": g["lr"], "weight_decay": g["weight_decay"]}
            for g in optimizer.param_groups]
        write_json(directory / "config.json", trial_config)
        # Reset sampling/augmentasi setelah konstruksi model pada setiap trial.
        set_seed(settings["seed"])
        generator = torch.Generator().manual_seed(settings["seed"])
        common = dict(batch_size=settings["batch_size"], num_workers=settings["num_workers"],
                      collate_fn=collate_with_ids)
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, generator=generator, **common)
        val_loader = DataLoader(val_dataset, shuffle=False, **common)
        train_eval_loader = DataLoader(train_eval_dataset, shuffle=False, **common)
        if not len(train_loader):
            raise ValueError("Batch size lebih besar daripada dataset training")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with (directory / "training.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)):
                print(json.dumps(clean_json(trial_config), indent=2))
                history = fit(
                    model, train_loader, val_loader, optimizer, device=device,
                    epochs=settings["epochs"], monitor=settings["monitor"],
                    checkpoint_path=checkpoint_path, restore_best=True, use_amp=settings["use_amp"],
                    early_stopping_patience=settings["patience"], early_stopping_min_delta=settings["min_delta"],
                    early_stopping_rule=settings.get("early_stopping_rule", "previous_epoch_val_loss"),
                    checkpoint_tie_break="val_loss" if settings["monitor"] == "val_f1" else None,
                    epoch_runner=epoch_runner, verbose=settings.get("verbose", True),
                )
        history.to_csv(directory / "history.csv")
        write_json(directory / "history_metadata.json", {**history.attrs, "auroc_definition": AUROC_DEFINITION})
        # Evaluasi FP32 selalu menggunakan checkpoint yang dipilih oleh monitor.
        train_metrics, _ = evaluate_and_save(model, train_eval_loader, device, directory,
                                            "train", checkpoint_path, settings["threshold"])
        val_metrics, _ = evaluate_and_save(model, val_loader, device, directory,
                                          "val", checkpoint_path, settings["threshold"])
        monitor_key = settings["monitor"].removeprefix("val_")
        evaluated_monitor = val_metrics["overall"]["f1"] if monitor_key == "f1" else val_metrics[monitor_key]
        if evaluated_monitor is None or not np.isclose(
                evaluated_monitor, history.attrs["best_value"], atol=1e-6, rtol=0):
            raise ValueError("Evaluasi checkpoint tidak mereproduksi monitor terbaik")
        if settings.get("save_plots", True):
            save_history_plots(history, directory)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint.update(scenario_id=scenario["scenario_id"], seed=settings["seed"],
                          auroc_definition=AUROC_DEFINITION, shared_artifacts=shared_artifacts)
        torch.save(checkpoint, checkpoint_path)
        # Metadata hash laporan harus menunjuk checkpoint final yang sama.
        checkpoint_hash = file_sha256(checkpoint_path)
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            metrics["checkpoint_sha256"] = checkpoint_hash
            write_json(directory / f"{split}_metrics.json", metrics)
        result = {
            **scenario, "status": "completed", "monitor": settings["monitor"],
            "best_epoch": history.attrs["best_epoch"], "epochs_ran": len(history),
            "train_auroc": train_metrics["auroc"], "val_auroc": val_metrics["auroc"],
            "val_diagnosis_auroc": val_metrics["diagnosis_auroc"],
            "val_severity_auroc": val_metrics["severity_auroc"],
            "train_accuracy": train_metrics["overall"]["accuracy"],
            "val_accuracy": val_metrics["overall"]["accuracy"], "val_f1": val_metrics["overall"]["f1"],
            "train_f1": train_metrics["overall"]["f1"],
            "val_loss": float(history.loc[history.attrs["best_epoch"], "val_loss"]),
            "train_loss_at_checkpoint": float(history.loc[history.attrs["best_epoch"], "train_loss"]),
            "early_stopping_rule": history.attrs["early_stopping_rule"],
            "early_stopping_triggered": history.attrs["early_stopping_triggered"],
            "seconds_total": time.perf_counter() - start,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None,
            "checkpoint_path": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_hash,
            "trial_directory": str(directory.resolve()), "test_evaluated": False,
        }
        write_json(directory / "result.json", result)
        write_json(directory / "status.json", {"status": "completed"})
        return result
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        (directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(directory / "status.json", {"status": status, "error": str(error)})
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return {**scenario, "status": status, "error": str(error), "trial_directory": str(directory.resolve())}
    finally:
        model = optimizer = None
        release_gpu()


def save_search_results(directory, records, monitor, total):
    directory = Path(directory)
    frame = pd.DataFrame(records)
    if len(frame):
        completed = frame[frame.status.eq("completed")].copy()
        if len(completed):
            sort_keys = [monitor]
            if monitor == "val_f1" and "val_loss" in completed:
                sort_keys.append("val_loss")
            completed = completed.sort_values(sort_keys + ["scenario_id"],
                                              ascending=[False] + [True] * len(sort_keys))
            completed.insert(0, "rank", np.arange(1, len(completed) + 1))
        rest = frame[~frame.status.eq("completed")].copy()
        leaderboard = pd.concat([completed, rest], ignore_index=True)
    else:
        completed = pd.DataFrame()
        leaderboard = frame
    leaderboard.to_csv(directory / "leaderboard.csv", index=False)
    failed = sum(r["status"] == "failed" for r in records)
    selection_final = len(records) == total and len(completed) + failed == total
    selection = {
        "monitor": monitor, "auroc_definition": AUROC_DEFINITION,
        "tie_break": ("lower_val_loss_then_scenario_id; within_trial_lower_val_loss_then_earliest_epoch"
                      if monitor == "val_f1" else "scenario_id_ascending; earliest_epoch_within_trial"),
        "total_scenarios": total, "attempted": len(records), "completed": len(completed),
        "failed": failed, "pending": total-len(completed)-failed,
        "selection_final": selection_final,
        "test_evaluated": False,
    }
    if len(completed):
        selection["winner"] = clean_json(completed.iloc[0].to_dict())
    # Rekam evaluasi terpisah: ranking sementara saat resume dapat menunjuk model lain.
    evaluation_record = directory / "test_evaluation.json"
    old_selection_path = directory / "best_model.json"
    previous = evaluation_record if evaluation_record.exists() else old_selection_path
    if previous.exists():
        old = json.loads(previous.read_text())
        if (old.get("test_evaluated") and "winner" in selection
                and old.get("winner", {}).get("checkpoint_sha256") is not None
                and old["winner"]["checkpoint_sha256"] == selection["winner"].get("checkpoint_sha256")):
            for key in ["test_evaluated", "test_metrics", "test_evaluation_utc"]:
                if key in old:
                    selection[key] = old[key]
    write_json(directory / "best_model.json", selection)
    write_json(directory / "status.json", {
        "status": "search_completed" if selection_final else "search_partial",
        "attempted": len(records), "completed": len(completed), "total": total,
    })
    return leaderboard


def run_scenario_search(directory, scenarios, base_model_options, base_optimizer_options,
                        num_features, train_dataset, val_dataset, train_eval_dataset,
                        settings, shared_artifacts, retry_failed=False):
    """Serial; completed dilewati, retry gagal/interupsi hanya jika diminta eksplisit."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if settings["monitor"] not in MONITORS:
        raise ValueError("Monitor search harus val_f1 atau salah satu AUROC yang didukung")
    if len({s["scenario_id"] for s in scenarios}) != len(scenarios):
        raise ValueError("Scenario ID duplikat")
    search_spec = {
        "scenarios": scenarios, "base_model_options": base_model_options,
        "base_optimizer_options": base_optimizer_options, "num_features": num_features,
        "settings": settings, "shared_artifacts": shared_artifacts,
    }
    signature = fingerprint(search_spec)
    spec_path = directory / "search_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text())["fingerprint"] != signature:
            raise ValueError("Konfigurasi/artefak berubah; gunakan folder search baru")
    else:
        write_json(spec_path, {"fingerprint": signature, **search_spec})
    pd.DataFrame(scenarios).to_csv(directory / "scenarios.csv", index=False)
    records = []
    for scenario in scenarios:
        folder = directory / "trials" / scenario["scenario_id"]
        result_path = folder / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if file_sha256(result["checkpoint_path"]) != result["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint berubah: {scenario['scenario_id']}")
            records.append(result)
        elif folder.exists() and not retry_failed:
            status_path = folder / "status.json"
            state = json.loads(status_path.read_text()) if status_path.exists() else {}
            records.append({**scenario, "status": state.get("status", "interrupted"),
                            "error": state.get("error", "Trial sebelumnya belum selesai"),
                            "trial_directory": str(folder.resolve())})
        else:
            if folder.exists():
                archived = folder.with_name(folder.name + "_previous_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
                # Kedua path tetap di direktori trials yang sama, bukan data sumber.
                if (archived.exists() or folder.is_symlink()
                        or not folder.resolve().is_relative_to((directory / "trials").resolve())
                        or archived.parent.resolve() != folder.parent.resolve()):
                    raise ValueError("Path arsip retry tidak valid")
                folder.rename(archived)
            model_options = {**base_model_options, "pretrained": scenario["pretrained"],
                             "freeze_backbone": scenario["freeze_backbone"],
                             "backbone_trainable_stage": None, "dropout_rate": scenario["dropout_rate"]}
            optimizer_options = {**base_optimizer_options, "name": scenario["optimizer"],
                                 "lr": scenario["lr"], "weight_decay": scenario["weight_decay"],
                                 "backbone_lr_multiplier": scenario["backbone_lr_multiplier"]}
            print(f"\n[{len(records)+1}/{len(scenarios)}] {scenario['scenario_id']}")
            try:
                result = run_trial(folder, scenario, ModelConfig(num_tabular_features=num_features, **model_options),
                                   OptimizerConfig(**optimizer_options), train_dataset, val_dataset,
                                   train_eval_dataset, settings, shared_artifacts)
            except (KeyboardInterrupt, SystemExit):
                save_search_results(directory, records, settings["monitor"], len(scenarios))
                raise
            records.append(result)
        save_search_results(directory, records, settings["monitor"], len(scenarios))
        release_gpu()
    return save_search_results(directory, records, settings["monitor"], len(scenarios))


def evaluate_search_winner(directory, test_dataset, settings):
    directory = Path(directory)
    selection = json.loads((directory / "best_model.json").read_text())
    if not selection["selection_final"] or "winner" not in selection:
        raise ValueError("Search belum selesai atau tidak ada model valid")
    winner = selection["winner"]
    path = Path(winner["checkpoint_path"])
    if file_sha256(path) != winner["checkpoint_sha256"]:
        raise ValueError("Checkpoint pemenang berubah")
    model = None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = MultimodalMILModel(ModelConfig(**{**checkpoint["config"], "pretrained": False}))
        model.load_state_dict(checkpoint["model_state"])
        device = torch.device(settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        loader = DataLoader(test_dataset, batch_size=settings["batch_size"], shuffle=False,
                            num_workers=settings["num_workers"], collate_fn=collate_with_ids)
        metrics, predictions = evaluate_and_save(model, loader, device, Path(winner["trial_directory"]),
                                                 "test", path, settings["threshold"])
        from experiment_evaluation import evaluate_test_suite
        report_dir = Path(winner["trial_directory"]) / "testing"
        evaluate_test_suite(model, loader, device, report_dir, path, settings["threshold"],
                            bootstrap_resamples=settings.get("test_bootstrap_resamples", 1000),
                            seed=settings.get("seed", 1), expected_predictions=predictions)
        metrics["comprehensive_report_directory"] = str(report_dir.resolve())
        write_json(Path(winner["trial_directory"]) / "test_metrics.json", metrics)
        selection.update(test_evaluated=True, test_metrics=metrics,
                         test_evaluation_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "test_evaluation.json", {
            key: selection[key] for key in ["winner", "test_evaluated", "test_metrics", "test_evaluation_utc"]
        })
        write_json(directory / "best_model.json", selection)
        return metrics, predictions
    finally:
        model = None
        release_gpu()
