from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def _autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _prepare_batch(batch, device: torch.device):
    videos = batch[0].to(device, non_blocking=True)
    labels = batch[1].to(device, non_blocking=True)
    sample_ids = batch[2] if len(batch) > 2 else None
    temporal_ids = batch[3] if len(batch) > 3 else None
    spatial_ids = batch[4] if len(batch) > 4 else None
    return videos, labels, sample_ids, temporal_ids, spatial_ids


def _to_list_of_ids(sample_ids) -> List[str]:
    if sample_ids is None:
        return []
    if isinstance(sample_ids, (list, tuple)):
        return [str(x) for x in sample_ids]
    if torch.is_tensor(sample_ids):
        return [str(x.item()) for x in sample_ids]
    return [str(sample_ids)]


def binary_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    num_pos = int(pos_mask.sum())
    num_neg = int(neg_mask.sum())

    if num_pos == 0 or num_neg == 0:
        return None

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.zeros_like(sorted_scores, dtype=float)

    start = 0
    n = len(sorted_scores)
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = avg_rank
        start = end

    full_ranks = np.zeros_like(ranks)
    full_ranks[order] = ranks

    pos_ranks_sum = full_ranks[pos_mask].sum()
    auc = (pos_ranks_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
    return float(auc)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    for t, p in zip(y_true, y_pred):
        confusion[label_to_index[t], label_to_index[p]] += 1

    metrics = {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0,
        "confusion_matrix": confusion.tolist(),
        "num_samples": int(len(y_true)),
    }

    if labels == [0, 1]:
        tn, fp, fn, tp = confusion.ravel()
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        fpr = _safe_divide(fp, fp + tn)
        fnr = _safe_divide(fn, fn + tp)
        auc = binary_auc_score(y_true, y_prob[:, 1]) if y_prob is not None and y_prob.shape[1] == 2 else None

        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
                "auc": auc,
            }
        )
    else:
        metrics.update(
            {
                "precision": None,
                "recall": None,
                "f1": None,
                "fpr": None,
                "fnr": None,
                "auc": None,
            }
        )

    return metrics


def aggregate_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_ids: List[str],
) -> Dict[str, np.ndarray]:
    merged = {}

    for prob, label, sample_id in zip(probabilities, labels, sample_ids):
        if sample_id not in merged:
            merged[sample_id] = {
                "prob_sum": prob.astype(np.float64).copy(),
                "count": 1,
                "label": int(label),
            }
        else:
            merged[sample_id]["prob_sum"] += prob
            merged[sample_id]["count"] += 1

    final_ids = []
    final_labels = []
    final_probs = []
    final_preds = []

    for sample_id in sorted(merged.keys()):
        item = merged[sample_id]
        mean_prob = item["prob_sum"] / item["count"]
        final_ids.append(sample_id)
        final_labels.append(item["label"])
        final_probs.append(mean_prob)
        final_preds.append(int(np.argmax(mean_prob)))

    return {
        "sample_ids": np.asarray(final_ids),
        "labels": np.asarray(final_labels, dtype=int),
        "probabilities": np.asarray(final_probs, dtype=np.float32),
        "predictions": np.asarray(final_preds, dtype=int),
    }


def build_train_criterion():
    return nn.CrossEntropyLoss()


def train_one_epoch(
    model: nn.Module,
    auxiliary_extractor: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    model.train(True)
    auxiliary_extractor.train(False)

    criterion = criterion or build_train_criterion()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in data_loader:
        videos, labels, _, _, _ = _prepare_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, use_amp):
            d = auxiliary_extractor(videos)
            logits, _feat, _ = model(videos, d)
            loss = criterion(logits, labels)

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)

        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    auxiliary_extractor: nn.Module,
    data_loader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = True,
    aggregate_by_id: bool = False,
) -> Dict[str, object]:
    model.eval()
    auxiliary_extractor.eval()

    criterion = criterion or nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_sample_ids = []

    for batch in data_loader:
        videos, labels, sample_ids, _, _ = _prepare_batch(batch, device)

        with _autocast_context(device, use_amp):
            d = auxiliary_extractor(videos)
            logits, _feat, _ = model(videos, d)
            loss = criterion(logits, labels)

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

        if sample_ids is not None:
            all_sample_ids.extend(_to_list_of_ids(sample_ids))

    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=int)
    all_predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([], dtype=int)
    all_probabilities = np.concatenate(all_probabilities, axis=0) if all_probabilities else np.array([], dtype=np.float32)

    if aggregate_by_id:
        merged = aggregate_predictions(
            probabilities=all_probabilities,
            labels=all_labels,
            sample_ids=all_sample_ids,
        )
        metrics = compute_classification_metrics(
            y_true=merged["labels"],
            y_pred=merged["predictions"],
            y_prob=merged["probabilities"],
        )
        metrics["sample_ids"] = merged["sample_ids"].tolist()
        metrics["labels"] = merged["labels"].tolist()
        metrics["predictions"] = merged["predictions"].tolist()
        metrics["probabilities"] = merged["probabilities"].tolist()
    else:
        metrics = compute_classification_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities,
        )
        metrics["sample_ids"] = all_sample_ids
        metrics["labels"] = all_labels.tolist()
        metrics["predictions"] = all_predictions.tolist()
        metrics["probabilities"] = all_probabilities.tolist()

    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def test(
    model: nn.Module,
    auxiliary_extractor: nn.Module,
    data_loader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = True,
) -> Dict[str, object]:
    return evaluate(
        model=model,
        auxiliary_extractor=auxiliary_extractor,
        data_loader=data_loader,
        device=device,
        criterion=criterion,
        use_amp=use_amp,
        aggregate_by_id=True,
    )


def save_best_checkpoint(
    path: str,
    model: nn.Module,
    auxiliary_extractor: Optional[nn.Module] = None,
    args: Optional[dict] = None,
    save_auxiliary: bool = False,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": args,
    }

    if save_auxiliary and auxiliary_extractor is not None:
        checkpoint["auxiliary_state_dict"] = auxiliary_extractor.state_dict()

    torch.save(checkpoint, path)


def save_resume_checkpoint(
    path: str,
    model: nn.Module,
    auxiliary_extractor: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    args: Optional[dict] = None,
    save_auxiliary: bool = False,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "args": args,
    }

    if save_auxiliary and auxiliary_extractor is not None:
        checkpoint["auxiliary_state_dict"] = auxiliary_extractor.state_dict()
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    auxiliary_extractor: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=map_location)

    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint

    model.load_state_dict(model_state, strict=strict)

    if auxiliary_extractor is not None and "auxiliary_state_dict" in checkpoint:
        auxiliary_extractor.load_state_dict(checkpoint["auxiliary_state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def format_metrics(metrics: Dict[str, object]) -> str:
    parts = [
        f"loss={metrics.get('loss', 0.0):.4f}",
        f"accuracy={metrics.get('accuracy', 0.0):.4f}",
    ]

    if metrics.get("precision") is not None:
        parts.extend(
            [
                f"precision={metrics['precision']:.4f}",
                f"recall={metrics['recall']:.4f}",
                f"f1={metrics['f1']:.4f}",
                f"fpr={metrics['fpr']:.4f}",
                f"fnr={metrics['fnr']:.4f}",
            ]
        )

    if metrics.get("auc") is not None:
        parts.append(f"auc={metrics['auc']:.4f}")

    return ", ".join(parts)