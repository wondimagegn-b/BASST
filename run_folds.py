import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from auxiliary import build_auxiliary_extractor
from basst_model import build_model
from dataset import build_dataset
from trainer import (
    train_one_epoch,
    evaluate,
    test,
    save_best_checkpoint,
    save_resume_checkpoint,
    load_checkpoint,
    format_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1])

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--train_label_template", type=str, required=True)
    parser.add_argument("--test_label_template", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--finetune_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--checkpoint_template", type=str, default="")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--selection_metric", type=str, choices=["accuracy", "f1", "auc"], default="accuracy")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--nb_classes", type=int, default=2)
    parser.add_argument("--file_ext", type=str, default="jpg")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--short_side_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=1)
    parser.add_argument("--test_num_segment", type=int, default=5)
    parser.add_argument("--test_num_crop", type=int, default=3)

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no_augment", action="store_false", dest="augment")
    parser.set_defaults(augment=True)

    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--aa", type=str, default="rand-m7-n4-mstd0.5-inc1")
    parser.add_argument("--train_interpolation", type=str, default="bicubic")
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)

    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--aux_feature_dim", type=int, default=128)
    parser.add_argument("--aux_feature_size", type=int, default=14)

    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.5)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--scale_factor", type=float, default=0.25)
    parser.add_argument("--prompt_hidden_dim", type=int, default=8)
    parser.add_argument("--prompt_type", type=str, default="deep", choices=["shallow", "deep"])
    parser.add_argument("--global_pool", type=str, default="token", choices=["token", "avg"])
    parser.add_argument("--qkv_bias", action="store_true")
    parser.set_defaults(qkv_bias=True)

    parser.add_argument("--openface_repo_root", type=str, required=True)
    parser.add_argument("--openface_weights_path", type=str, default="")

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    if args.openface_weights_path == "":
        args.openface_weights_path = None

    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_template_path(template: str, fold: int) -> str:
    return template.format(fold=fold) if "{fold}" in template else template


def resolve_checkpoint_path(args, fold: int) -> str:
    if args.checkpoint_template:
        return resolve_template_path(args.checkpoint_template, fold)
    return args.checkpoint_path


def make_fold_args(args, fold: int):
    fold_args = copy.deepcopy(args)
    fold_args.train_label_path = resolve_template_path(args.train_label_template, fold)
    fold_args.test_label_path = resolve_template_path(args.test_label_template, fold)
    return fold_args


def build_dataloaders(fold_args):
    train_dataset, _ = build_dataset(is_train=True, test_mode=False, args=fold_args)
    val_dataset, _ = build_dataset(is_train=False, test_mode=False, args=fold_args)
    test_dataset, _ = build_dataset(is_train=False, test_mode=True, args=fold_args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=fold_args.batch_size,
        shuffle=True,
        num_workers=fold_args.num_workers,
        pin_memory=fold_args.pin_mem,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=fold_args.eval_batch_size,
        shuffle=False,
        num_workers=fold_args.num_workers,
        pin_memory=fold_args.pin_mem,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=fold_args.eval_batch_size,
        shuffle=False,
        num_workers=fold_args.num_workers,
        pin_memory=fold_args.pin_mem,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def build_test_loader(fold_args):
    test_dataset, _ = build_dataset(is_train=False, test_mode=True, args=fold_args)

    test_loader = DataLoader(
        test_dataset,
        batch_size=fold_args.eval_batch_size,
        shuffle=False,
        num_workers=fold_args.num_workers,
        pin_memory=fold_args.pin_mem,
        drop_last=False,
    )
    return test_loader


def build_components(args, device):
    model = build_model(args).to(device)
    auxiliary_extractor = build_auxiliary_extractor(args).to(device)
    return model, auxiliary_extractor


def build_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def average_fold_metrics(fold_metrics):
    keys = ["loss", "accuracy", "precision", "recall", "f1", "fpr", "fnr", "auc"]
    summary = {}

    for key in keys:
        values = [m[key] for m in fold_metrics if key in m and m[key] is not None]
        summary[key] = float(np.mean(values)) if values else None

    return summary


def write_json(path: Path, content):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def run_train_fold(args, fold: int, device: torch.device):
    fold_args = make_fold_args(args, fold)

    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(fold_args)
    model, auxiliary_extractor = build_components(fold_args, device)

    optimizer = build_optimizer(model, fold_args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fold_args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=fold_args.use_amp and device.type == "cuda")

    history = []
    best_metric = -float("inf")
    best_checkpoint_path = fold_dir / "best.pth"
    last_checkpoint_path = fold_dir / "last.pth"

    for epoch in range(1, fold_args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            auxiliary_extractor=auxiliary_extractor,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=fold_args.use_amp,
            max_grad_norm=fold_args.max_grad_norm,
        )

        val_metrics = evaluate(
            model=model,
            auxiliary_extractor=auxiliary_extractor,
            data_loader=val_loader,
            device=device,
            use_amp=fold_args.use_amp,
            aggregate_by_id=False,
        )

        scheduler.step()

        current_metric = val_metrics.get(fold_args.selection_metric)
        if current_metric is None:
            current_metric = val_metrics["accuracy"]

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(epoch_record)

        print(f"[Fold {fold}] Epoch {epoch:03d} | train: {format_metrics(train_metrics)}")
        print(f"[Fold {fold}] Epoch {epoch:03d} | val:   {format_metrics(val_metrics)}")

        if current_metric > best_metric:
            best_metric = float(current_metric)
            save_best_checkpoint(
                path=str(best_checkpoint_path),
                model=model,
                auxiliary_extractor=auxiliary_extractor,
                args=vars(fold_args),
                save_auxiliary=False,
            )

        if epoch % fold_args.save_every == 0 or epoch == fold_args.epochs:
            save_resume_checkpoint(
                path=str(last_checkpoint_path),
                model=model,
                auxiliary_extractor=auxiliary_extractor,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                args=vars(fold_args),
                save_auxiliary=False,
            )

    write_json(fold_dir / "history.json", history)

    load_checkpoint(
        path=str(best_checkpoint_path),
        model=model,
        auxiliary_extractor=auxiliary_extractor,
        map_location=device,
        strict=True,
    )

    test_metrics = test(
        model=model,
        auxiliary_extractor=auxiliary_extractor,
        data_loader=test_loader,
        device=device,
        use_amp=fold_args.use_amp,
    )

    print(f"[Fold {fold}] Test: {format_metrics(test_metrics)}")
    write_json(fold_dir / "test_metrics.json", test_metrics)

    return {
        "fold": fold,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "test_metrics": test_metrics,
    }


def run_test_fold(args, fold: int, device: torch.device):
    fold_args = make_fold_args(args, fold)

    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_checkpoint_path(args, fold)
    if not checkpoint_path:
        raise ValueError("A trained ASD checkpoint is required in test mode.")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_loader = build_test_loader(fold_args)
    model, auxiliary_extractor = build_components(fold_args, device)

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        auxiliary_extractor=auxiliary_extractor,
        map_location=device,
        strict=True,
    )

    test_metrics = test(
        model=model,
        auxiliary_extractor=auxiliary_extractor,
        data_loader=test_loader,
        device=device,
        use_amp=fold_args.use_amp,
    )

    print(f"[Fold {fold}] Test: {format_metrics(test_metrics)}")
    write_json(fold_dir / "test_metrics.json", test_metrics)

    return {
        "fold": fold,
        "checkpoint": checkpoint_path,
        "test_metrics": test_metrics,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "test" and args.finetune_path:
        raise ValueError("--finetune_path is for train mode only.")
    if args.mode == "train" and args.checkpoint_path:
        print("Warning: --checkpoint_path is ignored in train mode.")
    if args.mode == "test" and not args.checkpoint_path and not args.checkpoint_template:
        raise ValueError("Provide --checkpoint_path or --checkpoint_template in test mode.")

    fold_results = []

    for fold in args.folds:
        if args.mode == "train":
            result = run_train_fold(args, fold, device)
        else:
            result = run_test_fold(args, fold, device)
        fold_results.append(result)

    fold_metrics = [item["test_metrics"] for item in fold_results]
    summary = {
        "mode": args.mode,
        "folds": args.folds,
        "metrics_mean": average_fold_metrics(fold_metrics),
        "fold_results": fold_results,
    }

    write_json(output_dir / "summary.json", summary)

    print("\nAverage test metrics across folds:")
    mean_metrics = summary["metrics_mean"]
    printable = {k: v for k, v in mean_metrics.items() if v is not None}
    print(format_metrics(printable))


if __name__ == "__main__":
    main()