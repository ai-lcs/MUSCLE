from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

from .data import audit_manifest, build_manifest, read_manifest, write_manifest
from .engine import (
    any_trainable_changed,
    append_jsonl,
    build_baseline,
    build_muscle,
    clone_state,
    create_loaders,
    environment_info,
    evaluate,
    file_sha256,
    inspect_muscle_evidence,
    make_conclusion,
    save_checkpoint,
    save_metric_artifacts,
    save_raw_weights,
    set_reproducibility,
    state_unchanged,
    table_iv_comparison,
    train_epoch,
    trainable_state,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=["kvasirv2", "isic2018", "aptos2019"])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT.parent / "outputs" / "paper_resnet50")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-path", type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MUSCLE ResNet-50 three-dataset short validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-data")
    common_parser(validate)
    validate.add_argument("--skip-decode-all", action="store_true")
    validate.add_argument("--skip-hash-all", action="store_true")
    benchmark = subparsers.add_parser("benchmark")
    common_parser(benchmark)
    benchmark.add_argument("--steps", type=int, default=50)
    benchmark.add_argument("--learning-rate", type=float, default=0.01)
    benchmark.add_argument("--annealing-step", type=int, default=50)
    run = subparsers.add_parser("run")
    common_parser(run)
    run.add_argument("--stage", required=True, choices=["baseline", "muscle"])
    run.add_argument("--baseline-epochs", type=int, choices=[5, 10], required=True)
    run.add_argument("--muscle-epochs", type=int, choices=[3, 5], required=True)
    run.add_argument("--learning-rate", type=float, default=0.01)
    run.add_argument("--momentum", type=float, default=0.9)
    run.add_argument("--weight-decay", type=float, default=0.0001)
    run.add_argument("--annealing-step", type=int, default=50)
    run.add_argument("--resume", choices=["auto", "none"], default="auto")
    return parser.parse_args()


def dataset_dir(args: argparse.Namespace) -> Path:
    return args.output_root / args.dataset


def ensure_manifest(args: argparse.Namespace) -> dict[str, Any]:
    path = dataset_dir(args) / "manifest.json"
    if path.exists():
        manifest = read_manifest(path)
        rebuilt = build_manifest(args.dataset, args.data_root)
        if manifest["sha256"] != write_hash(rebuilt):
            raise ValueError("Existing manifest differs from current data; run validate-data explicitly")
        return manifest
    manifest = build_manifest(args.dataset, args.data_root)
    manifest_hash = write_manifest(manifest, path)
    manifest["sha256"] = manifest_hash
    return manifest


def write_hash(manifest: dict[str, Any]) -> str:
    from .data import canonical_manifest_hash

    return canonical_manifest_hash(manifest)


def validate_data(args: argparse.Namespace) -> int:
    output_dir = dataset_dir(args)
    manifest = build_manifest(args.dataset, args.data_root)
    manifest_hash = write_manifest(manifest, output_dir / "manifest.json")
    manifest["sha256"] = manifest_hash
    audit = audit_manifest(
        manifest,
        args.data_root,
        decode_all=not args.skip_decode_all,
        hash_all=not args.skip_hash_all,
    )
    write_json(output_dir / "data_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 2


def build_stage(stage: str, args: argparse.Namespace, num_classes: int, baseline_weights: Path):
    if stage == "baseline":
        return build_baseline(num_classes, args.pretrained_path)
    if not baseline_weights.is_file():
        raise FileNotFoundError(f"MUSCLE requires the trained baseline checkpoint: {baseline_weights}")
    return build_muscle(num_classes, baseline_weights)


def benchmark(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires a CUDA GPU")
    set_reproducibility(args.seed)
    manifest = ensure_manifest(args)
    class_names = manifest["class_names"]
    loaders = create_loaders(
        manifest, args.data_root, image_size=args.image_size, batch_size=args.batch_size,
        workers=args.workers, seed=args.seed
    )
    output_dir = dataset_dir(args) / "benchmark"
    baseline_weights = output_dir / "temporary_baseline_weights.pth"
    reports = {}
    for stage in ("baseline", "muscle"):
        model = build_stage(stage, args, len(class_names), baseline_weights).cuda()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.SGD(trainable, lr=args.learning_rate, momentum=0.9, weight_decay=0.0001)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        torch.cuda.reset_peak_memory_stats()
        metrics, evidence = train_epoch(
            stage, model, loaders["train"], optimizer, scaler, device=torch.device("cuda"),
            amp=args.amp, accumulation=args.accumulation, epoch=1, epochs=1,
            base_lr=args.learning_rate, annealing_step=args.annealing_step,
            class_names=class_names, max_steps=args.steps
        )
        if stage == "baseline":
            save_raw_weights(baseline_weights, model)
        reports[stage] = {
            "steps": metrics["steps"],
            "seconds_per_step": metrics["elapsed_seconds"] / metrics["steps"],
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "evidence": evidence,
        }
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    reports["estimated_hours"] = {
        profile: sum(
            reports[stage]["seconds_per_step"] * sum(
                math.ceil(samples / args.batch_size) for samples in (5600, 10015, 2563)
            ) * epochs
            for stage, epochs in stages
        ) / 3600
        for profile, stages in {
            "three_datasets_10+5": (("baseline", 10), ("muscle", 5)),
            "three_datasets_5+3": (("baseline", 5), ("muscle", 3)),
        }.items()
    }
    reports["estimated_train_samples"] = {"kvasirv2": 5600, "isic2018": 10015, "aptos2019": 2563}
    reports["environment"] = environment_info(REPO_ROOT)
    reports["config"] = serializable_config(args)
    write_json(output_dir / "benchmark.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def shared_training_profile(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "baseline_epochs": args.baseline_epochs,
        "muscle_epochs": args.muscle_epochs,
        "batch_size": args.batch_size,
        "accumulation": args.accumulation,
        "image_size": args.image_size,
        "workers": args.workers,
        "seed": args.seed,
        "amp": args.amp,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "annealing_step": args.annealing_step,
    }


def check_shared_profile(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    path = args.output_root / "short_run_profile.json"
    profile = shared_training_profile(args)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != profile:
            raise ValueError(
                "Training profile differs from the profile already used for another dataset: "
                f"{path}"
            )
    return path, profile


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("formal short validation requires a CUDA GPU")
    set_reproducibility(args.seed)
    shared_profile_path, shared_profile = check_shared_profile(args)
    manifest = ensure_manifest(args)
    class_names = manifest["class_names"]
    profile = f"b{args.baseline_epochs}_m{args.muscle_epochs}_bs{args.batch_size}_a{args.accumulation}"
    profile_dir = dataset_dir(args) / profile
    stage_dir = profile_dir / args.stage
    checkpoint_dir = stage_dir / "checkpoints"
    baseline_weights = profile_dir / "baseline" / "checkpoints" / "best_ACC_weights.pth"
    model = build_stage(args.stage, args, len(class_names), baseline_weights).cuda()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.SGD(
        trainable, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    epochs = args.baseline_epochs if args.stage == "baseline" else args.muscle_epochs
    config = serializable_config(args)
    input_checkpoint = args.pretrained_path if args.stage == "baseline" else baseline_weights
    checkpoint_metadata = (
        {"path": str(input_checkpoint.resolve()), "sha256": file_sha256(input_checkpoint)}
        if input_checkpoint is not None
        else {
            "source": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
            "sha256": "verified by torch.hub check_hash",
        }
    )
    metadata = {
        "scope": "Full-data short-run validation; not a reproduction of paper Table IV values.",
        "environment": environment_info(REPO_ROOT),
        "config": config,
        "manifest_sha256": manifest["sha256"],
        "class_names": class_names,
        "input_checkpoint": checkpoint_metadata,
    }
    write_json(stage_dir / "run_metadata.json", metadata)
    loaders = create_loaders(
        manifest, args.data_root, image_size=args.image_size, batch_size=args.batch_size,
        workers=args.workers, seed=args.seed
    )
    last_path = checkpoint_dir / "last.ckpt"
    start_epoch, best_acc, history = 1, -1.0, []
    backbone_before = clone_state(model.original_net) if args.stage == "muscle" else None
    fusion_before = trainable_state(model) if args.stage == "muscle" else None
    if args.resume == "auto" and last_path.is_file():
        from .engine import load_checkpoint

        start_epoch, best_acc, history = load_checkpoint(
            last_path, model, optimizer, scaler, expected_config=config
        )
    elif args.resume == "none" and last_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing run with --resume none: {last_path}"
        )
    first_evidence = None
    for epoch in range(start_epoch, epochs + 1):
        train_metrics, evidence = train_epoch(
            args.stage, model, loaders["train"], optimizer, scaler, device=torch.device("cuda"),
            amp=args.amp, accumulation=args.accumulation, epoch=epoch, epochs=epochs,
            base_lr=args.learning_rate, annealing_step=args.annealing_step,
            class_names=class_names
        )
        first_evidence = first_evidence or evidence
        val_metrics = evaluate(
            args.stage, model, loaders["val"], class_names=class_names, device=torch.device("cuda"),
            amp=args.amp, epoch=epoch, annealing_step=args.annealing_step
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        append_jsonl(stage_dir / "epoch_log.jsonl", record)
        if not shared_profile_path.exists():
            write_json(shared_profile_path, shared_profile)
        if val_metrics["ACC"] > best_acc:
            best_acc = val_metrics["ACC"]
            save_raw_weights(checkpoint_dir / "best_ACC_weights.pth", model)
        save_checkpoint(
            last_path, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch,
            best_acc=best_acc, history=history, config=config
        )
        write_json(stage_dir / "history.json", history)
        print(json.dumps({"dataset": args.dataset, "stage": args.stage, **record}, ensure_ascii=False))
    best_path = checkpoint_dir / "best_ACC_weights.pth"
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    test_metrics = evaluate(
        args.stage, model, loaders["test"], class_names=class_names, device=torch.device("cuda"),
        amp=args.amp, epoch=epochs, annealing_step=args.annealing_step
    )
    mechanism = None
    if args.stage == "muscle":
        first_evidence = first_evidence or inspect_muscle_evidence(
            model, loaders["val"], torch.device("cuda"), args.amp
        )
        mechanism = {
            "evidence": first_evidence,
            "four_evidence_views": first_evidence is not None and first_evidence["view_count"] == 4,
            "evidence_nonnegative": first_evidence is not None and first_evidence["minimum"] >= 0,
            "backbone_all_parameters_frozen": all(
                not parameter.requires_grad for parameter in model.original_net.parameters()
            ),
            "backbone_parameters_and_buffers_unchanged": state_unchanged(backbone_before, model.original_net),
            "fusion_parameters_updated": any_trainable_changed(fusion_before, model),
        }
        write_json(stage_dir / "mechanism_checks.json", mechanism)
    save_metric_artifacts(test_metrics, stage_dir, "test_metrics")
    comparison = table_iv_comparison(args.dataset, args.stage, test_metrics)
    write_json(stage_dir / "table_iv_difference.json", comparison)
    conclusion = make_conclusion(args.dataset, args.stage, test_metrics)
    (stage_dir / "conclusion_zh.txt").write_text(conclusion, encoding="utf-8")
    write_json(
        stage_dir / "result.json",
        {"test": test_metrics, "mechanism": mechanism, "table_iv_comparison": comparison},
    )
    print(conclusion)
    return 0


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.accumulation < 1 or args.workers < 0:
        raise ValueError("batch-size and accumulation must be positive; workers cannot be negative")
    if args.image_size != 256:
        raise ValueError("The upstream ResNet-50 MUSCLE feature shapes require --image-size 256")
    if args.command == "validate-data":
        return validate_data(args)
    if args.command == "benchmark":
        return benchmark(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
