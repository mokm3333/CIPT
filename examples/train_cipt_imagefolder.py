"""Minimal ImageFolder training entry for CIPT.

Expected layout:

    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from cipt import build_cipt, cosine_scheduler, evaluate, make_optimizer, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=4, help="number of diverse templates sampled per batch")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    classnames = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    model, preprocess = build_cipt(
        classnames,
        backbone=args.backbone,
        device=device,
        num_diverse_templates=args.k,
    )

    train_set = ImageFolder(train_dir, transform=preprocess)
    val_set = ImageFolder(val_dir, transform=preprocess)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = make_optimizer(model, lr=args.lr)
    scheduler = cosine_scheduler(optimizer, epochs=args.epochs)

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            beta=args.beta,
            gamma=args.gamma,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device=device)
        print(
            f"epoch={epoch + 1:03d} "
            f"loss={train_metrics['loss']:.4f} "
            f"cls={train_metrics['classification']:.4f} "
            f"acc={val_metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()

