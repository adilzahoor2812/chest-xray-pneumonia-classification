"""Train ResNet-18 with partial fine-tuning and training logs."""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import (
    BACKBONE_LR_FACTOR,
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    RANDOM_SEED,
    TRAINING_CURVES_PATH,
    TRAINING_LOG_PATH,
    TRAIN_CSV,
    UNFREEZE_FROM_LAYER,
    VAL_CSV,
    WEIGHT_DECAY,
)
from dataset import ChestXrayDataset
from model_factory import build_model, trainable_parameter_groups
from plots import plot_training_curves
from transforms_util import build_eval_transform, build_train_transform


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    all_labels: list[float] = []
    all_probabilities: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = loss_function(logits, labels)
            probabilities = torch.sigmoid(logits)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_probabilities.extend(
                probabilities.cpu().numpy().flatten().tolist()
            )

    average_loss = total_loss / max(len(loader), 1)

    try:
        auc = float(roc_auc_score(all_labels, all_probabilities))
    except ValueError:
        auc = float("nan")

    return average_loss, auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train chest X-ray ResNet-18")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--backbone-lr-factor",
        type=float,
        default=BACKBONE_LR_FACTOR,
    )
    parser.add_argument(
        "--unfreeze-from",
        type=int,
        default=UNFREEZE_FROM_LAYER,
        help="Unfreeze from this ResNet layer onward (1=full, 3=default, 5=head only)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(
        f"Epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
        f"unfreeze_from={args.unfreeze_from}"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = ChestXrayDataset(
        csv_file=TRAIN_CSV,
        data_folder=DATA_DIR,
        transform=build_train_transform(),
    )
    validation_dataset = ChestXrayDataset(
        csv_file=VAL_CSV,
        data_folder=DATA_DIR,
        transform=build_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        pretrained=True,
        unfreeze_from_layer=args.unfreeze_from,
    ).to(device)

    positive_count = int((train_dataset.labels["label"] == 1).sum())
    negative_count = int((train_dataset.labels["label"] == 0).sum())
    if positive_count == 0:
        raise RuntimeError("No positive (Pneumonia) samples in training set.")

    positive_weight = negative_count / positive_count
    print("Negative training images:", negative_count)
    print("Positive training images:", positive_count)
    print("Positive class weight:", round(positive_weight, 2))

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        trainable_parameter_groups(
            model,
            learning_rate=args.lr,
            backbone_lr_factor=args.backbone_lr_factor,
        ),
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_validation_auc = float("-inf")
    history_rows: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_number, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_number % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Batch {batch_number}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        training_loss = running_loss / max(len(train_loader), 1)
        validation_loss, validation_auc = evaluate_model(
            model,
            validation_loader,
            loss_function,
            device,
        )

        print()
        print(f"Epoch {epoch + 1} complete")
        print(f"Training loss:   {training_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation AUC:  {validation_auc:.4f}")
        print()

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": training_loss,
                "val_loss": validation_loss,
                "val_auc": validation_auc,
            }
        )

        if validation_auc == validation_auc:  # not NaN
            scheduler.step(validation_auc)

        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved improved model to {MODEL_PATH}")
            print()

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(TRAINING_LOG_PATH, index=False)
    plot_training_curves(history_df, TRAINING_CURVES_PATH)

    print("Training finished.")
    print(f"Best validation AUC: {best_validation_auc:.4f}")
    print(f"Wrote training log to {TRAINING_LOG_PATH}")
    print(f"Wrote training curves to {TRAINING_CURVES_PATH}")


if __name__ == "__main__":
    main()
