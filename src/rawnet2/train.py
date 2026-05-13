import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from tqdm import tqdm

from .dataset import get_dataloaders
from .model import RawNet2
from .utils import compute_eer, compute_min_tdcf, get_device, set_seed


def train_epoch(train_loader, model, criterion, optimizer, device, log_interval, run, epoch):
    model.train()
    running_loss = 0.0
    num_correct = 0
    num_total = 0
    global_step = epoch * len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        num_correct += (predicted == batch_y).sum().item()
        num_total += batch_x.size(0)
        global_step += 1

        if batch_idx % log_interval == 0:
            run.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/global_step": global_step,
                }
            )

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / num_total
    train_acc = 100.0 * num_correct / num_total
    return avg_loss, train_acc


def validate_epoch(val_loader, model, device):
    model.eval()
    num_correct = 0
    num_total = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Validation"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x, is_test=True)
            _, predicted = outputs.max(1)

            num_correct += (predicted == batch_y).sum().item()
            num_total += batch_x.size(0)

            all_scores.extend(outputs[:, 1].cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    val_acc = 100.0 * num_correct / num_total

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    val_eer = compute_eer(all_scores, all_labels)
    val_tdcf = compute_min_tdcf(all_scores, all_labels)

    return val_acc, val_eer, val_tdcf


def main():
    parser = argparse.ArgumentParser(description="RawNet2 Anti-Spoofing Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seed and device
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # W&B init
    wandb_config = config.get("wandb", {})
    default_run_name = f"RawNet2-{config['model']['sinc_scale']}-train"
    run_name = wandb_config.get("name") or default_run_name
    run = wandb.init(
        project=wandb_config.get("project", "rawnet2-antispoofing"),
        entity=wandb_config.get("entity", None),
        name=run_name,
        group=wandb_config.get("group", None),
        tags=wandb_config.get("tags", []),
        notes=wandb_config.get("notes", None),
        config=config,
        mode=wandb_config.get("mode", "online"),
    )

    # Data loaders
    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        input_length=config["data"]["input_length"],
        sample_rate=config["data"]["sample_rate"],
        seed=config["training"]["seed"],
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = RawNet2(d_args=config["model"], device=device).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Watch model with W&B
    run.watch(model, log="all", log_freq=100)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        amsgrad=True,
    )

    # Loss function
    if config["training"]["loss"] == "weighted_ce":
        weight = torch.FloatTensor([1.0, 9.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    # Training loop
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    num_epochs = config["training"]["epochs"]
    log_interval = config["training"]["log_interval"]

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            train_loader, model, criterion, optimizer, device, log_interval, run, epoch
        )

        val_acc, val_eer, val_tdcf = validate_epoch(val_loader, model, device)

        # Log epoch metrics
        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/accuracy": val_acc,
                "val/eer": val_eer,
                "val/min_tdcf": val_tdcf,
            }
        )

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | Val EER: {val_eer:.2f}% | Val t-DCF: {val_tdcf:.4f}"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(save_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved (val_acc={val_acc:.2f}%)")

            # Log model artifact to W&B
            if wandb_config.get("log_model", True):
                run.log_model(
                    path=best_path,
                    name=f"RawNet2-{config['model']['sinc_scale']}",
                    aliases=["best", f"epoch-{epoch}"],
                )

        # Save checkpoint every epoch
        epoch_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_path)

    run.finish()
    print("Training completed.")


if __name__ == "__main__":
    main()
