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


def train_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    device,
    log_interval,
    run,
    epoch,
    accumulation_steps=1,
    scaler=None,
):
    model.train()
    running_loss = 0.0
    num_correct = 0
    num_total = 0
    global_step = (epoch - 1) * len(train_loader)
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * batch_x.size(0) * accumulation_steps
        _, predicted = outputs.max(1)
        num_correct += (predicted == batch_y).sum().item()
        num_total += batch_x.size(0)
        global_step += 1

        if batch_idx % log_interval == 0:
            run.log(
                {
                    "train/batch_loss": loss.item() * accumulation_steps,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/global_step": global_step,
                }
            )

        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    avg_loss = running_loss / num_total
    train_acc = 100.0 * num_correct / num_total
    return avg_loss, train_acc


def validate_epoch(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    num_correct = 0
    num_total = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Validation"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(batch_x, is_test=False)
                loss = criterion(logits, batch_y)
                outputs = torch.softmax(logits, dim=1)
            _, predicted = outputs.max(1)

            running_loss += loss.item() * batch_x.size(0)
            num_correct += (predicted == batch_y).sum().item()
            num_total += batch_x.size(0)

            all_scores.extend(outputs[:, 1].cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    val_loss = running_loss / num_total
    val_acc = 100.0 * num_correct / num_total

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    val_eer = compute_eer(all_scores, all_labels)
    val_tdcf = compute_min_tdcf(all_scores, all_labels)

    return val_loss, val_acc, val_eer, val_tdcf


def main():
    parser = argparse.ArgumentParser(description="RawNet2 Anti-Spoofing Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        total_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_mem:.1f}GB)")

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
        job_type="train",
    )

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        input_length=config["data"]["input_length"],
        sample_rate=config["data"]["sample_rate"],
        seed=config["training"]["seed"],
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", False),
        persistent_workers=config["data"].get("persistent_workers", False),
        subset_fraction=config["data"].get("subset_fraction", 1.0),
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = RawNet2(
        d_args=config["model"],
        device=device,
        input_length=config["data"]["input_length"],
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    run.watch(model, log="gradients", log_freq=100)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        amsgrad=True,
    )

    if config["training"]["loss"] == "weighted_ce":
        weight = torch.tensor([1.0, 9.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    amp_config = config["training"].get("amp", {})
    use_amp = amp_config.get("enabled", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    accumulation_steps = config["training"].get("accumulation_steps", 1)

    if use_amp:
        print(f"AMP enabled (float16), gradient accumulation: {accumulation_steps}x")

    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    best_eer = float("inf")
    num_epochs = config["training"]["epochs"]
    log_interval = config["training"]["log_interval"]

    es_config = config["training"].get("early_stopping", {})
    es_enabled = es_config.get("enabled", False)
    es_patience = es_config.get("patience", 10)
    es_min_delta = es_config.get("min_delta", 0.0)
    es_mode = es_config.get("mode", "min")
    es_best = float("inf") if es_mode == "min" else float("-inf")
    es_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            log_interval,
            run,
            epoch,
            accumulation_steps=accumulation_steps,
            scaler=scaler,
        )

        val_loss, val_acc, val_eer, val_tdcf = validate_epoch(val_loader, model, criterion, device)

        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/eer": val_eer,
                "val/min_tdcf": val_tdcf,
            }
        )

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Val EER: {val_eer:.2f}% | Val t-DCF: {val_tdcf:.4f}"
        )

        if val_eer < best_eer:
            best_eer = val_eer
            best_path = os.path.join(save_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved (val_eer={val_eer:.2f}%)")

            if wandb_config.get("log_model", True):
                run.log_model(
                    path=best_path,
                    name=f"RawNet2-{config['model']['sinc_scale']}",
                    aliases=["best", f"epoch-{epoch}"],
                )

        epoch_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_path)

        if es_enabled:
            if es_mode == "min":
                improved = not np.isnan(val_eer) and val_eer < es_best - es_min_delta
                current_metric = val_eer
            else:
                improved = not np.isnan(val_acc) and val_acc > es_best + es_min_delta
                current_metric = val_acc

            if improved:
                es_best = current_metric
                es_counter = 0
            else:
                es_counter += 1
                print(f"Early stopping: {es_counter}/{es_patience} epochs without improvement")

                if es_counter >= es_patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best val_eer={es_best:.2f}%")
                    run.log(
                        {
                            "early_stopping/triggered": True,
                            "early_stopping/best_epoch": epoch - es_counter,
                            "early_stopping/best_metric": es_best,
                        }
                    )
                    break

            run.log(
                {
                    "early_stopping/patience_remaining": es_patience - es_counter,
                    "early_stopping/best_metric": es_best,
                    "early_stopping/counter": es_counter,
                }
            )

    run.finish()
    print("Training completed.")


if __name__ == "__main__":
    main()
