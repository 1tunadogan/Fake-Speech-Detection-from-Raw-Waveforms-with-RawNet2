import argparse
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

import wandb

from .dataset import get_eval_dataloader
from .model import RawNet2
from .utils import compute_eer, compute_min_tdcf, get_device


def evaluate(model, eval_loader, device, output_path):
    model.eval()
    all_scores = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(eval_loader, desc="Evaluation"):
            batch_x = batch_x.to(device)

            outputs = model(batch_x, is_test=True)
            scores = outputs[:, 1]
            _, predicted = outputs.max(1)

            all_scores.extend(scores.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    eer = compute_eer(all_scores, all_labels)
    min_tdcf = compute_min_tdcf(all_scores, all_labels)

    # Write scores file
    with open(output_path, "w") as f:
        for i in range(len(all_scores)):
            file_name = os.path.basename(eval_loader.dataset.utterances[i]).replace(".flac", "")
            label_str = "spoof" if all_labels[i] == 1 else "bonafide"
            f.write(f"{file_name} {label_str} {all_scores[i]:.6f}\n")

    return accuracy, eer, min_tdcf


def main():
    parser = argparse.ArgumentParser(description="RawNet2 Anti-Spoofing Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to output scores file")
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Evaluate a randomly initialized model if no checkpoint can be found",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    # W&B init for evaluation
    wandb_config = config.get("wandb", {})
    default_run_name = f"RawNet2-{config['model']['sinc_scale']}-eval"
    run_name = wandb_config.get("name") or default_run_name
    run = wandb.init(
        project=wandb_config.get("project", "rawnet2-antispoofing"),
        entity=wandb_config.get("entity", None),
        name=run_name,
        group=wandb_config.get("group", None),
        tags=wandb_config.get("tags", []) + ["evaluation"],
        notes=wandb_config.get("notes", None),
        config=config,
        mode=wandb_config.get("mode", "online"),
        job_type="eval",
    )

    # Model
    model = RawNet2(
        d_args=config["model"],
        device=device,
        input_length=config["data"]["input_length"],
    ).to(device)

    # Load checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = config["eval"].get("checkpoint", None)

    if checkpoint_path is None:
        # Try to load from W&B artifact
        try:
            model_path = run.use_model(f"RawNet2-{config['model']['sinc_scale']}:best")
            checkpoint_path = model_path
            print(f"Loaded model artifact from W&B: {checkpoint_path}")
        except Exception as e:
            print(f"Could not load model from W&B artifact: {e}")
            # Fallback to local best checkpoint
            checkpoint_path = os.path.join(config["training"]["save_dir"], "best.pth")
            print(f"Falling back to local checkpoint: {checkpoint_path}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    elif args.allow_random_init:
        print("Warning: No checkpoint loaded. Using randomly initialized model.")
    else:
        run.finish()
        raise FileNotFoundError(
            "No checkpoint found for evaluation. Provide --checkpoint, set eval.checkpoint, "
            "make the W&B artifact/local best checkpoint available, or pass --allow-random-init."
        )

    # Eval data loader
    eval_loader = get_eval_dataloader(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        input_length=config["data"]["input_length"],
        sample_rate=config["data"]["sample_rate"],
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", False),
        persistent_workers=config["data"].get("persistent_workers", False),
        subset_fraction=config["data"].get("subset_fraction", 1.0),
        seed=config["training"].get("seed", 1234),
    )

    # Output path
    output_path = args.output
    if output_path is None:
        output_path = config["eval"].get("eval_output", "scores.txt")

    # Evaluate
    accuracy, eer, min_tdcf = evaluate(model, eval_loader, device, output_path)

    print("Evaluation Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  EER: {eer:.2f}%")
    print(f"  min t-DCF: {min_tdcf:.4f}")
    print(f"  Scores saved to: {output_path}")

    # Log to W&B
    run.log(
        {
            "eval/accuracy": accuracy,
            "eval/eer": eer,
            "eval/min_tdcf": min_tdcf,
        }
    )

    run.finish()


if __name__ == "__main__":
    main()
