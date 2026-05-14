import os
import tempfile

import pytest
import torch
import torch.nn as nn

import wandb
from rawnet2.model import RawNet2


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def base_config():
    return {
        "sinc_filters": 128,
        "sinc_kernel_size": 129,
        "sinc_scale": "mel",
        "resblock_filts": [[128, 128], [128, 512]],
        "resblock_blocks": [2, 4],
        "gru_hidden": 1024,
        "gru_layers": 3,
        "fc_hidden": 1024,
        "num_classes": 2,
    }


class TestWandbOffline:
    """Tests that work without internet using wandb offline mode."""

    def test_offline_init(self):
        with wandb.init(project="test", mode="offline") as run:
            assert run is not None
            # In offline mode, run URL should be None
            assert run.url is None

    def test_offline_log_metrics(self):
        with wandb.init(project="test", mode="offline") as run:
            run.log({"loss": 0.5, "accuracy": 0.9, "epoch": 1})
            assert os.path.exists(run.dir)

    def test_offline_config(self):
        config = {"lr": 0.0001, "batch_size": 32}
        with wandb.init(project="test", config=config, mode="offline") as run:
            assert run.config["lr"] == 0.0001
            assert run.config["batch_size"] == 32

    def test_offline_watch_model(self):
        model = torch.nn.Linear(10, 2)
        with wandb.init(project="test", mode="offline") as run:
            run.watch(model, log="all", log_freq=1)
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            run.log({"dummy_loss": loss.item()})

    def test_offline_log_model_artifact(self):
        model = torch.nn.Linear(10, 2)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            model_path = f.name

        with wandb.init(project="test", mode="offline") as run:
            run.log_model(path=model_path, name="test-model", aliases=["best"])

        os.remove(model_path)


class TestWandbOnline:
    """Tests that require WANDB_API_KEY and internet connection."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        self.api_key = os.environ.get("WANDB_API_KEY", "")
        if not self.api_key:
            pytest.skip("WANDB_API_KEY not set — skipping online tests")

    def test_online_full_flow(self, base_config, device):
        model = RawNet2(base_config, device)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            model_path = f.name

        try:
            with wandb.init(
                project="rawnet2-antispoofing",
                name="test-run-automated",
                group="test-group",
                tags=["test", "automated"],
                config={
                    "test": True,
                    "sinc_scale": "mel",
                    "batch_size": 4,
                },
                mode="online",
            ) as run:
                assert run is not None
                assert run.url is not None

                run.watch(model, log="gradients", log_freq=1)

                x = torch.randn(2, 64000)
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, torch.randint(0, 2, (2,)))

                run.log(
                    {
                        "test/loss": loss.item(),
                        "test/shape_check": x.shape[0],
                    }
                )

                run.log_model(path=model_path, name="test-rawnet2")

                print("Online W&B test completed successfully")

        finally:
            os.remove(model_path)

    def test_online_run_url(self, base_config, device):
        with wandb.init(
            project="rawnet2-antispoofing",
            name="test-run-url",
            mode="online",
        ) as run:
            run.log({"test_metric": 1.0})
            assert run.url is not None
            assert "wandb" in run.url
