import pytest
import torch
import torch.nn as nn

from rawnet2.model import RawNet2
from rawnet2.utils import compute_eer, compute_min_tdcf, get_device


class TestIntegration:
    """End-to-end tests tying all components together."""

    @pytest.fixture
    def config(self):
        return {
            "sinc_filters": 128,
            "sinc_kernel_size": 129,
            "sinc_scale": "mel",
            "resblock_filts": [[128, 128], [128, 512], [512, 512]],
            "resblock_blocks": [2, 4],
            "gru_hidden": 1024,
            "gru_layers": 3,
            "fc_hidden": 1024,
            "num_classes": 2,
        }

    def test_full_pipeline_forward_backward(self, config):
        """Mock data → model → loss → backward. No crash, valid grads."""
        device = get_device()
        model = RawNet2(config, device)
        model.train()

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.0]))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Dummy batch
        x = torch.randn(4, 64000)
        y = torch.randint(0, 2, (4,))

        # Forward
        out = model(x)
        loss = criterion(out, y)

        assert out.shape == (4, 2)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_train_epoch_simulation(self, config):
        """Simulate one epoch: 3 mini-batches."""
        device = get_device()
        model = RawNet2(config, device)
        model.train()

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.0]))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        losses = []
        for _ in range(3):
            x = torch.randn(4, 64000)
            y = torch.randint(0, 2, (4,))

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        assert len(losses) == 3
        assert all(not (loss_val != loss_val) for loss_val in losses)  # No NaN

    def test_eval_flow(self, config):
        """Generate scores → compute EER + t-DCF."""
        device = get_device()
        model = RawNet2(config, device)
        model.eval()

        # Generate mock predictions
        scores_list = []
        labels_list = []

        with torch.no_grad():
            for _ in range(5):
                x = torch.randn(4, 64000)
                out = model(x, is_test=True)
                scores_list.extend(out[:, 1].cpu().numpy())
                labels_list.extend(torch.randint(0, 2, (4,)).numpy())

        import numpy as np

        scores = np.array(scores_list)
        labels = np.array(labels_list)

        eer = compute_eer(scores, labels)
        tdcf = compute_min_tdcf(scores, labels)

        assert eer >= 0
        assert tdcf >= 0

    def test_model_on_different_batch_sizes(self, config):
        """Model should handle different batch sizes."""
        device = get_device()
        model = RawNet2(config, device)
        model.eval()

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 64000)
            out = model(x)
            assert out.shape == (batch_size, 2)

    def test_weighted_vs_unweighted_loss(self, config):
        """Weighted loss should differ from unweighted."""
        device = get_device()
        model = RawNet2(config, device)

        x = torch.randn(4, 64000)
        y = torch.tensor([0, 0, 1, 1])

        out = model(x)

        criterion_unweighted = nn.CrossEntropyLoss()
        loss_unweighted = criterion_unweighted(out, y)

        criterion_weighted = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.0]))
        loss_weighted = criterion_weighted(out, y)

        # Weighted and unweighted should be different
        # (because weight[1]=9.0 affects the loss)
        assert not torch.isclose(loss_weighted, loss_unweighted, atol=1e-3)

    def test_model_state_dict_save_load(self, config):
        """Save and load model state dict."""
        device = get_device()
        model1 = RawNet2(config, device)

        # Get state dict
        state_dict = model1.state_dict()

        # Create new model and load
        model2 = RawNet2(config, device)
        model2.load_state_dict(state_dict)

        # Both should produce same output
        x = torch.randn(2, 64000)
        out1 = model1(x)
        out2 = model2(x)

        assert torch.allclose(out1, out2)

    def test_inference_vs_training_mode(self, config):
        """is_test=True should return probabilities, is_test=False logits."""
        device = get_device()
        model = RawNet2(config, device)
        model.eval()

        x = torch.randn(2, 64000)

        out_train = model(x, is_test=False)
        out_eval = model(x, is_test=True)

        # Eval output should be probabilities (sum to 1)
        assert torch.allclose(out_eval.sum(dim=1), torch.ones(2), atol=1e-5)
        assert (out_eval >= 0).all() and (out_eval <= 1).all()

        # Train output can be any real number (logits)
        assert not torch.allclose(out_train, out_eval)

    def test_batch_norm_running_stats(self, config):
        """Batch norm should update running stats during training."""
        device = get_device()
        model = RawNet2(config, device)
        model.train()

        # Get initial running mean
        bn = model.first_bn
        initial_mean = bn.running_mean.clone()

        # Forward pass
        x = torch.randn(4, 64000)
        _ = model(x)

        # Running mean should have changed
        assert not torch.allclose(bn.running_mean, initial_mean)
