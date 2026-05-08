import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rawnet2.model import FMS, RawNet2, ResidualBlock, SincConv


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def base_config():
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


class TestSincConv:
    def test_output_shape(self, device):
        layer = SincConv(
            device=device,
            out_channels=128,
            kernel_size=129,
            freq_scale="mel",
        )
        x = torch.randn(2, 64000)
        x = x.view(2, 1, 64000)
        out = layer(x)
        # (64000 - 129 + 1) = 63872
        assert out.shape == (2, 128, 63872)

    def test_kernel_size_odd(self, device):
        layer = SincConv(
            device=device,
            out_channels=128,
            kernel_size=128,  # even, should be bumped to 129
            freq_scale="mel",
        )
        assert layer.kernel_size == 129

    def test_all_scales(self, device):
        for scale in ["mel", "inverse-mel", "linear"]:
            layer = SincConv(
                device=device,
                out_channels=128,
                kernel_size=129,
                freq_scale=scale,
            )
            x = torch.randn(1, 64000).view(1, 1, 64000)
            out = layer(x)
            assert out.shape == (1, 128, 63872)

    def test_raises_on_bias(self, device):
        with pytest.raises(ValueError, match="bias"):
            SincConv(device=device, out_channels=128, kernel_size=129, bias=True)

    def test_raises_on_groups(self, device):
        with pytest.raises(ValueError, match="groups"):
            SincConv(device=device, out_channels=128, kernel_size=129, groups=2)

    def test_raises_on_multi_channel(self, device):
        with pytest.raises(ValueError, match="input channel"):
            SincConv(device=device, out_channels=128, kernel_size=129, in_channels=2)

    def test_different_scales_produce_different_filters(self, device):
        """Mel, inverse-mel, and linear scales should produce different frequency arrays."""
        layer_mel = SincConv(device=device, out_channels=128, kernel_size=129, freq_scale="mel")
        layer_inv = SincConv(
            device=device, out_channels=128, kernel_size=129, freq_scale="inverse-mel"
        )
        layer_lin = SincConv(device=device, out_channels=128, kernel_size=129, freq_scale="linear")

        # All three should have different frequency arrays
        assert not np.allclose(layer_mel.freq, layer_lin.freq, atol=1)
        assert not np.allclose(layer_mel.freq, layer_inv.freq, atol=1)
        assert not np.allclose(layer_inv.freq, layer_lin.freq, atol=1)

    def test_invalid_scale_raises(self, device):
        """Unknown freq_scale should raise ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown freq_scale"):
            SincConv(device=device, out_channels=128, kernel_size=129, freq_scale="invalid")

    def test_unknown_scale_case_variants(self, device):
        """Only exact 'mel', 'inverse-mel', 'linear' are accepted."""
        with pytest.raises(ValueError, match="Unknown freq_scale"):
            SincConv(device=device, out_channels=128, kernel_size=129, freq_scale="Mel")
        with pytest.raises(ValueError, match="Unknown freq_scale"):
            SincConv(device=device, out_channels=128, kernel_size=129, freq_scale="MEL")


class TestFMS:
    def test_shape_preservation(self, device):
        fms = FMS([128, 128])
        x = torch.randn(2, 128, 100)
        out = fms(x)
        assert out.shape == (2, 128, 100)

    def test_scale_range(self, device):
        fms = FMS([128, 128])
        x = torch.randn(2, 128, 100)
        out = fms(x)
        # FMS applies x * sigmoid(fc(avgpool)) + sigmoid(fc(avgpool))
        # So output should be in a reasonable range
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestResidualBlock:
    def test_no_downsample(self, device):
        block = ResidualBlock([128, 128], first=True)
        x = torch.randn(2, 128, 100)
        out = block(x)
        # After MaxPool1d(3): floor((100 - 3)/3 + 1) = 33
        assert out.shape == (2, 128, 33)

    def test_with_downsample(self, device):
        block = ResidualBlock([128, 512], first=False)
        x = torch.randn(2, 128, 100)
        out = block(x)
        assert out.shape == (2, 512, 33)

    def test_first_no_bn(self, device):
        block = ResidualBlock([128, 128], first=True)
        assert not hasattr(block, "bn1") or block.bn1 is None

    def test_second_has_bn(self, device):
        block = ResidualBlock([128, 128], first=False)
        assert hasattr(block, "bn1")

    def test_fms_applied(self, device):
        block = ResidualBlock([128, 128], first=True)
        x = torch.randn(2, 128, 100)
        out = block(x)
        # FMS should have been applied (check via shape)
        assert out.shape == (2, 128, 33)


class TestRawNet2Shapes:
    """Verify output shapes match paper Table 1."""

    def test_stage1_sincconv_maxpool_bn(self, base_config, device):
        model = RawNet2(base_config, device)
        model.eval()

        x = torch.randn(2, 64000).view(2, 1, 64000)
        x = model.sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = model.first_bn(x)
        x = model.lrelu(x)

        assert x.shape == (2, 128, 21290)

    def test_stage2_resblock1(self, base_config, device):
        model = RawNet2(base_config, device)
        model.eval()

        x = torch.randn(2, 128, 21290)
        for block in model.blocks[:2]:
            x = block(x)

        assert x.shape == (2, 128, 2365)

    def test_stage3_resblock2(self, base_config, device):
        model = RawNet2(base_config, device)
        model.eval()

        x = torch.randn(2, 128, 2365)
        for block in model.blocks[2:]:
            x = block(x)

        assert x.shape == (2, 512, 29)

    def test_stage4_gru(self, base_config, device):
        model = RawNet2(base_config, device)
        model.eval()

        x = torch.randn(2, 512, 29)
        x = model.bn_before_gru(x)
        x = model.lrelu(x)
        x = x.permute(0, 2, 1)
        model.gru.flatten_parameters()
        x, _ = model.gru(x)
        x = x[:, -1, :]

        assert x.shape == (2, 1024)

    def test_stage5_fc_output(self, base_config, device):
        model = RawNet2(base_config, device)
        model.eval()

        x = torch.randn(2, 1024)
        x = model.fc1_gru(x)
        x = model.fc2_gru(x)

        assert x.shape == (2, 2)

    def test_full_forward_train(self, base_config, device):
        model = RawNet2(base_config, device)
        model.eval()

        x = torch.randn(2, 64000)
        out = model(x, is_test=False)

        assert out.shape == (2, 2)
        assert not torch.isnan(out).any()

    def test_full_forward_eval(self, base_config, device):
        model = RawNet2(base_config, device)
        model.eval()

        x = torch.randn(2, 64000)
        out = model(x, is_test=True)

        assert out.shape == (2, 2)
        assert (out >= 0).all() and (out <= 1).all()
        # Softmax sums to 1
        assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_parameter_count_reasonable(self, base_config, device):
        model = RawNet2(base_config, device)
        num_params = sum(p.numel() for p in model.parameters())
        # Should be around 27-30M based on paper architecture
        assert 20_000_000 < num_params < 40_000_000


class TestRawNet2AllScales:
    """Verify model works with all three sinc filter scales."""

    @pytest.mark.parametrize("scale", ["mel", "inverse-mel", "linear"])
    def test_forward_all_scales(self, scale, base_config, device):
        config = {**base_config, "sinc_scale": scale}
        model = RawNet2(config, device)
        model.eval()

        x = torch.randn(2, 64000)
        out = model(x)
        assert out.shape == (2, 2)

    @pytest.mark.parametrize("scale", ["mel", "inverse-mel", "linear"])
    def test_paper_table1_shape(self, scale, base_config, device):
        config = {**base_config, "sinc_scale": scale}
        model = RawNet2(config, device)
        model.eval()

        # Stage 1
        x = torch.randn(2, 64000).view(2, 1, 64000)
        x = model.sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        assert x.shape == (2, 128, 21290)

        # After BN + LReLU
        x = model.first_bn(x)
        x = model.lrelu(x)

        # After all residual blocks
        for block in model.blocks:
            x = block(x)

        assert x.shape == (2, 512, 29)


class TestRawNet2Gradients:
    """Verify gradients flow through the model."""

    def test_backward_no_nan(self, base_config, device):
        model = RawNet2(base_config, device)
        model.train()

        x = torch.randn(2, 64000)
        y = torch.randint(0, 2, (2,))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        out = model(x)
        loss = criterion(out, y)
        assert not torch.isnan(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check all parameters have valid gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
