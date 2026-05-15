import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    def __init__(
        self,
        device,
        out_channels,
        kernel_size,
        in_channels=1,
        sample_rate=16000,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        freq_scale="mel",
    ):
        super(SincConv, self).__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel (got {in_channels})")

        # Store out_channels boundaries so forward() can iterate
        # freqs.numel()-1 adjacent pairs and produce exactly out_channels filters.
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        if self.kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        # Initialize filterbanks
        nfft = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(nfft / 2) + 1)

        if freq_scale == "mel":
            fmel = self._hz_to_mel(f)
            fmelmax = np.max(fmel)
            fmelmin = np.min(fmel)
            filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
            filbandwidthsf = self._mel_to_hz(filbandwidthsmel)
            freq_values = filbandwidthsf

        elif freq_scale == "inverse-mel":
            fmel = self._hz_to_mel(f)
            fmelmax = np.max(fmel)
            fmelmin = np.min(fmel)
            filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 2)
            filbandwidthsf = self._mel_to_hz(filbandwidthsmel)
            mel_freqs = filbandwidthsf[: self.out_channels + 1]
            freq_values = np.abs(np.flip(mel_freqs) - 1)

        elif freq_scale == "linear":
            fmin = np.min(f)
            fmax = np.max(f)
            filbandwidths = np.linspace(fmin, fmax, self.out_channels + 1)
            freq_values = filbandwidths

        else:
            raise ValueError(
                f"Unknown freq_scale: {freq_scale}. Expected one of: 'mel', 'inverse-mel', 'linear'"
            )

        hsupp = torch.arange(
            -(self.kernel_size - 1) / 2,
            (self.kernel_size - 1) / 2 + 1,
            dtype=torch.float32,
        )
        self.register_buffer("hsupp", hsupp)
        self.register_buffer("freq", torch.tensor(freq_values, dtype=torch.float32))

    @staticmethod
    def _hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x):
        freqs = self.freq.to(device=x.device)
        hsupp = self.hsupp.to(device=x.device, dtype=x.dtype)
        window = torch.hamming_window(
            self.kernel_size,
            periodic=False,
            device=x.device,
            dtype=x.dtype,
        )

        band_pass = []
        for i in range(freqs.numel() - 1):
            fmin = freqs[i]
            fmax = freqs[i + 1]
            h_high = (2 * fmax / self.sample_rate) * torch.sinc(2 * fmax * hsupp / self.sample_rate)
            h_low = (2 * fmin / self.sample_rate) * torch.sinc(2 * fmin * hsupp / self.sample_rate)
            h_ideal = h_high - h_low
            band_pass.append(window * h_ideal)

        filters = torch.stack(band_pass, dim=0).unsqueeze(1)

        return F.conv1d(
            x,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class FMS(nn.Module):
    def __init__(self, nb_filts):
        super(FMS, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filts[-1], nb_filts[-1])
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x).view(x.size(0), -1)
        y = self.fc(y)
        y = self.sig(y).view(y.size(0), y.size(1), -1)
        return x * y + y


class ResidualBlock(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(ResidualBlock, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            padding=1,
            kernel_size=3,
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=0,
                kernel_size=1,
                stride=1,
            )
        else:
            self.downsample = False

        self.mp = nn.MaxPool1d(3)
        self.fms = FMS(nb_filts)

    def forward(self, x):
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        out = self.fms(out)

        return out


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class RawNet2(nn.Module):
    def __init__(self, d_args, device, input_length=64000):
        super(RawNet2, self).__init__()

        self.device = device

        self.ln = LayerNorm(input_length)

        self.sinc_conv = SincConv(
            device=self.device,
            out_channels=d_args["sinc_filters"],
            kernel_size=d_args["sinc_kernel_size"],
            in_channels=1,
            freq_scale=d_args["sinc_scale"],
        )

        self.first_bn = nn.BatchNorm1d(num_features=d_args["sinc_filters"])
        self.selu = nn.SELU(inplace=True)

        # Build residual blocks
        self.blocks = nn.ModuleList()
        filts = d_args["resblock_filts"]
        blocks_config = d_args["resblock_blocks"]

        block_idx = 0
        for group_idx, num_blocks in enumerate(blocks_config):
            for b in range(num_blocks):
                first = block_idx == 0
                if b == 0 and group_idx > 0:
                    # Transition block: channel size changes
                    in_ch = filts[group_idx - 1][-1]
                    out_ch = filts[group_idx][-1]
                else:
                    in_ch = filts[group_idx][-1]
                    out_ch = filts[group_idx][-1]

                self.blocks.append(ResidualBlock([in_ch, out_ch], first=first))
                block_idx += 1

        self.bn_before_gru = nn.BatchNorm1d(num_features=filts[-1][-1])

        self.gru = nn.GRU(
            input_size=filts[-1][-1],
            hidden_size=d_args["gru_hidden"],
            num_layers=d_args["gru_layers"],
            batch_first=True,
        )

        self.fc1_gru = nn.Linear(in_features=d_args["gru_hidden"], out_features=d_args["fc_hidden"])
        self.fc2_gru = nn.Linear(
            in_features=d_args["fc_hidden"], out_features=d_args["num_classes"], bias=True
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0001)
            elif isinstance(m, nn.BatchNorm1d):
                pass
            else:
                if hasattr(m, "weight"):
                    nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")

    def forward(self, x, is_test=False):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = self.ln(x)
        x = x.view(nb_samp, 1, len_seq)

        x = self.sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        for block in self.blocks:
            x = block(x)

        x = self.bn_before_gru(x)
        x = self.selu(x)

        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]

        x = self.fc1_gru(x)
        x = self.fc2_gru(x)

        if not is_test:
            return x
        else:
            return F.softmax(x, dim=1)
