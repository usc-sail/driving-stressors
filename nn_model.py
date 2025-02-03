import torch, torch.nn as nn
from resnet_ts import ResNet1D


class ModalityNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        n_block: int,
    ) -> None:
        super(ModalityNN, self).__init__()
        self.resnet = ResNet1D(
            in_channels=in_channels,
            base_filters=base_filters,
            kernel_size=3,
            stride=1,
            groups=1,
            n_block=n_block,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class MultimodalNN(nn.Module):
    def __init__(
        self,
        n_modalities: int,
        in_channels: int,
        base_filters: int,
        n_block: int,
        n_classes: int,
    ) -> None:
        super(MultimodalNN, self).__init__()

        self.modality_nns = [
            ModalityNN(in_channels, base_filters, n_block) for _ in range(n_modalities)
        ]
        self.modality_nns = nn.ModuleList(self.modality_nns)

        self.rnn = nn.GRU(
            input_size=64 * n_modalities,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run resnet for each modality
        modality_outs = []
        for m in range(len(self.modality_nns)):
            this_mod = x[:, m, :].unsqueeze(1)
            this_out = self.modality_nns[m](this_mod)
            modality_outs.append(this_out)

        # concatenate and run through the RNN
        x = torch.cat(modality_outs, dim=1)  # [bs, n_feats, seq_len]
        x, _ = self.rnn(x.permute(0, 2, 1))  # [bs, seq_len, n_feats]

        # run through the final FC layer
        return self.fc(x.mean(dim=1))


if __name__ == "__main__":
    sample = torch.randn(64, 4, 300)
    model = MultimodalNN(
        n_modalities=4,
        in_channels=1,
        base_filters=32,
        n_block=2,
        n_classes=2,
    )
    print(model(sample.float()).shape)
