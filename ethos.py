import math, torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 31):  # max_len + 1 for CLS token
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PhysioBind(nn.Module):
    def __init__(
        self,
        input_dim: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_len: int = 30,
        num_classes: int = 2,
    ):
        super().__init__()

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding (max_len + 1 for CLS token)
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.d_model = d_model

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input tensor (batch_size, seq_len, input_dim)
            src_mask: Optional mask for padding (batch_size, seq_len)
        Returns:
            classification: Binary classification output (batch_size, 1)
            sequence_output: Full sequence output (batch_size, seq_len + 1, d_model)
        """
        batch_size = src.shape[0]

        # Project input to d_model dimensions
        x = self.input_projection(src)

        # Scale embeddings
        x = x * math.sqrt(self.d_model)

        # Expand CLS token to batch size and concatenate
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Update mask if provided to include CLS token
        padding_mask = None
        if src_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=src_mask.device
            )
            padding_mask = ~torch.cat((cls_mask, src_mask), dim=1)

        # Pass through transformer encoder
        sequence_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Use CLS token output for classification
        cls_output = sequence_output[:, 0]
        return self.classifier(cls_output)


def example_usage():
    # Create random input data
    batch_size = 32
    seq_len = 25  # Can be <= 30
    features = 20

    model = PhysioBind(
        input_dim=20,  # Your feature dimension
        d_model=64,  # Internal transformer dimension
        nhead=4,  # Number of attention heads
        num_layers=2,  # Number of transformer layers
        dim_feedforward=128,  # Feedforward network dimension
        dropout=0.2,  # Dropout rate
        max_len=30,  # Maximum sequence length
        num_classes=2,  # Number of classes
    )
    x = torch.randn(batch_size, seq_len, features)

    # output: (batch_size, num_classes)
    return model(x)


if __name__ == "__main__":
    result = example_usage()
    print(result.shape)
