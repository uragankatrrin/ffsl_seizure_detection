import torch
import torch.nn as nn
import math

class FCNet(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=3):
        super(FCNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class TransformerClassifier(nn.Module):
    def __init__(self, emb_size=256, num_layers=3, nhead=1, dim_feedforward=128, n_classes=3, max_len=1024):
        super(TransformerClassifier, self).__init__()
        self.projection = nn.Linear(emb_size, emb_size)
        self.positional_encoding = PositionalEncoding(d_model=emb_size, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.2, 
            activation="gelu",
            batch_first=True  # batch first ensures (batch, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pooling = nn.AdaptiveAvgPool1d(1) 
        self.classifier = nn.Linear(emb_size, n_classes)  # This expects (batch_size, emb_size) as input

    def forward(self, x):
        if x.dim() == 2:  # Input is (seq_len, emb_size)
            x = x.unsqueeze(1)  # Add batch dimension: (1, seq_len, emb_size)

        # Projection layer
        x = self.projection(x)  # Shape: (batch, seq_len, emb_size)
        # Positional encoding
        x = self.positional_encoding(x)  # Add positional encodings
        # Transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch, seq_len, emb_size)
        # Pooling
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)  # Pooling over sequence
        # Classification
        x = self.classifier(x)  # Final classification layer

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        """
        Positional encoding module to add positional information to embeddings.
        
        Args:
            d_model: The embedding dimension size.
            dropout: Dropout rate applied after adding positional encodings.
            max_len: The maximum sequence length for which positional encodings are precomputed.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings for a range of sequence lengths.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model).
        Returns:
            Tensor of shape (batch, seq_len, d_model) with positional encodings added.
        """
        seq_len = x.size(1)  # Get the sequence length from the input
        if seq_len > self.pe.size(1):  # If sequence length exceeds max_len
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds the maximum precomputed length ({self.pe.size(1)}). "
                "Increase max_len in PositionalEncoding."
            )
        # Add positional encodings to input embeddings
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)