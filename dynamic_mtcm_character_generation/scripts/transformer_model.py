import torch
import torch.nn as nn

class ConsistencyTransformer(nn.Module):
    def __init__(self, input_dim=394, model_dim=256, num_layers=4, num_heads=4, dropout=0.1, use_pose_suggestion=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.masked_head = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, input_dim))
        self.use_pose_suggestion = use_pose_suggestion
        if use_pose_suggestion:
            self.pose_head = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, 7))

    def forward(self, input_tokens, mask_indices, return_attention=False):
        x = self.input_proj(input_tokens)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        masked_out = x[torch.arange(x.size(0)).unsqueeze(1), mask_indices]
        output = {
            "reconstructed_tokens": self.masked_head(masked_out),
            "pose_suggestion": self.pose_head(x.mean(dim=1)) if self.use_pose_suggestion else None,
            "attention": None,
            "pose_embedding": x.mean(dim=1)
        }
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
