import torch.nn as nn
import torch.nn.functional as F


class SentimentAnalysisModel(nn.Module):
    def __init__(self, d_model, n_head, d_ff, n_layer, num_classes):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = TransformerEncoder(d_model, n_head, d_ff, n_layer)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
