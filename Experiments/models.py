import numpy as np
import torch
import torch.nn as nn

import torch.nn as nn

from torch.nn import Linear, Embedding
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, TAGConv, ChebConv, ARMAConv,
    TransformerConv, GPSConv, global_mean_pool
)

# Define the Transformer model with Batch Normalization
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, n_layers,dropout_prob, n_heads=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_timesteps, hidden_dim))
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers,
                                          num_decoder_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Flatten the output of the transformer
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, src):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, feature)
        transformer_output = self.transformer.encoder(src_emb)
        transformer_output = transformer_output.permute(1, 0, 2).contiguous().view(src.size(0), -1)  # Flatten
        transformer_output = self.dropout(transformer_output)
        predictions = self.fc(transformer_output)
        return predictions


class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_layers, dropout_prob):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob,
                          nonlinearity='tanh')
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Adjust to match flattened output dimensions

    def forward(self, x):
        rnn_output, _ = self.rnn(x)  # (batch_size, seq_len, hidden_dim)
        rnn_output = self.dropout(rnn_output)  # Apply dropout to RNN output
        flattened_output = rnn_output.contiguous().view(x.size(0), -1)  # Flatten all time steps
        predictions = self.fc(flattened_output)  # Fully connected layer for classification
        return predictions


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_layers, dropout_prob):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Corrected for seq_len (num_timesteps)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        lstm_output = self.dropout(lstm_output)  # Apply dropout to LSTM output
        flattened_output = lstm_output.contiguous().view(x.size(0), -1)  # Flatten all time steps
        predictions = self.fc(flattened_output)  # Fully connected layer for classification
        return predictions

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_layers, dropout_prob):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Adjust to match flattened output dimensions

    def forward(self, x):
        gru_output, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)
        gru_output = self.dropout(gru_output)  # Apply dropout to GRU output
        flattened_output = gru_output.contiguous().view(x.size(0), -1)  # Flatten all time steps
        predictions = self.fc(flattened_output)  # Fully connected layer for classification
        return predictions

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class GraphBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, model_type='GCN'):
        super(GraphBlock, self).__init__()
        self.model_type = model_type
        if model_type == 'GIN':
            nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.gnn = GINConv(nn1)
        else:
            self.gnn = GCNConv(input_dim, hidden_dim)

    def forward(self, x, edge_index):
        return self.gnn(x, edge_index)


class TGCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_layers=1, dropout=0.5, model_type='GCN'):
        super(TGCNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.model_type = model_type

        self.graph_block = GraphBlock(input_dim, hidden_dim, model_type)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.graph_block.gnn.reset_parameters()
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x_seq, edge_index):
        # x_seq: [batch, T, N, F]
        batch_size, T, N, F = x_seq.shape
        #print(batch_size,T,N,F)
        # x_seq = x_seq.view(-1, N, F)  # [B*T, N, F]
        # x_seq = x_seq.transpose(0, 1)  # [N, B*T, F]
        x_seq = x_seq.view(batch_size, T, N, F)

        outputs = []
        for t in range(T):
            #print(f"x_seq[t].shape = {x_seq[t].shape}, trying to reshape to ({batch_size}, {N}, {F})")
            # xt = x_seq[t].reshape(batch_size, N, F)  # [B, N, F]
            xt = x_seq[:, t, :, :]
            xt_out = []
            for b in range(batch_size):
                x = xt[b]  # [N, F]
                x_gnn = self.graph_block(x, edge_index)  # [N, hidden]
                x_pool = torch.mean(x_gnn, dim=0)  # mean pooling per graph
                xt_out.append(x_pool)
            xt_out = torch.stack(xt_out)  # [B, hidden]
            outputs.append(xt_out)

        seq = torch.stack(outputs, dim=1)  # [B, T, hidden]
        lstm_out, _ = self.lstm(seq)  # [B, T, hidden]
        final_out = lstm_out[:, -1, :]  # [B, hidden]
        final_out = self.dropout(final_out)
        return self.classifier(final_out)




# Standard GNNs
class GNN(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.model_type = model_type

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=False)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        elif model_type == 'GIN':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        elif model_type == 'TAG':
            self.conv1 = TAGConv(in_channels, hidden_channels)
            self.conv2 = TAGConv(hidden_channels, hidden_channels)
        elif model_type == 'Cheb':
            self.conv1 = ChebConv(in_channels, hidden_channels, K=3)
            self.conv2 = ChebConv(hidden_channels, hidden_channels, K=3)
        elif model_type == 'ARMA':
            self.conv1 = ARMAConv(in_channels, hidden_channels)
            self.conv2 = ARMAConv(hidden_channels, hidden_channels)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        #node_means = x.mean(dim=1)
        #print(len(node_means))
        return self.lin(x)


# Transformer-based GNNs
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GPSModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)

        self.conv1 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.conv2 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)



class Graphormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2, heads=4, max_degree=10):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)

        # Structural encodings (e.g., node degree encoding as Graphormer does)
        self.degree_emb = Embedding(max_degree + 1, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, concat=True)
            )

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, deg=None):
        x = self.input_proj(x)

        if deg is not None:
            deg = deg.clamp(max=self.degree_emb.num_embeddings - 1)
            x = x + self.degree_emb(deg)

        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x + residual)

        x = global_mean_pool(x, batch)
        return self.classifier(x)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)