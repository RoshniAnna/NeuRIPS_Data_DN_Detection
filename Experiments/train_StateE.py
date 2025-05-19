import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random
import numpy as np
from data_loader import *
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
import torch



class FeatureImputer(nn.Module):
    def __init__(self, in_channels, hidden_channels, model_name='gcn'):
        super().__init__()

        # Store model name
        self.model_name = model_name.lower()

        # Choose convolution layer
        if self.model_name == 'gcn':
            ConvLayer = GCNConv
        elif self.model_name == 'sage':
            ConvLayer = SAGEConv
        elif self.model_name == 'gat':
            ConvLayer = GATConv
        elif self.model_name == 'gin':
            nn1 = nn.Sequential(nn.Linear(in_channels + 1, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn1)
            nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.conv2 = GINConv(nn2)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Define convolutional layers if not GIN
        if self.model_name != 'gin':
            self.conv1 = ConvLayer(in_channels + 1, hidden_channels)
            self.conv2 = ConvLayer(hidden_channels, hidden_channels)

        self.out = nn.Linear(hidden_channels, 3)

    def forward(self, x, edge_index, mask_flag):
        x = torch.cat([x, mask_flag.unsqueeze(-1)], dim=1)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.out(x)


def r2_score(pred, true):
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0)


def evaluate_graph_imputation(predictions, average='per_graph'):
    """
    Evaluate predictions across multiple graphs.

    Args:
        predictions: List of (pred, true) tuples.
        average: 'per_graph' or 'total'

    Returns:
        dict with MSE, MAE, R2 scores
    """
    if average == 'per_graph':
        mse_list, mae_list, r2_list = [], [], []
        for pred, true in predictions:
            mse = F.mse_loss(pred, true).item()
            mae = F.l1_loss(pred, true).item()
            r2 = r2_score(pred, true).item()
            mse_list.append(mse)
            mae_list.append(mae)
            r2_list.append(r2)
        return {
            'avg_mse': sum(mse_list) / len(mse_list),
            'avg_mae': sum(mae_list) / len(mae_list),
            'avg_r2': sum(r2_list) / len(r2_list)
        }

    elif average == 'total':
        all_preds = torch.cat([pred for pred, _ in predictions], dim=0)
        all_trues = torch.cat([true for _, true in predictions], dim=0)
        return {
            'total_mse': F.mse_loss(all_preds, all_trues).item(),
            'total_mae': F.l1_loss(all_preds, all_trues).item(),
            'total_r2': r2_score(all_preds, all_trues).item()
        }

    else:
        raise ValueError("average must be 'per_graph' or 'total'")


parser = argparse.ArgumentParser()
parser.add_argument('--bus',type=str,default='8500bus',help='One or more feeder names; use space‑separated list'
)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-2)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')


A, node_fe,edge_fe,known_mask=load_data_StateEstimate('StateEstimate', args.bus)
graphs=make_graph_data(A, node_fe)
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=23)
unknown_mask = ~known_mask

# Step 5: Training and Evaluation Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def run_for_model(model_name):
    print(f"\n=== Running model: {model_name.upper()} === on {args.bus}")

    model = FeatureImputer(in_channels=3, hidden_channels=64, model_name=model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for data in train_graphs:
            data = data.to(device)
            mask_flag = torch.ones(data.x.size(0), dtype=torch.float, device=device)
            out = model(data.x, data.edge_index, mask_flag)
            loss = F.mse_loss(out, data.x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_graphs)

    def test():
        model.eval()
        predictions = []
        with torch.no_grad():
            for data in test_graphs:
                data = data.to(device)
                x_masked = data.x.clone()
                x_masked[unknown_mask] = 0.0
                mask_flag = known_mask.to(device).float()
                out = model(x_masked, data.edge_index, mask_flag)
                pred = out[unknown_mask]
                true = data.x[unknown_mask]
                predictions.append((pred.cpu(), true.cpu()))
        return predictions

    # Training Loop
    for epoch in range(1, args.epochs):
        loss = train()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Testing
    predictions = test()
    results = evaluate_graph_imputation(predictions, average='per_graph')
    print(
        f"Per-Graph Evaluation - MSE: {results['avg_mse']:.4f}, MAE: {results['avg_mae']:.4f}, R²: {results['avg_r2']:.4f}")
#'gcn', 'sage',
for model_name in [ 'gat']:
    run_for_model(model_name)