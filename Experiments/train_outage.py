import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import argparse
import random
from models import GNN,GPSModel,GraphTransformer,Graphormer
from data_loader import load_data_outage, load_graph_data
from logger import Logger


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct += (out.argmax(dim=1) == data.y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Evaluation function with AUC
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1)
        labels = data.y.cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels)
        correct += (preds == data.y).sum().item()
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    acc = correct / len(loader.dataset)
    return acc, auc


def main():
    parser = argparse.ArgumentParser(description="GNN Model Trainer with Train/Val/Test Split")
    parser.add_argument('--model', type=str, choices=['GCN', 'SAGE', 'GAT', 'GIN', 'all'], default='all')
    parser.add_argument('--data_type', type=str, default='LineFailures')
    parser.add_argument('--bus', type=str, default='8500bus')
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    A, node_fe, Class = load_data_outage(args.data_type,args.bus)
    data_list = load_graph_data(A, node_fe, Class)

    models_to_run = ['graphormer','GCN', 'SAGE', 'GAT', 'GIN','TAG','UniMP','GPS'] if args.model == 'all' else [args.model]
    #models_to_run = ['graphormer', 'UniMP', 'GPS'] if args.model == 'all' else [args.model]
    for model_name in models_to_run:
        print(f"\n================== Running Model: {model_name} ==================on {args.bus}")
        logger_acc = Logger(args.runs)
        logger_auc = Logger(args.runs)

        for run in range(args.runs):
            indices = np.arange(len(data_list))
            train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=run)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=run)

            train_data = [data_list[i] for i in train_idx]
            val_data = [data_list[i] for i in val_idx]
            test_data = [data_list[i] for i in test_idx]

            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=args.batch_size)
            test_loader = DataLoader(test_data, batch_size=args.batch_size)

            if model_name in ['GCN', 'SAGE', 'GAT', 'GIN','TAG']:
                model = GNN(model_name, in_channels=data_list[0].x.size(1),
                        hidden_channels=args.hidden_dim, num_classes=2).to(device)
            elif model_name=='UniMP':
                model=GraphTransformer(in_channels=data_list[0].x.size(1),
                        hidden_channels=args.hidden_dim, num_classes=2).to(device)
            elif model_name=='graphormer':
                model=Graphormer(in_channels=data_list[0].x.size(1),
                        hidden_channels=args.hidden_dim, num_classes=2).to(device)
            elif model_name=='GPS':
                model=GPSModel(in_channels=data_list[0].x.size(1),
                        hidden_channels=args.hidden_dim, num_classes=2).to(device)
            else:
                raise ValueError(f"Model '{model_name}' is not defined.")

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in tqdm(range(1, args.epochs + 1), desc=f"Run {run+1}"):
                loss, train_acc = train(model, train_loader, optimizer, criterion)
                val_acc, val_auc = evaluate(model, val_loader)
                test_acc, test_auc = evaluate(model, test_loader)
                logger_acc.add_result(run, (train_acc, val_acc, test_acc))
                logger_auc.add_result(run, (0.0, val_auc, test_auc))  # No train AUC
                print(
                    f"Epoch {epoch + 1:02d}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            print("\nAccuracy results:")
            logger_acc.print_statistics(run)
            print("\nROC-AUC results:")
            logger_auc.print_statistics(run)

        print("\n Final Accuracy results:")
        logger_acc.print_statistics()
        print("\nFinal ROC-AUC results:")
        logger_auc.print_statistics()


if __name__ == "__main__":
    main()
