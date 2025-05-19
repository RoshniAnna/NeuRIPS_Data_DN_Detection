import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import argparse
import random
from models import MLP
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
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        correct += (out.argmax(dim=1) == y_batch).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Evaluation function with AUC
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    all_probs = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model(X_batch)
        probs = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_batch.cpu().numpy())
        correct += (out.argmax(dim=1) == y_batch).sum().item()

    acc = correct / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return acc, auc


def main():
    parser = argparse.ArgumentParser(description="GNN Model Trainer with Train/Val/Test Split")
    parser.add_argument('--model', type=str, choices='MLP')
    parser.add_argument('--data_type', type=str, default='LineFailures')
    parser.add_argument('--bus', type=str, default='123bus')
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    feat=torch.load(f'MP_features_{args.bus}_{args.data_type}.pt').to(device)
    X = torch.tensor(feat, dtype=torch.float32)
    A, node_fe,edge_fe, Class,N,E = load_data_outage(args.data_type, args.bus)
    #y=torch.tensor(Class[:5000],dtype=torch.long)
    y = torch.tensor(Class, dtype=torch.long)
    models_to_run = ['MP-MLP']
    for model_name in models_to_run:
        print(f"\n================== Running Model: {model_name} ==================on {args.bus}")
        logger_acc = Logger(args.runs)
        logger_auc = Logger(args.runs)

        for run in range(args.runs):
            indices = np.arange(len(X))
            train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=run)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=run)

            X_train,X_val, X_test = X[train_idx],X[val_idx], X[test_idx]
            y_train,y_val, y_test = y[train_idx],y[val_idx], y[test_idx]

            train_data = TensorDataset(X_train, y_train)
            val_data = TensorDataset(X_val, y_val)
            test_data = TensorDataset(X_test, y_test)

            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=args.batch_size)
            test_loader = DataLoader(test_data, batch_size=args.batch_size)

            if model_name in ['MP-MLP']:
                model = MLP( in_channels=len(X[0]),
                        hidden_channels=args.hidden_dim,out_channels=2,num_layers=2,dropout=0.5).to(device)
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
                #print(
                #    f"Epoch {epoch + 1:02d}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
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
