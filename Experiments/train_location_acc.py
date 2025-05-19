import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
from logger import Logger  # assumes logger is in logger.py
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score,accuracy_score

# Example: from your_model_file import TransformerClassifier, LSTMClassifier
from models import TransformerClassifier, LSTMClassifier,RNNClassifier,GRUClassifier
from data_loader import *


def train(model, loaders, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loaders:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)  # shape: [batch_size, num_classes]
            probs = torch.sigmoid(outputs)  # multi-label prob scores
            preds = (probs > 0.5).float()   # multi-label binary predictions

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_targets.append(y_batch.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    acc = accuracy_score(all_targets, all_preds)  # You can optionally include this
    return acc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    parser = argparse.ArgumentParser(description='Sequential Experiment')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data_type', type=str, default='EVCSAttacks')  # PVAttacks
    parser.add_argument('--log_steps', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()

    torch.set_printoptions(precision=8)
    #device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")

    for bus_name in ['34bus','123bus','8500bus']:
        print(f"\n\n====== Running for dataset: {bus_name} ====== {args.data_type}")

        G, node_voltage, branch_flow, Class_labels = load_dataset_localization_multi(args.data_type, bus_name)
        print(Class_labels)
        dataset = TimeSeriesGraphDataset(
            adj_matrix=nx.to_numpy_array(G),
            node_features=node_voltage,
            edge_features=branch_flow,
            labels=Class_labels
        )
        mean_voltage = [np.mean(np.array(node_voltage[i]), axis=-1) for i in range(len(node_voltage))]
        X0 = np.array(mean_voltage)
        X = [MinMaxScaler().fit_transform(X0[i]) for i in range(len(X0))]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(Class_labels, dtype=torch.float32)

        input_dim = X.shape[-1]
        num_timesteps = X.shape[1]
        hidden_dim = args.hidden_channels
        output_dim = y.shape[-1]
        batch_size = 32

        model_classes = {
            "RNN": RNNClassifier,
            "Transformer": TransformerClassifier,
            "LSTM": LSTMClassifier,
            "GRU": GRUClassifier,
        }

        for model_name, ModelClass in model_classes.items():
            print(f"\n=== Running model: {model_name} on {bus_name} ===")
            logger = Logger(runs=args.runs)
            logger_roc = Logger(runs=args.runs)

            for run_id in range(args.runs):
                print(f"\n--- Run {run_id + 1} ---")
                torch.manual_seed(run_id)
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=run_id)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=run_id)

                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

                model = ModelClass(input_dim, hidden_dim, output_dim, num_timesteps, args.num_layers, args.dropout).to(
                    device)
                if hasattr(model, "reset_parameters"):
                    model.reset_parameters()

                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-2)

                for epoch in range(1, 1 + args.epochs):
                    loss = train(model, train_loader, criterion, optimizer)
                    train_acc = evaluate(model, train_loader)
                    val_acc = evaluate(model, val_loader)
                    test_acc = evaluate(model, test_loader)

                    logger.add_result(run_id, (train_acc, val_acc, test_acc))

                    if epoch % args.log_steps == 0:
                        print(
                            f"Epoch {epoch + 1:02d}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

                print("\nAccuracy Results:")
                logger.print_statistics(run_id)

            print("\nFinal Accuracy over Runs:")
            logger.print_statistics()
if __name__ == "__main__":
    main()
