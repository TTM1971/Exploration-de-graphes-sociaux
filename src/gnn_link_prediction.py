# gnn_link_prediction_improved.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx


RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_graph(file_path: Path) -> nx.Graph:
    """Charge le graphe global."""
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    edges = pd.read_csv(
        file_path,
        sep=" ",
        header=None,
        names=["source", "target"],
        compression="gzip",
    )

    graph = nx.from_pandas_edgelist(edges, source="source", target="target")
    return graph


def compute_node_features(graph: nx.Graph) -> Tuple[List[int], np.ndarray]:
    """Construit de vraies features de nœuds à partir du graphe."""
    nodes = list(graph.nodes())

    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    pagerank_centrality = nx.pagerank(graph)
    clustering = nx.clustering(graph)

    x = []
    for node in nodes:
        x.append([
            graph.degree(node),
            degree_centrality[node],
            closeness_centrality[node],
            betweenness_centrality[node],
            pagerank_centrality[node],
            clustering[node],
        ])

    x = np.array(x, dtype=np.float32)

    # Normalisation simple colonne par colonne
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    x = (x - x_min) / (x_max - x_min + 1e-8)

    return nodes, x


def graph_to_pyg_data(graph: nx.Graph) -> Data:
    """Convertit le graphe en Data PyTorch Geometric avec features de nœuds."""
    nodes, x = compute_node_features(graph)
    data = from_networkx(graph)
    data.x = torch.tensor(x, dtype=torch.float)
    return data


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels, dropout=dropout)

    def decode(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor) -> Tensor:
        z = self.encoder(x, edge_index)
        return self.decode(z, edge_label_index)


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, data: Data) -> float:
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index, data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(logits, data.edge_label.float())

    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate(model: nn.Module, data: Data) -> Tuple[float, float, float, float, float]:
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_label_index)
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    labels = data.edge_label.cpu().numpy()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    return accuracy, precision, recall, f1, auc


def plot_training_history(history: Dict[str, List[float]], output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_f1"], label="Val F1")
    plt.plot(history["epoch"], history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Score / Loss")
    plt.title("Historique d'entraînement du GNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "facebook_combined.txt.gz"
    output_tables = base_dir / "outputs" / "tables"
    output_figures = base_dir / "outputs" / "figures"

    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    print("Chargement du graphe...")
    graph = load_graph(data_path)
    print(f"Graphe chargé : {graph.number_of_nodes()} nœuds, {graph.number_of_edges()} arêtes")

    print("Conversion vers PyTorch Geometric avec vraies features...")
    data = graph_to_pyg_data(graph)

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )

    train_data, val_data, test_data = transform(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinkPredictor(
        in_channels=train_data.x.shape[1],
        hidden_channels=64,
        out_channels=32,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    print(f"Device utilisé : {device}")
    print("Début de l'entraînement...")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
    }

    best_val_f1 = -1.0
    best_state = None
    patience = 15
    patience_counter = 0
    max_epochs = 150

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_data)
        val_accuracy, val_precision, val_recall, val_f1, val_auc = evaluate(model, val_data)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Val Prec: {val_precision:.4f} | "
                f"Val Rec: {val_recall:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val AUC: {val_auc:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping à l'époque {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nÉvaluation finale sur le test set...")
    test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate(model, test_data)

    results_df = pd.DataFrame([
        {
            "model": "GCN Link Prediction Improved",
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1,
            "roc_auc": test_auc,
            "best_val_f1": best_val_f1,
        }
    ])

    history_df = pd.DataFrame(history)

    results_path = output_tables / "gnn_link_prediction_improved_results.csv"
    history_path = output_tables / "gnn_training_history.csv"
    plot_path = output_figures / "gnn_training_history.png"
    model_path = output_tables / "gnn_best_model.pt"

    results_df.to_csv(results_path, index=False)
    history_df.to_csv(history_path, index=False)
    plot_training_history(history, plot_path)
    torch.save(model.state_dict(), model_path)

    print("\nRésultats test :")
    print(results_df)
    print(f"\nRésultats sauvegardés dans : {results_path}")
    print(f"Historique sauvegardé dans : {history_path}")
    print(f"Courbe sauvegardée dans : {plot_path}")
    print(f"Meilleur modèle sauvegardé dans : {model_path}")


if __name__ == "__main__":
    main()