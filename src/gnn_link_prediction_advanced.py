from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx

try:
    import community as community_louvain
except ImportError as exc:
    raise ImportError(
        "Le package 'python-louvain' est requis. Installe-le avec : pip install python-louvain"
    ) from exc


RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# Chargement du graphe
# ============================================================

def load_graph(file_path: Path) -> nx.Graph:
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


# ============================================================
# Calcul des métriques du graphe pour construire les features
# ============================================================

def compute_graph_statistics(
    graph: nx.Graph,
) -> Tuple[
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    Dict[int, float],
    Dict[int, int],
]:
    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    pagerank_centrality = nx.pagerank(graph)
    clustering = nx.clustering(graph)
    partition = community_louvain.best_partition(graph)

    return (
        degree_centrality,
        closeness_centrality,
        betweenness_centrality,
        pagerank_centrality,
        clustering,
        partition,
    )


# ============================================================
# Features de nœuds
# ============================================================

def build_node_features(graph: nx.Graph) -> Tuple[List[int], np.ndarray, Dict[int, int]]:
    (
        degree_centrality,
        closeness_centrality,
        betweenness_centrality,
        pagerank_centrality,
        clustering,
        partition,
    ) = compute_graph_statistics(graph)

    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    features = []
    for node in nodes:
        features.append([
            graph.degree(node),
            degree_centrality[node],
            closeness_centrality[node],
            betweenness_centrality[node],
            pagerank_centrality[node],
            clustering[node],
            partition[node],
        ])

    x = np.array(features, dtype=np.float32)

    # Normalisation
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    x = (x - x_min) / (x_max - x_min + 1e-8)

    return nodes, x, partition


# ============================================================
# Features de paire
# ============================================================

def pair_features_for_edge(
    graph: nx.Graph,
    u: int,
    v: int,
    partition: Dict[int, int],
) -> List[float]:
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))

    common_neighbors = len(neighbors_u & neighbors_v)
    union_neighbors = len(neighbors_u | neighbors_v)
    jaccard = common_neighbors / union_neighbors if union_neighbors > 0 else 0.0

    adamic_adar = 0.0
    for w in neighbors_u & neighbors_v:
        deg_w = graph.degree(w)
        if deg_w > 1:
            adamic_adar += 1.0 / np.log(deg_w)

    pref_attach = graph.degree(u) * graph.degree(v)
    degree_diff = abs(graph.degree(u) - graph.degree(v))
    same_community = 1.0 if partition[u] == partition[v] else 0.0

    return [
        float(common_neighbors),
        float(jaccard),
        float(adamic_adar),
        float(pref_attach),
        float(degree_diff),
        float(same_community),
    ]


def build_pair_feature_tensor(
    graph: nx.Graph,
    edge_label_index: Tensor,
    partition: Dict[int, int],
    idx_to_node: Dict[int, int],
) -> Tensor:
    rows = []
    edge_array = edge_label_index.cpu().numpy()

    for i in range(edge_array.shape[1]):
        u_idx = int(edge_array[0, i])
        v_idx = int(edge_array[1, i])
        u = idx_to_node[u_idx]
        v = idx_to_node[v_idx]
        rows.append(pair_features_for_edge(graph, u, v, partition))

    x = np.array(rows, dtype=np.float32)

    # Normalisation locale
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    x = (x - x_min) / (x_max - x_min + 1e-8)

    return torch.tensor(x, dtype=torch.float)


# ============================================================
# Conversion NetworkX -> PyG
# ============================================================

def graph_to_pyg_data(graph: nx.Graph) -> Tuple[Data, Dict[int, int], Dict[int, int]]:
    nodes, x, partition = build_node_features(graph)

    data = from_networkx(graph)
    data.x = torch.tensor(x, dtype=torch.float)

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    return data, partition, idx_to_node


# ============================================================
# Hard negative sampling
# ============================================================

def enrich_negative_edges_with_hard_cases(
    graph: nx.Graph,
    edge_label_index: Tensor,
    edge_label: Tensor,
    partition: Dict[int, int],
    idx_to_node: Dict[int, int],
    hard_ratio: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """
    Remplace une partie des négatifs aléatoires par des négatifs difficiles :
    paires non connectées mais avec voisins communs ou même communauté.
    """
    edge_idx_np = edge_label_index.cpu().numpy()
    labels_np = edge_label.cpu().numpy()

    positive_pairs = []
    negative_pairs = []

    for i in range(edge_idx_np.shape[1]):
        u_idx = int(edge_idx_np[0, i])
        v_idx = int(edge_idx_np[1, i])
        if labels_np[i] == 1:
            positive_pairs.append((u_idx, v_idx))
        else:
            negative_pairs.append((u_idx, v_idx))

    num_neg = len(negative_pairs)
    num_hard = int(num_neg * hard_ratio)

    nodes_idx = list(idx_to_node.keys())
    existing_edges = {tuple(sorted(e)) for e in graph.edges()}

    hard_negatives = set()
    attempts = 0
    max_attempts = num_hard * 100

    while len(hard_negatives) < num_hard and attempts < max_attempts:
        attempts += 1
        u_idx, v_idx = np.random.choice(nodes_idx, size=2, replace=False)
        u = idx_to_node[int(u_idx)]
        v = idx_to_node[int(v_idx)]

        if tuple(sorted((u, v))) in existing_edges:
            continue

        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))

        same_comm = partition[u] == partition[v]
        common_neighbors = len(neighbors_u & neighbors_v)

        if same_comm or common_neighbors >= 2:
            hard_negatives.add((int(u_idx), int(v_idx)))

    random_negatives = negative_pairs[: max(0, num_neg - len(hard_negatives))]
    final_negatives = random_negatives + list(hard_negatives)

    final_pairs = positive_pairs + final_negatives
    final_labels = [1] * len(positive_pairs) + [0] * len(final_negatives)

    final_edge_index = torch.tensor(final_pairs, dtype=torch.long).t().contiguous()
    final_edge_label = torch.tensor(final_labels, dtype=torch.float)

    return final_edge_index, final_edge_label


# ============================================================
# Modèle GraphSAGE + décodeur MLP
# ============================================================

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkMLPDecoder(nn.Module):
    def __init__(self, embedding_dim: int, pair_feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4 + pair_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_u: Tensor, z_v: Tensor, pair_feats: Tensor) -> Tensor:
        x = torch.cat(
            [
                z_u,
                z_v,
                torch.abs(z_u - z_v),
                z_u * z_v,
                pair_feats,
            ],
            dim=1,
        )
        return self.mlp(x).view(-1)


class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32, pair_feat_dim: int = 6):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels, dropout=0.3)
        self.decoder = LinkMLPDecoder(out_channels, pair_feat_dim, hidden_dim=64)

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor, pair_feats: Tensor) -> Tensor:
        z = self.encoder(x, edge_index)
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return self.decoder(src, dst, pair_feats)


# ============================================================
# Entraînement / évaluation
# ============================================================

def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, data: Data) -> float:
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index, data.edge_label_index, data.pair_feats)
    loss = F.binary_cross_entropy_with_logits(logits, data.edge_label)

    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model: nn.Module, data: Data) -> Tuple[float, float, float, float, float]:
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_label_index, data.pair_feats)
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
    plt.title("Historique d'entraînement - GraphSAGE Link Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Fonction principale
# ============================================================

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

    print("Conversion vers PyTorch Geometric...")
    data, partition, idx_to_node = graph_to_pyg_data(graph)

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )

    train_data, val_data, test_data = transform(data)

    # Hard negatives
    train_edge_label_index, train_edge_label = enrich_negative_edges_with_hard_cases(
        graph,
        train_data.edge_label_index,
        train_data.edge_label,
        partition,
        idx_to_node,
        hard_ratio=0.5,
    )
    train_data.edge_label_index = train_edge_label_index
    train_data.edge_label = train_edge_label

    # Pair features
    train_data.pair_feats = build_pair_feature_tensor(graph, train_data.edge_label_index, partition, idx_to_node)
    val_data.pair_feats = build_pair_feature_tensor(graph, val_data.edge_label_index, partition, idx_to_node)
    test_data.pair_feats = build_pair_feature_tensor(graph, test_data.edge_label_index, partition, idx_to_node)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphSAGELinkPredictor(
        in_channels=train_data.x.shape[1],
        hidden_channels=64,
        out_channels=32,
        pair_feat_dim=train_data.pair_feats.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

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
    patience = 20
    patience_counter = 0
    max_epochs = 200

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
            "model": "GraphSAGE Link Prediction Advanced",
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1,
            "roc_auc": test_auc,
            "best_val_f1": best_val_f1,
        }
    ])

    history_df = pd.DataFrame(history)

    results_path = output_tables / "gnn_link_prediction_advanced_results.csv"
    history_path = output_tables / "gnn_advanced_training_history.csv"
    plot_path = output_figures / "gnn_advanced_training_history.png"
    model_path = output_tables / "gnn_advanced_best_model.pt"

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