# classic_ml_link_prediction_improved.py

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import community as community_louvain
except ImportError as exc:
    raise ImportError(
        "Le package 'python-louvain' est requis. Installe-le avec : pip install python-louvain"
    ) from exc


RANDOM_STATE = 42
random.seed(RANDOM_STATE)
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


def split_edges_for_link_prediction(
    graph: nx.Graph,
    test_size: float = 0.2,
    max_positive_samples: int = 20000,
) -> Tuple[nx.Graph, List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Sépare les arêtes en train/test pour une vraie prédiction de liens.
    Les features seront calculées uniquement sur le graphe d'entraînement.
    """
    all_edges = list(graph.edges())
    if len(all_edges) > max_positive_samples:
        all_edges = random.sample(all_edges, max_positive_samples)

    train_edges, test_edges = train_test_split(
        all_edges,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    train_graph = graph.copy()
    train_graph.remove_edges_from(test_edges)

    nodes = list(graph.nodes())

    def sample_negative_edges(n_samples: int, forbidden_graph: nx.Graph, forbidden_edges: set) -> List[Tuple[int, int]]:
        negatives = set()
        while len(negatives) < n_samples:
            u, v = random.sample(nodes, 2)
            if u == v:
                continue
            pair = tuple(sorted((u, v)))
            if forbidden_graph.has_edge(u, v):
                continue
            if pair in forbidden_edges:
                continue
            negatives.add(pair)
        return list(negatives)

    # Négatifs train : absents du train_graph
    train_negatives = sample_negative_edges(
        len(train_edges),
        forbidden_graph=train_graph,
        forbidden_edges=set(),
    )

    # Négatifs test : absents du graphe original et distincts
    all_positive_set = {tuple(sorted(e)) for e in graph.edges()}
    train_negative_set = set(train_negatives)

    test_negatives = set()
    while len(test_negatives) < len(test_edges):
        u, v = random.sample(nodes, 2)
        if u == v:
            continue
        pair = tuple(sorted((u, v)))
        if pair in all_positive_set:
            continue
        if pair in train_negative_set:
            continue
        test_negatives.add(pair)

    return train_graph, train_edges, test_edges, train_negatives, list(test_negatives)


def compute_graph_metrics(
    graph: nx.Graph,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, int], Dict[int, float]]:
    """Calcule les métriques uniquement sur le graphe d'entraînement."""
    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    pagerank_centrality = nx.pagerank(graph)
    clustering = nx.clustering(graph)

    if graph.number_of_edges() > 0:
        partition = community_louvain.best_partition(graph)
    else:
        partition = {node: 0 for node in graph.nodes()}

    return (
        degree_centrality,
        closeness_centrality,
        betweenness_centrality,
        pagerank_centrality,
        partition,
        clustering,
    )


def compute_link_features(
    graph: nx.Graph,
    u: int,
    v: int,
    closeness_centrality: Dict[int, float],
    betweenness_centrality: Dict[int, float],
    pagerank_centrality: Dict[int, float],
    degree_centrality: Dict[int, float],
    partition: Dict[int, int],
    clustering: Dict[int, float],
) -> Dict[str, float]:
    """Calcule les features de lien à partir du graphe d'entraînement."""
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

    preferential_attachment = graph.degree(u) * graph.degree(v)
    degree_diff = abs(graph.degree(u) - graph.degree(v))
    closeness_diff = abs(closeness_centrality[u] - closeness_centrality[v])
    betweenness_diff = abs(betweenness_centrality[u] - betweenness_centrality[v])
    pagerank_diff = abs(pagerank_centrality[u] - pagerank_centrality[v])
    same_community = int(partition[u] == partition[v])
    clustering_diff = abs(clustering[u] - clustering[v])
    degree_centrality_diff = abs(degree_centrality[u] - degree_centrality[v])

    return {
        "common_neighbors": common_neighbors,
        "jaccard": jaccard,
        "adamic_adar": adamic_adar,
        "preferential_attachment": preferential_attachment,
        "degree_u": graph.degree(u),
        "degree_v": graph.degree(v),
        "degree_diff": degree_diff,
        "same_community": same_community,
        "closeness_diff": closeness_diff,
        "betweenness_diff": betweenness_diff,
        "pagerank_diff": pagerank_diff,
        "clustering_u": clustering[u],
        "clustering_v": clustering[v],
        "clustering_diff": clustering_diff,
        "degree_centrality_u": degree_centrality[u],
        "degree_centrality_v": degree_centrality[v],
        "degree_centrality_diff": degree_centrality_diff,
    }


def build_dataset(
    graph: nx.Graph,
    positive_edges: List[Tuple[int, int]],
    negative_edges: List[Tuple[int, int]],
    degree_centrality: Dict[int, float],
    closeness_centrality: Dict[int, float],
    betweenness_centrality: Dict[int, float],
    pagerank_centrality: Dict[int, float],
    partition: Dict[int, int],
    clustering: Dict[int, float],
) -> pd.DataFrame:
    """Construit le dataset supervisé à partir du graphe d'entraînement."""
    rows = []

    for u, v in positive_edges:
        row = compute_link_features(
            graph,
            u,
            v,
            closeness_centrality,
            betweenness_centrality,
            pagerank_centrality,
            degree_centrality,
            partition,
            clustering,
        )
        row["label"] = 1
        rows.append(row)

    for u, v in negative_edges:
        row = compute_link_features(
            graph,
            u,
            v,
            closeness_centrality,
            betweenness_centrality,
            pagerank_centrality,
            degree_centrality,
            partition,
            clustering,
        )
        row["label"] = 0
        rows.append(row)

    return pd.DataFrame(rows)


def get_models():
    """Retourne les modèles améliorés."""
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs", random_state=RANDOM_STATE)),
    ])

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )

    return {
        "Logistic Regression": logreg,
        "Random Forest": rf,
        "Gradient Boosting": gb,
    }


def evaluate_train_test(model, X_train, X_test, y_train, y_test, model_name: str) -> Dict[str, float]:
    """Évalue train vs test pour détecter un éventuel surapprentissage."""
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_train_proba = None
        y_test_proba = None

    results = {
        "model": model_name,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "train_recall": recall_score(y_train, y_train_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "train_f1": f1_score(y_train, y_train_pred),
        "test_f1": f1_score(y_test, y_test_pred),
    }

    if y_train_proba is not None and y_test_proba is not None:
        results["train_auc"] = roc_auc_score(y_train, y_train_proba)
        results["test_auc"] = roc_auc_score(y_test, y_test_proba)

    return results


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    """Validation croisée stratifiée."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    auc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        acc_scores.append(accuracy_score(y_val, y_pred))
        prec_scores.append(precision_score(y_val, y_pred))
        rec_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, y_proba))

    return {
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_precision_mean": float(np.mean(prec_scores)),
        "cv_recall_mean": float(np.mean(rec_scores)),
        "cv_f1_mean": float(np.mean(f1_scores)),
        "cv_auc_mean": float(np.mean(auc_scores)) if auc_scores else np.nan,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "facebook_combined.txt.gz"
    output_dir = base_dir / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement du graphe...")
    graph = load_graph(data_path)
    print(f"Graphe chargé : {graph.number_of_nodes()} nœuds, {graph.number_of_edges()} arêtes")

    print("Séparation train/test des arêtes...")
    train_graph, train_pos, test_pos, train_neg, test_neg = split_edges_for_link_prediction(
        graph,
        test_size=0.2,
        max_positive_samples=20000,
    )

    print(f"Graphe d'entraînement : {train_graph.number_of_nodes()} nœuds, {train_graph.number_of_edges()} arêtes")
    print(f"Positifs train : {len(train_pos)} | Positifs test : {len(test_pos)}")
    print(f"Négatifs train : {len(train_neg)} | Négatifs test : {len(test_neg)}")

    print("Calcul des métriques sur le graphe d'entraînement...")
    (
        degree_centrality,
        closeness_centrality,
        betweenness_centrality,
        pagerank_centrality,
        partition,
        clustering,
    ) = compute_graph_metrics(train_graph)

    print("Construction des datasets train et test...")
    train_df = build_dataset(
        train_graph,
        train_pos,
        train_neg,
        degree_centrality,
        closeness_centrality,
        betweenness_centrality,
        pagerank_centrality,
        partition,
        clustering,
    )

    test_df = build_dataset(
        train_graph,  # IMPORTANT : on calcule les features du test à partir du graphe d'entraînement
        test_pos,
        test_neg,
        degree_centrality,
        closeness_centrality,
        betweenness_centrality,
        pagerank_centrality,
        partition,
        clustering,
    )

    feature_cols = [
        "common_neighbors",
        "jaccard",
        "adamic_adar",
        "preferential_attachment",
        "degree_u",
        "degree_v",
        "degree_diff",
        "same_community",
        "closeness_diff",
        "betweenness_diff",
        "pagerank_diff",
        "clustering_u",
        "clustering_v",
        "clustering_diff",
        "degree_centrality_u",
        "degree_centrality_v",
        "degree_centrality_diff",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    models = get_models()

    train_test_results = []
    cv_results = []

    print("Entraînement, évaluation et validation croisée...")
    for model_name, model in models.items():
        tt_result = evaluate_train_test(model, X_train, X_test, y_train, y_test, model_name)
        cv_result = cross_validate_model(model, X_train, y_train, n_splits=5)
        cv_result["model"] = model_name

        train_test_results.append(tt_result)
        cv_results.append(cv_result)

    train_test_df = pd.DataFrame(train_test_results).sort_values("test_f1", ascending=False).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_results).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)

    overfit_df = train_test_df[["model", "train_f1", "test_f1", "train_auc", "test_auc"]].copy()
    overfit_df["f1_gap"] = overfit_df["train_f1"] - overfit_df["test_f1"]
    overfit_df["auc_gap"] = overfit_df["train_auc"] - overfit_df["test_auc"]

    train_test_path = output_dir / "classic_ml_train_test_results.csv"
    cv_path = output_dir / "classic_ml_cross_validation_results.csv"
    overfit_path = output_dir / "classic_ml_overfitting_check.csv"

    train_test_df.to_csv(train_test_path, index=False)
    cv_df.to_csv(cv_path, index=False)
    overfit_df.to_csv(overfit_path, index=False)

    print("\n=== Résultats train/test ===")
    print(train_test_df)

    print("\n=== Résultats validation croisée ===")
    print(cv_df)

    print("\n=== Vérification du surapprentissage ===")
    print(overfit_df)

    print(f"\nSauvegardé : {train_test_path}")
    print(f"Sauvegardé : {cv_path}")
    print(f"Sauvegardé : {overfit_path}")


if __name__ == "__main__":
    main()