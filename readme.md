# Exploration de graphes sociaux : centralité, communautés, interprétation métier et prédiction de liens

## Description du projet

Ce projet a été réalisé dans le cadre du cours **INF1853 – Introduction à l’intelligence artificielle**.

Il porte sur l’analyse d’un réseau social Facebook à l’aide de techniques de théorie des graphes, de fouille de réseaux et d’apprentissage supervisé.

L’objectif général est de :

- construire un graphe social à partir du dataset **Facebook Social Circles** ;
- identifier les nœuds les plus importants grâce aux mesures de centralité ;
- détecter automatiquement des communautés avec **Louvain** ;
- comparer les communautés détectées aux **circles** fournis dans le dataset ;
- ajouter une **couche d’interprétation métier** pour identifier des acteurs clés et des groupes stratégiques ;
- étendre le projet à une tâche de **prédiction de liens** avec des modèles classiques et un **GNN avancé**.

---

## Objectifs du projet

Le projet cherche à répondre aux questions suivantes :

1. Quels sont les nœuds les plus centraux dans le réseau ?
2. Quelles communautés peut-on détecter automatiquement ?
3. Dans quelle mesure ces communautés recouvrent-elles les **circles** du dataset ?
4. Comment transformer les résultats en indicateurs utiles pour une entreprise ?
5. Peut-on prédire automatiquement si deux utilisateurs devraient être connectés ?

---

## Dataset utilisé

Le projet utilise le dataset **Facebook Social Circles** provenant de la plateforme **SNAP (Stanford Network Analysis Project)**.

### Fichiers exploités

- `facebook_combined.txt.gz` : arêtes du graphe global
- `facebook.tar.gz` : archive contenant les ego-networks détaillés
- `nodeId.edges` : arêtes d’un ego-network
- `nodeId.circles` : groupes sociaux de l’ego
- `nodeId.feat` : caractéristiques anonymisées des nœuds
- `nodeId.egofeat` : caractéristiques de l’utilisateur ego
- `nodeId.featnames` : noms anonymisés des dimensions de caractéristiques

---

## Structure du projet

```text
projet_IA/
├── data/
│   ├── extracted/
│   ├── facebook_combined.txt.gz
│   └── facebook.tar.gz
│
├── notebooks/
│   └── code.ipynb
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── classic_ml_link_prediction.py
│   ├── classic_ml_link_prediction_improved.py
│   ├── gnn_link_prediction.py
│   ├── gnn_link_prediction_improved.py
│   ├── gnn_link_prediction_advanced.py
│   └── ...
│
└── README.md


Modèles classiques testés

  Logistic Regression
  Random Forest
  Gradient Boosting

Modèle neuronal testé

  GraphSAGE avancé
    features de nœuds
    features de paire
    décodeur MLP
    hard negative sampling
