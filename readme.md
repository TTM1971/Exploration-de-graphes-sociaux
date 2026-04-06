# Exploration de graphes sociaux : centralité, communautés et structures relationnelles

## Description du projet

Ce projet a été réalisé dans le cadre du cours **INF1853 – Introduction à l’intelligence artificielle**.  
Il porte sur l’analyse d’un réseau social Facebook à l’aide de techniques de théorie des graphes et de fouille de réseaux.

L’objectif principal est de :

- construire un graphe social à partir du dataset **Facebook Social Circles** ;
- identifier les nœuds les plus importants à l’aide de mesures de centralité ;
- détecter des communautés d’utilisateurs fortement connectés ;
- comparer les communautés détectées automatiquement aux **circles** fournis dans le dataset ;
- analyser à la fois le **graphe global** et certains **ego-networks** locaux.

---

## Jeu de données utilisé

Le projet utilise le dataset **Facebook Social Circles** provenant de la plateforme **SNAP (Stanford Network Analysis Project)**.

### Fichiers exploités

- `facebook_combined.txt.gz` : arêtes du graphe global
- `facebook.tar.gz` : archive contenant les ego-networks détaillés
- `nodeId.edges` : arêtes d’un ego-network
- `nodeId.circles` : cercles sociaux de l’ego
- `nodeId.feat` : caractéristiques anonymisées des nœuds
- `nodeId.egofeat` : caractéristiques de l’utilisateur ego
- `nodeId.featnames` : noms anonymisés des dimensions de caractéristiques

---

## Objectifs du projet

Ce projet vise à répondre aux questions suivantes :

1. Quels sont les nœuds les plus centraux dans le réseau ?
2. Quelles communautés peut-on détecter automatiquement dans le graphe ?
3. Dans quelle mesure les communautés détectées ressemblent-elles aux cercles sociaux fournis dans les données ?
4. Comment la structure locale varie-t-elle d’un ego-network à un autre ?

---

## Méthodologie

Le projet a été structuré en plusieurs étapes :

1. lecture du graphe global à partir de `facebook_combined.txt.gz` ;
2. construction du graphe non orienté avec **NetworkX** ;
3. calcul des statistiques descriptives du réseau ;
4. calcul des mesures de centralité ;
5. détection de communautés avec l’algorithme **Louvain** ;
6. extraction et analyse détaillée de deux ego-networks : **0** et **107** ;
7. comparaison entre les communautés détectées et les `circles` du dataset ;
8. sauvegarde des tableaux et figures pour le rapport.

---

## Méthodes principales utilisées

### Statistiques de graphe
- nombre de nœuds
- nombre d’arêtes
- densité
- degré moyen
- nombre de composantes connexes
- coefficient de clustering moyen

### Mesures de centralité
- Degree Centrality
- Closeness Centrality
- Betweenness Centrality
- PageRank

### Détection de communautés
- **Louvain** (méthode principale retenue)

### Comparaison avec les cercles
- comparaison des tailles des groupes
- mesure de chevauchement
- indice de Jaccard simplifié

---

## Structure du projet

```text
PROJET_IA/
│
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
└── README.md
