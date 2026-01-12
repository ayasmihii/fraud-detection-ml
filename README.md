# Détection de fraudes bancaires par Machine Learning

## 📌 Contexte
La fraude bancaire représente un enjeu majeur pour les institutions financières :  
elle est rare mais génère des pertes financières importantes.  
Ce projet vise à construire un **système de détection de fraudes** à partir de données transactionnelles réelles, en adoptant une démarche proche des pratiques industrielles.

---

## 🎯 Objectifs du projet
- Analyser des données de transactions fortement déséquilibrées
- Construire un modèle de détection de fraude fiable
- Comparer des approches simples et avancées
- Optimiser la décision finale via un seuil métier
- Justifier chaque choix par des métriques adaptées

---

## 🗂️ Dataset
- **Source** : Credit Card Fraud Detection (Université Libre de Bruxelles)
- **Taille** : 284 807 transactions
- **Fraudes** : 492 (≈ 0.17 %)
- **Variables** :
  - V1 à V28 : variables anonymisées (PCA)
  - Time : temps écoulé
  - Amount : montant de la transaction
  - Class : cible (0 = normal, 1 = fraude)

---

## 🔍 Analyse exploratoire (EDA)
- Mise en évidence du **fort déséquilibre des classes**
- Analyse des montants et du comportement temporel
- Identification des implications métier :
  - l’accuracy est trompeuse
  - priorité au recall et à la Precision–Recall AUC

---

## 🤖 Modélisation

### 1️⃣ Modèle baseline — Régression Logistique
- Modèle simple et interprétable
- Résultats :
  - accuracy élevée mais recall fraude insuffisant
- Objectif : établir un point de référence

### 2️⃣ Régression Logistique pondérée
- Gestion du déséquilibre via `class_weight="balanced"`
- Amélioration forte du recall
- Explosion des faux positifs
- Optimisation du seuil de décision (seuil retenu : **0.85**)

### 3️⃣ Modèle avancé — XGBoost
- Modèle non linéaire adapté aux données tabulaires
- Gestion native du déséquilibre (`scale_pos_weight`)
- Meilleure séparation des classes
- Optimisation du seuil (seuil retenu : **0.50**)

---

## 📊 Comparaison finale des modèles

| Modèle | Seuil | Precision (Fraude) | Recall (Fraude) | Faux positifs | Faux négatifs | ROC-AUC |
|------|------|-------------------|----------------|--------------|--------------|---------|
| Logistic Regression (pondérée) | 0.85 | 0.176 | 0.898 | 412 | 10 | ~0.97 |
| XGBoost | 0.50 | 0.837 | 0.837 | 16 | 16 | ~0.98 |

---

## 🧠 Conclusion métier
- La régression logistique nécessite un seuil très agressif pour atteindre un recall élevé, ce qui dégrade fortement l’expérience client.
- XGBoost offre un **meilleur compromis precision / recall** avec beaucoup moins de faux positifs.
- L’optimisation du seuil est une étape clé pour transformer un modèle ML en **outil de décision opérationnel**.

👉 **XGBoost est retenu comme solution finale**, car il est plus robuste et plus réaliste pour un déploiement en production.

---

## 🛠️ Technologies utilisées
- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📊 Dashboard interactif (Streamlit)

Un mini dashboard Streamlit a été développé afin de démontrer l’utilisation du modèle dans un contexte applicatif.

Fonctionnalités :
- Chargement du modèle XGBoost entraîné
- Simulation de transactions
- Slider interactif pour le seuil de décision
- Mode simplifié (Time, Amount) et mode expert (toutes les variables)
- Exemples préchargés de transaction normale et frauduleuse

Le dashboard illustre la transformation d’un modèle de machine learning en outil de décision exploitable.

---

## ▶️ Lancer le projet
```bash
pip install -r requirements.txt
jupyter notebook