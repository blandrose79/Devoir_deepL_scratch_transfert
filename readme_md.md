# Devoir CNN : "From Scratch" vs "Transfer Learning" (Cats & Dogs)

Ce projet est réalisé dans le cadre du **Master 1 en Intelligence Artificielle** (Dakar Institute of Technology). Il a pour objectif de comparer les performances, la vitesse de convergence et la robustesse entre un modèle de réseau de neurones convolutif (CNN) entraîné **"from scratch"** (à partir de zéro) et un modèle utilisant le **transfert d'apprentissage** (*Transfer Learning*) sur le jeu de données *Cats vs Dogs* de Kaggle.

Nous constatons une amélioration spectaculaire des performances du modèle grâce à l'apprentissage par transfert (transfert learning), obtenant de meilleurs résultats en seulement 5 époques, contre 30 avec l'approche from scratch (voir ci-dessous).

---

## 📋 Table des Matières

- [Aperçu du Projet](#-aperçu-du-projet)
- [Architecture du Projet](#-architecture-du-projet)
- [Jeu de Données (Dataset)](#-jeu-de-données-dataset)
- [Méthodologie](#-méthodologie)
  - [1. Modèle CNN From Scratch](#1-modèle-cnn-from-scratch)
  - [2. Modèle en Transfert d'Apprentissage](#2-modèle-en-transfert-dapprentissage)
- [Configuration des Hyperparamètres](#-configuration-des-hyperparamètres)
- [Installation et Configuration](#-installation-et-configuration)
- [Utilisation et performance des methodes](#-utilisation)
- [Reproductibilité](#-reproductibilité)
- [Bibliothèques Utilisées](#-bibliothèques-utilisées)

---

## 🎯 Aperçu du Projet

L'objectif principal est de classifier correctement des images de chats et de chiens. Le projet met en évidence l'impact du transfert d'apprentissage comparativement à une architecture classique construite manuellement en PyTorch.

**Les critères d'évaluation comparatifs sont :**
* La vitesse de convergence de la perte (*Loss*).
* La précision (*Accuracy*) et la matrice de confusion sur les ensembles de validation et de test.
* La robustesse face à la variabilité des données (augmentation de données).

---

## 📁 Architecture du Projet

```text
├── data/
│   ├── Cat_Dog_data/       # Dataset brut initial
│   └── Cat_Dog_ordered/    # Dataset réorganisé pour le chargement
│       ├── cat/
│       └── dog/
├── my_helper.py            # Fonctions utilitaires personnalisées
├── devoir_cnn.ipynb        # Notebook Jupyter principal
├── best_cats_dogs_model.pt # Sauvegarde du meilleur modèle
└── README.md               # Documentation du projet
```

---

## 📊 Jeu de Données (Dataset)

Le projet utilise le jeu de données **Cats vs Dogs** composé d'images nettoyées et réorganisées.Le projet utilise le jeu de données **Cats vs Dogs** composé d'images nettoyées et réorganisées. Ces données ont été téléchargées sur Kaggle qui est une plateforme en ligne portant sur les jeux de données publics mis en ligne par la communauté, téléchargeables et utilisables librement pour la science des données et l'apprentissage automatique (machine learning).
Les fichiers de données (Data Files) sont organisés pour la plupart en dossiers d'entraînement et de test (Train / Test Split).
Pour faciliter le travail demandé dans ce devoir, les données ont été réorganisées et réparties en 3 catégories comme vous pouvez voir ci-dessous :

* **Split des données :**
  * **Train (Entraînement) :** 80% des images
  * **Validation :** 10% des images
  * **Test :** 10% des images
* **Taille des images :** Redimensionnées en $224 \times 224$ pixels.
* **Augmentation de données (Train) :**
  * Redimensionnement (`Resize`)
  * Retournement horizontal aléatoire (`RandomHorizontalFlip`, $p=0.5$)
  * Rotation aléatoire (`RandomRotation`, $\pm 15^\circ$)
  * Normalisation basée sur les statistiques ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## 🛠 Méthodologie

### 1. Modèle CNN "From Scratch"
* Conception d'un réseau convolutif personnalisé comprenant **au minimum 3 couches de convolution** (`Conv2d`), associées à des fonctions d'activation (ex: `ReLU`), du *Pooling* (`MaxPool2d`), et des couches entièrement connectées (*Fully Connected* / `Linear`).
* Définition explicite de la propagation avant (*feedforward*).
* Entraînement complet des poids à partir d'une initialisation aléatoire.

### 2. Modèle en Transfert d'Apprentissage
* Utilisation d'un modèle pré-entraîné sur ImageNet plus precisement image ResNet50.
* Gel (*Freezing*) des couches d'extraction de caractéristiques (*Feature Extractor*).
* Remplacement et entraînement de la tête de classification finale pour la classe binaire (Chat vs Chien).

---

## ⚙️ Configuration des Hyperparamètres

Les hyperparamètres configurés pour l'entraînement sont les suivants :

| Paramètre | Valeur | Description |
| :--- | :--- | :--- |
| `batch_size` | `32` | Taille des lots |
| `lr` | `1e-3` | Taux d'apprentissage (*Learning Rate*) |
| `epochs` | `30` | Nombre d'époques d'entraînement pour la methode from scratch|
| `valid_size` | `0.1` | Proportion du jeu de validation |
| `test_size` | `0.1` | Proportion du jeu de test |
| `train_size` | `0.8` | Proportion du jeu d'entraînement |
| `seed` | `42` | Graine de reproductibilité |

---

## 🚀 Installation et Configuration

1. **Cloner le dépôt ou ouvrir le projet :**
   ```bash
   git clone <url_du_depot>
   cd <nom_du_dossier>
   ```

2. **Créer et activer un environnement virtuel :**
   ```bash
   python -m venv .venv
   # Sur Windows :
   .venv\Scripts\activate
   # Sur Linux/macOS :
   source .venv/bin/activate
   ```

3. **Installer les dépendances nécessaires :**
   ```bash
   pip install torch torchvision torchmetrics pandas numpy matplotlib seaborn tqdm mlflow
   ```

---

## 💻 Utilisation et performance des methodes

1. Les données sont bien placées dans le dossier `data/Cat_Dog_data`.
2. Lancez le notebook Jupyter :
   ```bash
   jupyter notebook
   ```
3. Exécutez les cellules du notebook `devoir_cnn.fromScratch_transfert.ipynb` séquentiellement :
   * **Réorganisation des dossiers :** La cellule dédiée déplace et renomme automatiquement les images vers `data/Cat_Dog_ordered/`.
   * **Chargement des DataLoaders :** Prépare les jeux *train*, *valid* et *test*.
   * **Entraînement & Évaluation :** Entraînement des modèles et suivi des métriques (*Accuracy*, *Loss*, Matrice de confusion).

---
Impact du transfert learning sur la convergence, la performance et la robustesse.
Pour l’entrainement from transfert, l’analyse des métriques de performance (Précision et Rappel) donne des proportions d’au moins 90% pour chaque classe comme il ressort ci-après :
2. Métriques clés de performance
•	Exactitude globale (Accuracy) :
Exactitude (Accuracy)= (1245 + 1214)/2500 = 98,36%
•	Performance pour la classe "cat" :
o	Rappel (Recall) : 1245/(1245 + 14)= 98,89% des chats ont été identifiés. 
o	Précision : 1245/(1245 + 27)= 97,88% des prédictions "chat" étaient exactes. 
•	Performance pour la classe "dog" :
o	Rappel : 1214/(1214 + 27)= 97,82% des chiens ont été identifiés. 
o	Précision : 1214 /(1214 + 14)= 98,86% des prédictions "chien" étaient exactes. 
3. Synthèse et comparaison avec le premier modèle
•	Performance excellente : Avec un taux de réussite global de 98,36%, ce modèle commet très peu d'erreurs (seulement 41 erreurs sur 2500 images). 
•	Efficacité du Transfer Learning : Le titre "Matrice de Confusion transfert" indique l'utilisation du Transfer Learning (apprentissage par transfert). La comparaison montre un saut spectaculaire de performance, passant de 71,32% d'exactitude (sur la matrice précédente) à 98,36% ici
Nous constatons une amélioration spectaculaire des performances du modèle grâce à l'apprentissage par transfert (transfert learning), obtenant de meilleurs résultats en seulement 5 époques, contre 30 avec l'approche from scratch.


## 🎲 Reproductibilité

Pour garantir la reproductibilité absolue des résultats, une fonction `set_seed(42)` est définie au début du script pour fixer les seeds aléatoires de Python, NumPy, PyTorch (CPU & CUDA) ainsi que le comportement déterministe de CuDNN.

```python
set_seed(42)
```

---

## 📚 Bibliothèques Utilisées

* **Deep Learning :** `torch`, `torchvision`, `torchmetrics`
* **Traitement & Analyse de données :** `numpy`, `pandas`
* **Visualisation :** `matplotlib`, `seaborn`
* **Suivi d'expérience & Utilitaires :** `mlflow`, `tqdm`, `pathlib`, `shutil`

![Matrice de confusion from scratch]