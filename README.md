# PyTorch Basics — Rapport des activités pratiques

## 1. Présentation

Ce dépôt regroupe les travaux pratiques réalisés pour découvrir les bases de
PyTorch et les principales étapes d'un projet de machine learning :

- création et manipulation de tenseurs ;
- calcul automatique des gradients avec Autograd ;
- construction de modèles avec `torch.nn` ;
- définition d'une fonction de coût et d'un optimiseur ;
- entraînement et évaluation de modèles ;
- régression linéaire ;
- classification multi-classe ;
- utilisation de `Dataset` et `DataLoader` ;
- classification d'images avec un réseau dense et un CNN ;
- sauvegarde et rechargement des paramètres d'un modèle.

Les activités sont présentées dans trois notebooks :

| Fichier | Activité principale |
| --- | --- |
| [`lab1.ipynb`](./lab1.ipynb) | Premiers pas avec PyTorch, les tenseurs et Autograd |
| [`reg.ipynb`](./reg.ipynb) | Régression linéaire supervisée |
| [`Activité_pratique_Bases_de_Pytorch.ipynb`](./Activité_pratique_Bases_de_Pytorch.ipynb) | Synthèse et exercices de régression et de classification |

## 2. Environnement de travail

Les notebooks ont été exécutés principalement sur CPU. Le code sélectionne
automatiquement CUDA lorsqu'un GPU compatible est disponible :

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Les bibliothèques utilisées sont notamment :

- PyTorch ;
- `torchvision` pour MNIST ;
- `scikit-learn` pour le jeu de données Iris et la normalisation ;
- Matplotlib pour les visualisations ;
- `torchmetrics` et `torcheval` pour les métriques.

Installation indicative :

```bash
pip install torch torchvision torcheval torchmetrics scikit-learn matplotlib
```

## 3. Activité 1 — Bases de PyTorch (`lab1.ipynb`)

### Objectifs

Cette première activité a permis de prendre en main l'environnement PyTorch et
de comprendre les objets fondamentaux utilisés dans les réseaux de neurones.

### Travaux réalisés

#### Vérification de l'environnement

- import de PyTorch ;
- affichage de la version installée ;
- vérification de la disponibilité de CUDA ;
- sélection du périphérique CPU ou GPU.

#### Création et manipulation de tenseurs

Plusieurs tenseurs ont été créés avec :

- `torch.tensor` ;
- `torch.zeros` ;
- `torch.randn` ;
- `torch.linspace`.

Les propriétés suivantes ont été observées :

- la forme (`shape`) ;
- le type (`dtype`) ;
- le déplacement d'un tenseur vers un `device` ;
- la conversion d'un type entier vers `torch.float32`.

L'activité montre également que les opérations entre tenseurs doivent respecter
des dimensions compatibles. Une addition entre matrices de tailles
incompatibles provoque une erreur, tandis que le produit matriciel est réalisé
avec l'opérateur `@`.

#### Autograd

Le notebook utilise une fonction simple :

```python
y = w * x + b
loss = (y - 10) ** 2
```

Les gradients ont été obtenus de deux façons :

```python
loss.backward()
```

et :

```python
dw, db = torch.autograd.grad(loss, (w, b))
```

Les résultats numériques ont été comparés aux dérivées analytiques. Cette
vérification confirme le fonctionnement du graphe de calcul automatique.

## 4. Activité 2 — Régression linéaire (`reg.ipynb`)

### Objectif

L'objectif est d'apprendre une relation linéaire entre une variable d'entrée
`x` et une variable cible `y`, générée à partir de la relation approximative :

```text
y = 3x + 2 + bruit
```

### Étapes réalisées

1. Génération de 50 valeurs régulièrement espacées entre `-2` et `2`.
2. Ajout d'un bruit aléatoire aux valeurs cibles.
3. Visualisation du nuage de points avec Matplotlib.
4. Construction d'un modèle `nn.Linear(1, 1)`.
5. Utilisation de l'erreur quadratique moyenne `nn.MSELoss`.
6. Mise à jour des paramètres avec un optimiseur.
7. Calcul des gradients avec `loss.backward()`.
8. Mise à jour des poids avec `optimizer.step()`.

### Interprétation

Le modèle doit apprendre un poids proche de `3` et un biais proche de `2`.
L'écart restant s'explique par le bruit ajouté aux données. Cette activité
illustre la boucle d'apprentissage fondamentale :

```text
prédiction → perte → gradients → mise à jour des paramètres
```

## 5. Activité 3 — Bases complètes et exercices (`Activité_pratique_Bases_de_Pytorch.ipynb`)

Ce notebook constitue une synthèse plus complète des notions précédentes et
contient plusieurs exercices appliqués.

### 5.1 Tenseurs, devices et Autograd

Le notebook rappelle :

- les différences entre scalaire, vecteur et matrice ;
- les opérations de reshape, d'indexation et de multiplication matricielle ;
- le calcul de moyennes et de sommes ;
- le choix automatique du CPU ou du GPU ;
- le calcul de gradients pour une fonction de coût.

### 5.2 Réseaux de neurones

Plusieurs architectures sont construites avec :

- `nn.Linear` pour un neurone ;
- `nn.Sequential` pour enchaîner les couches ;
- une classe personnalisée héritant de `nn.Module` ;
- la fonction d'activation `ReLU`.

Le rôle des fonctions de coût et des optimiseurs est également étudié avec
`MSELoss`, `CrossEntropyLoss`, SGD et Adam.

### 5.3 Exercice de régression des salaires

Le salaire mensuel est prédit à partir de deux variables :

- l'ancienneté ;
- le niveau d'expertise.

Les données artificielles suivent approximativement la relation :

```text
salaire = 4000 + 600 × ancienneté + 1500 × expertise + bruit
```

Les étapes réalisées sont :

- génération de 300 exemples ;
- séparation en 80 % d'entraînement et 20 % de test ;
- standardisation des variables d'entrée à partir du seul jeu d'entraînement ;
- conversion des salaires en milliers de dirhams ;
- entraînement d'un modèle `nn.Linear(2, 1)` ;
- suivi de la perte MSE ;
- calcul de l'erreur absolue moyenne (MAE) en dirhams ;
- prédiction du salaire d'un salarié ayant 7 ans d'ancienneté et un niveau
  d'expertise de 4/5.

La prédiction attendue doit être proche de `14 200 DH`, qui correspond à la
valeur théorique sans bruit.

### 5.4 Exercice de classification Iris

Le jeu de données Iris contient 150 fleurs appartenant à trois espèces :

- `setosa` ;
- `versicolor` ;
- `virginica`.

Les quatre variables d'entrée sont les longueurs et largeurs des sépales et
des pétales.

Le travail comprend :

- séparation stratifiée entre entraînement et test ;
- standardisation avec `StandardScaler` ;
- conversion des étiquettes en `torch.long` ;
- construction d'un réseau `4 → 16 → 3` ;
- entraînement avec `CrossEntropyLoss` et Adam ;
- calcul de l'accuracy ;
- tracé de la courbe de perte ;
- génération d'une matrice de confusion ;
- prédiction d'une nouvelle fleur.

La sortie du réseau est constituée de logits. Aucun `Softmax` n'est ajouté,
car `CrossEntropyLoss` applique déjà le traitement nécessaire.

### 5.5 Exercice MNIST avec un réseau dense

MNIST est utilisé pour reconnaître les chiffres manuscrits de `0` à `9`.
Le modèle comprend :

```text
Flatten → Linear(784, 128) → ReLU → Linear(128, 10)
```

Le notebook met en œuvre :

- le chargement des données avec `torchvision.datasets.MNIST` ;
- les mini-lots avec `DataLoader` ;
- cinq époques d'entraînement ;
- le mode `model.train()` pendant l'apprentissage ;
- le mode `model.eval()` et `torch.no_grad()` pendant l'évaluation ;
- la sélection de la classe avec `argmax(dim=1)` ;
- l'accuracy, le rappel, la précision et le score F1 multi-classe ;
- la matrice de confusion ;
- l'affichage de plusieurs images mal classées ;
- la sauvegarde et le rechargement avec `state_dict`.

Les métriques utilisées sont multi-classes, contrairement aux métriques
binaires qui ne conviennent pas directement à MNIST.

### 5.6 Exercice MNIST avec un CNN

Une seconde solution utilise un réseau convolutif :

```text
Conv2d(1, 16) → ReLU → MaxPool2d
Conv2d(16, 32) → ReLU → MaxPool2d
Flatten → Linear(32 × 7 × 7, 128) → ReLU → Linear(128, 10)
```

Les deux opérations de pooling réduisent la taille spatiale :

```text
28 × 28 → 14 × 14 → 7 × 7
```

La première couche linéaire reçoit donc `32 × 7 × 7 = 1568` valeurs.
L'activité comprend l'entraînement, l'évaluation, l'analyse des erreurs et la
sauvegarde des paramètres dans `mnist_cnn.pth`.

## 6. Notions importantes retenues

### Normalisation et fuite de données

Les statistiques de normalisation doivent être calculées sur le jeu
d'entraînement uniquement. Le jeu de test doit être transformé avec les mêmes
statistiques, sans réajuster le scaler.

### `train()` et `eval()`

- `model.train()` active le comportement d'entraînement ;
- `model.eval()` active le comportement d'évaluation ;
- `torch.no_grad()` évite de construire inutilement le graphe des gradients
  pendant les prédictions.

### Logits et `CrossEntropyLoss`

Pour une classification multi-classe, le modèle doit retourner les scores bruts
et les étiquettes doivent être des entiers de type `torch.long`. Il ne faut pas
ajouter de `Softmax` avant `CrossEntropyLoss`.

### Sauvegarde

La méthode recommandée utilisée dans les notebooks consiste à sauvegarder les
paramètres :

```python
torch.save(model.state_dict(), "modele.pth")
```

Le modèle doit ensuite être reconstruit avec la même architecture avant le
chargement de ces paramètres.

## 7. Bilan

Ces trois activités permettent de passer progressivement de la manipulation
élémentaire des tenseurs à l'entraînement de modèles de classification
d'images. Le parcours met en évidence la structure commune d'un projet
PyTorch :

```text
données → tenseurs → modèle → prédictions → perte
→ rétropropagation → mise à jour → évaluation
```

Les exercices montrent aussi que le choix du modèle dépend de la nature des
données : une couche linéaire suffit pour une relation simple, tandis qu'un
CNN est mieux adapté à l'extraction de motifs spatiaux dans les images.

## 8. Exécution des notebooks

Depuis la racine du dépôt :

```bash
jupyter notebook
```

Puis ouvrir le notebook souhaité et exécuter les cellules dans l'ordre.
Certains exercices MNIST téléchargent automatiquement les données dans le
dossier `data/` lors de la première exécution.