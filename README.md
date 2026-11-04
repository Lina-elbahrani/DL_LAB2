# TP Deep Learning : CNN, Faster R-CNN, Transfer Learning et Vision Transformer sur MNIST



## 1. Objectif
Prendre en main PyTorch et construire des architectures de vision (CNN, Faster R-CNN, VGG16 / AlexNet fine-tunés, ViT) entraînées en **mode GPU** (Kaggle), puis les comparer sur MNIST avec plusieurs métriques : Accuracy, F1-score, Loss, temps d'entraînement.

## 2. Contenu du dépôt
| Fichier | Description |
|---|---|
| `TP_DeepLearning.ipynb` | Notebook complet : toutes les parties, de l'entraînement à la comparaison finale |



## 3. Partie 1 : CNN, Faster R-CNN, transfer learning

### Q1. CNN from scratch
| Élément | Choix |
|---|---|
| Convolutions | 3 blocs `Conv 3x3, stride 1, padding 1` (32, 64, 128 filtres) |
| Normalisation / activation | BatchNorm + ReLU |
| Pooling | MaxPool 2x2 après chaque bloc (28 → 14 → 7 → 3) |
| Fully connected | 1152 → 256 → Dropout(0.5) → 10 |
| Optimiseur | Adam, lr = 1e-3, scheduler StepLR (÷2 toutes les 3 epochs) |
| Régularisation | L2 (weight_decay = 1e-4), Dropout, BatchNorm |
| Entraînement | batch 128, 8 epochs, CrossEntropyLoss, GPU, 60 000 images |

### Q2. Faster R-CNN
Faster R-CNN est un détecteur d'objets. MNIST a donc été transformé en jeu de détection : chaque chiffre 28x28 est placé à une position aléatoire dans une image 128x128, avec sa boîte englobante (label = chiffre + 1, 0 = fond). La classe prédite pour une image est celle de la détection au meilleur score.
- Modèle : `fasterrcnn_mobilenet_v3_large_fpn`, backbone pré-entraîné, tête `FastRCNNPredictor` à 11 classes.
- Optimiseur : SGD (momentum 0.9, weight decay 5e-4), lr = 0.003 avec warmup de 300 itérations, gradient clipping (norme 10), lr ÷10 à la dernière epoch.
- Entraînement : 3 epochs, sous-ensemble de 10 000 images (test : 2 000 images).
- Remarque : une première tentative avec lr = 0.01 avait divergé (`loss = nan`, F1 = 0). Elle a été corrigée avec le warmup et le gradient clipping.

### Q4. Fine-tuning de VGG16 et AlexNet
Poids ImageNet, couches convolutives gelées (seul le classifieur est entraîné), dernière couche remplacée par `Linear(·, 10)`. Images redimensionnées en 64x64, 3 canaux, normalisation ImageNet. Adam lr = 1e-4, 3 epochs, sous-ensemble de 20 000 images.

## 4. Partie 2 : Vision Transformer (ViT) from scratch
Implémentation suivant le tutoriel indiqué dans l'énoncé :
patchify (patchs 4x4, soit 49 patchs) → projection linéaire (d = 64) → token `[CLS]` → encodage positionnel sinusoïdal → 4 blocs Transformer (LayerNorm, attention multi-têtes à 4 têtes, MLP x4, connexions résiduelles) → tête de classification sur `[CLS]`.
Adam lr = 5e-3, weight decay 1e-4, cosine annealing, 10 epochs, batch 128, 60 000 images.

## 5. Résultats (Kaggle, GPU)

| Modèle | Accuracy | F1 (macro) | Loss | Temps (s) | Paramètres |
|---|---|---|---|---|---|
| **CNN** | **0.9939** | **0.9939** | **0.0195** | 76.8 | 390 858 |
| ViT | 0.9870 | 0.9869 | 0.0384 | 118.8 | 201 866 |
| VGG16 (fine-tuné) | 0.9615 | 0.9612 | 0.1184 | 84.1 | 134 301 514 |
| AlexNet (fine-tuné) | 0.9411 | 0.9399 | 0.1775 | **27.0** | 57 044 810 |
| Faster R-CNN | 0.8895 | 0.8890 | 0.9239 (*) | 289.6 | 18 976 354 |

(*) Pour Faster R-CNN, la « Loss » est la loss d'entraînement (somme des 4 pertes RPN + ROI heads). Pour les autres modèles, c'est la loss de test en cross-entropy. Ces valeurs ne sont pas comparables entre elles.

Ajouter ici les courbes (capture du graphique du notebook : train loss, test accuracy, temps d'entraînement).

### Conditions d'entraînement à garder en tête
| Modèle | Données d'entraînement | Epochs |
|---|---|---|
| CNN | 60 000 | 8 |
| ViT | 60 000 | 10 |
| Faster R-CNN | 10 000 | 3 |
| AlexNet / VGG16 | 20 000 | 3 |

La comparaison n'est donc pas à budget égal : les modèles de Faster R-CNN et de transfer learning ont vu 3 à 6 fois moins de données et beaucoup moins d'epochs.

## 6. Interprétation et comparaison

**CNN.** Meilleur résultat : 99.39 % d'accuracy, F1 identique, loss la plus basse (0.0195). Il converge dès les premières epochs (98.6 % après la première) et ne dépasse pas 77 s. Les convolutions (localité, partage de poids, invariance par translation) sont parfaitement adaptées à MNIST, avec seulement 390 k paramètres.

**Faster R-CNN.** Pire résultat (88.95 %) et temps le plus long (290 s, soit près de 4 fois celui du CNN). Ce modèle est conçu pour localiser des objets : son architecture (backbone, FPN, RPN, RoI heads) est surdimensionnée pour reconnaître un chiffre. Sa précision augmente encore à chaque epoch (53.5 % → 84.9 % → 89.0 %) : il est sous-entraîné, avec 3 epochs et 10 000 images seulement. Il faudrait plus de données et d'epochs pour atteindre le CNN.

**VGG16 et AlexNet fine-tunés.** Ils atteignent 96.2 % et 94.1 %, nettement en dessous du CNN malgré des dizaines de millions de paramètres. Les features ImageNet (objets naturels en couleur) sont mal adaptées à des chiffres manuscrits en niveaux de gris, et les couches convolutives sont gelées. VGG16 fait mieux qu'AlexNet (+2 points) mais coûte 3 fois plus de temps. AlexNet est le plus rapide (27 s) mais le moins précis des modèles pré-entraînés.

**ViT.** 98.70 % d'accuracy et 118.8 s : proche du CNN (0.7 point d'écart) avec deux fois moins de paramètres (202 k). Sa loss diminue régulièrement et son accuracy progresse jusqu'à la dernière epoch, grâce au cosine annealing. Il reste derrière le CNN car il n'a pas le biais inductif des convolutions : il doit apprendre la structure spatiale à partir des données, ce qui demande plus de données ou d'epochs. Il est aussi plus lent à entraîner (attention + boucle Python dans le `forward`).

**Conclusion.** Sur un dataset petit et simple comme MNIST, un petit CNN est le meilleur compromis précision / temps / ressources. Le ViT est très compétitif mais plus coûteux, et Faster R-CNN est inadapté à une tâche de simple classification. Le transfer learning est utile surtout quand les données sont peu nombreuses ou proches du domaine source (ImageNet), ce qui n'est pas le cas de MNIST, où un CNN léger entraîné from scratch fait mieux.


## 8. Synthèse personnelle
Ce TP m'a permis de :
- construire et entraîner des réseaux avec PyTorch sur GPU (`Dataset`, `DataLoader`, `nn.Module`, boucle d'entraînement, évaluation) ;
- choisir les hyper-paramètres (noyaux, padding, stride, optimiseur, learning rate, régularisation) et voir leur effet : par exemple un learning rate trop élevé a fait diverger Faster R-CNN (`loss = nan`), corrigé par un warmup et un gradient clipping ;
- évaluer avec plusieurs métriques (Accuracy, F1, Loss, temps d'entraînement) et comprendre qu'une comparaison n'a de sens qu'à conditions égales ;
- adapter un détecteur d'objets (Faster R-CNN) à une tâche de classification ;
- appliquer le fine-tuning de modèles pré-entraînés (VGG16, AlexNet) et comprendre ses limites quand le domaine source est éloigné ;
- implémenter un Vision Transformer from scratch (patchs, attention multi-têtes, token CLS, encodage positionnel) et le comparer au CNN.

