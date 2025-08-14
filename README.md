# NanoGrowth-UNet
NanoGrowth-UNet est une boîte à outils basée sur Python et PyTorch, conçue pour l'analyse de vidéos de microscopie électronique en transmission (TEM) de la croissance de nanoparticules. Elle met en œuvre une architecture U-Net pour deux tâches cruciales : le débruitage d'images et la segmentation de nanoparticules.

Ce projet est structuré pour être facilement utilisable et extensible, idéal pour la recherche et comme projet de portfolio. Il inclut un générateur de données synthétiques pour permettre de tester l'intégralité du pipeline sans avoir besoin de données expérimentales.

## Fonctionnalités
- Architecture U-Net : Implémentation robuste et standard en PyTorch.

- Générateur de Données Synthétiques : Un script pour créer un jeu de données de démonstration.

- Double Tâche : Le modèle peut être entraîné pour le débruitage ou la segmentation binaire.

- Gestion des Données : Utilitaire Dataset personnalisé pour charger efficacement les données de microscopie.

- Augmentation de Données : Utilise albumentations pour améliorer la robustesse du modèle.

- Scripts Modulaires : Scripts séparés et clairs pour l'entraînement (train.py) et l'inférence (predict.py).

## Structure du Projet

NanoGrowth-UNet/
│
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
│
├── models/
│   └── unet.py
│
├── notebooks/
│   └── 1_data_exploration.ipynb
│
├── utils/
│   ├── dataset.py
│   └── transforms.py
│
├── saved_models/
│
├── generate_synthetic_data.py  <-- NOUVEAU
├── train.py
├── predict.py
├── requirements.txt
└── README.md

## Installation
1. Clonez le dépôt :
'''
git clone https://github.com/VOTRE_NOM_UTILISATEUR/NanoGrowth-UNet.git
cd NanoGrowth-UNet
'''

2. Créez un environnement virtuel et installez les dépendances :
'''
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
.\venv\Scripts\activate  # Sur Windows
pip install -r requirements.txt
'''

## Utilisation
1. Préparation des Données
Vous avez deux options :

Option A : Utiliser vos propres données

- Placez vos images brutes dans data/train/images/ et data/val/images/.

- Placez les masques de segmentation correspondants dans data/train/masks/ et data/val/masks/.

- Important : Le nom d'un masque doit correspondre exactement au nom de son image.

Option B : Générer des données synthétiques (Recommandé pour commencer)

Si vous n'avez pas de données, exécutez le script suivant pour créer un jeu de données de démonstration :

'''
python generate_synthetic_data.py
'''

Ce script remplira automatiquement les dossiers data/train et data/val avec des images et des masques.

2. Entraînement du Modèle
Une fois vos données prêtes (réelles ou synthétiques), lancez le script train.py.

'''
# Lancer l'entraînement avec les paramètres par défaut
python train.py

# Lancer l'entraînement avec des paramètres personnalisés
python train.py --epochs 100 --batch-size 4 --lr 0.0001
'''

Le modèle entraîné (best_model.pth) et un graphique de la perte seront sauvegardés dans le dossier saved_models/.

3. Prédiction (Segmentation/Débruitage)
Une fois le modèle entraîné, utilisez predict.py pour l'appliquer à de nouvelles images.

'''
# Lancer la prédiction sur une image unique
python predict.py --model saved_models/best_model.pth --input /chemin/vers/votre/image.png --output /chemin/vers/le/resultat/

# Lancer la prédiction sur un dossier d'images (ex: les images de validation synthétiques)
python predict.py --model saved_models/best_model.pth --input data/val/images/ --output results/
'''