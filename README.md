# NanoGrowth-UNet : Segmentation et Analyse Quantitative pour la Microscopie
NanoGrowth-UNet est une boîte à outils complète pour l'analyse d'images de microscopie, spécialisée dans la croissance de nanoparticules. Ce projet utilise un réseau de neurones U-Net (PyTorch) pour la segmentation sémantique, couplé à des techniques de traitement d'images avancées pour la segmentation d'instance et l'analyse statistique.

Le but est de passer d'une image de microscope bruitée à une analyse quantitative complète des particules observées.

## Fonctionnalités
- Segmentation Sémantique : Utilise une architecture U-Net robuste pour générer des masques binaires (particules vs fond).

- Segmentation d'Instance : Implémente l'algorithme Watershed pour séparer avec précision les particules qui se touchent ou s'agglomèrent.

- Analyse Quantitative : Un script d'analyse complet (analysis.py) pour extraire des métriques clés pour chaque particule détectée :

 - Aire et périmètre.

 - Diamètre équivalent.

 - Estimation de volume 3D (en supposant une géométrie sphérique).

 - Indice de circularité pour l'analyse de forme.

- Rapports et Visualisations : Génère automatiquement un rapport statistics.csv et des graphiques de distribution (taille, aire...).

- Générateur de Données Synthétiques : Inclut un script pour créer un jeu de données de test avec des particules qui se chevauchent.

## Structure du Projet

NanoGrowth-UNet/
│
├── data/                     # Données d'entraînement et de validation
├── models/                   # Architecture U-Net
├── notebooks/                # Notebook d'exploration
├── results/                  # Dossier pour les résultats d'analyse (rapports, graphiques)
├── utils/                    # Utilitaires (Dataset, transforms)
│
├── generate_synthetic_data.py # Script pour créer des données de test
├── train.py                  # Script pour entraîner le modèle U-Net
├── predict.py                # Script pour la segmentation binaire
├── analysis.py               # NOUVEAU: Script pour l'analyse quantitative
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

À ce stade, predicted_masks/ contient des images en noir et blanc où toutes les particules sont fusionnées.

4. Analyse Quantitative
C'est ici que la nouvelle fonctionnalité prend tout son sens. Lancez le script d'analyse sur les masques que vous venez de générer.

'''
python analysis.py --input predicted_masks/ --output results/
'''

Ce que fait ce script :

- Pour chaque masque binaire, il sépare les particules collées (segmentation d'instance).

- Il sauvegarde une version colorée de cette segmentation dans results/.

- Il calcule les statistiques pour chaque particule dans chaque image.

- Il compile tout dans un fichier results/statistics.csv.

- Il génère des graphiques de distribution (area_distribution.png, etc.) dans results/.

## Améliorations Futures Possibles

Voici quelques pistes pour aller encore plus loin :

- Tracking Temporel : Analyser la séquence vidéo complète pour suivre les particules individuelles au fil du temps, mesurer leur vitesse de croissance, et détecter les événements de fusion.

- Modèles d'Instance Segmentation End-to-End : Remplacer le pipeline U-Net + Watershed par un modèle plus avancé comme Mask R-CNN ou StarDist, qui sont spécifiquement conçus pour la segmentation d'instance et peuvent donner de meilleurs résultats.

- Classification de Formes : Utiliser les métriques de forme (circularité, ellipticité) pour entraîner un petit classifieur (ex: SVM, Random Forest) capable de catégoriser automatiquement les particules (sphérique, bâtonnet, agrégat...).

- Interface Utilisateur : Développer une interface graphique simple (avec Gradio ou Streamlit) pour permettre de charger une image et d'obtenir la segmentation et l'analyse de manière interactive.