# NanoGrowth-UNet : Segmentation, Suivi et Analyse de Formes pour la Microscopie
NanoGrowth-UNet est une boîte à outils complète pour l'analyse de vidéos de microscopie, spécialisée dans la croissance de nanoparticules. Ce projet utilise un réseau de neurones U-Net pour la segmentation, des algorithmes de traitement d'images pour l'analyse quantitative, un classificateur de formes et un suivi (tracking) d'objets pour analyser la dynamique des particules au fil du temps.

Le workflow complet permet de passer d'une vidéo brute à une analyse détaillée des trajectoires, de la croissance et de la morphologie de chaque particule.

## Fonctionnalités
- Segmentation Sémantique : Utilise une architecture U-Net robuste pour générer des masques binaires.

- Segmentation d'Instance : Implémente l'algorithme Watershed pour séparer avec précision les particules qui se touchent.

- Analyse Quantitative et Classification de Formes : Un script d'analyse complet (analysis.py) pour extraire des métriques et classifier la forme de chaque particule détectée (sphérique, nanorod, cube, etc.).

- Suivi Temporel (Tracking) : Un script tracking.py pour suivre les particules à travers les images d'une vidéo, leur assigner un ID unique et enregistrer leurs trajectoires et l'évolution de leurs propriétés.

- Générateurs de Données Synthétiques Avancés :

 - generate_synthetic_data.py : Crée des images fixes avec des particules qui se chevauchent.

 - generate_synthetic_video.py : Crée des vidéos réalistes avec des particules de formes variées qui se déplacent, tournent, croissent et décroissent.

- Rapports et Visualisations : Génère des fichiers .csv détaillés et des graphiques de distribution (taille, forme...).

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
├── generate_synthetic_video.py 
├── train.py                  # Script pour entraîner le modèle U-Net
├── predict.py                # Script pour la segmentation binaire
├── analysis.py               # Script pour l'analyse quantitative
├── tracking.py
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
python analysis.py --input results/ --output analysis_results/
'''

Ce que fait ce script :

- Pour chaque masque binaire, il sépare les particules collées (segmentation d'instance).

- Il sauvegarde une version colorée de cette segmentation dans results/.

- Il calcule les statistiques pour chaque particule dans chaque image.

- Il compile tout dans un fichier results/statistics.csv.

- Il génère des graphiques de distribution (area_distribution.png, etc.) dans results/.

5. Lancement du Suivi sur Vidéo
- Générez une vidéo synthétique pour le test :
'''
python generate_synthetic_video.py --frames 300 --output test_video.avi
'''

- Lancez le script de suivi en utilisant le modèle entraîné et la vidéo générée :
'''
python tracking.py --video test_video.avi --model saved_models/best_model.pth --output-video tracked_results.avi --output-csv tracking_results.csv
'''

Ce que fait ce script :

- Il lit la vidéo image par image.

- Pour chaque image, il prédit un masque, le segmente en instances, et analyse et classifie la forme de chaque particule.

- Il relie les particules entre les images pour créer des trajectoires.

- Il sauvegarde une nouvelle vidéo tracked_results.avi où chaque particule est annotée avec son ID et sa forme classifiée.

- Il compile toutes les données de suivi (ID, position, taille, forme, etc. pour chaque image) dans un fichier tracking_results.csv.

6. Analyse des Données de Suivi
Le fichier tracking_results.csv est une mine d'informations. Vous pouvez l'utiliser (par exemple dans un notebook Jupyter avec Pandas) pour étudier :

- La vitesse de croissance moyenne par type de forme.

- Les trajectoires et le type de mouvement (ex: calcul du déplacement quadratique moyen).

- L'évolution de la distribution des formes au cours du temps.

## Améliorations Futures Possibles

Voici quelques pistes pour aller encore plus loin :

- Algorithme de Tracking plus Robuste : Remplacer le suivi par centroïde simple par des méthodes plus avancées comme le filtre de Kalman pour mieux prédire les positions et gérer les occultations.

- Analyse de Trajectoire Avancée : Calculer le déplacement quadratique moyen (MSD) à partir des trajectoires pour caractériser le type de mouvement (ex: Brownien, dirigé).

- Optimisation des Performances : Le traitement vidéo peut être lent. Optimiser le pipeline (ex: traitement par lots, parallélisation) pour accélérer l'analyse de vidéos longues.

- Modèles d'Instance Segmentation End-to-End : Remplacer le pipeline U-Net + Watershed par un modèle plus avancé comme Mask R-CNN ou StarDist, qui sont spécifiquement conçus pour la segmentation d'instance et peuvent donner de meilleurs résultats.

- Analyse de Fusion/Fragmentation : Développer une logique dans le script de suivi pour détecter explicitement les événements où des particules fusionnent ou se brisent.

- Interface Utilisateur : Développer une interface graphique simple (avec Gradio ou Streamlit) pour permettre de charger une image et d'obtenir la segmentation et l'analyse de manière interactive.