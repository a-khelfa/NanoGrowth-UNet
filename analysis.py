# analysis.py
# This script performs instance segmentation and shape classification on predicted masks.

import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from tqdm import tqdm
import argparse

def segment_instances(binary_mask):
    distance_map = ndimage.distance_transform_edt(binary_mask)
    coords = peak_local_max(distance_map, min_distance=5, labels=binary_mask)
    local_maxi_mask = np.zeros(distance_map.shape, dtype=bool)
    local_maxi_mask[tuple(coords.T)] = True
    markers = ndimage.label(local_maxi_mask, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance_map, markers, mask=binary_mask)
    return labels

def classify_shape(stats):
    """Classifies shape based on geometric properties."""
    circularity = stats["circularity"]
    aspect_ratio = stats["aspect_ratio"]

    if aspect_ratio > 2.2:
        return "nanorod"
    elif circularity > 0.9:
        return "spherical"
    elif circularity > 0.8 and aspect_ratio < 1.3:
        return "icosahedron/cube" # Hard to distinguish in 2D
    elif circularity > 0.65:
        return "ellipse"
    else:
        return "triangle/other"

def analyze_particles(labels_mask):
    """Computes statistics and classifies shape for each labeled particle."""
    particle_stats = []
    for label in np.unique(labels_mask):
        if label == 0: continue
        
        particle_mask = np.zeros_like(labels_mask, dtype=np.uint8)
        particle_mask[labels_mask == label] = 255
        contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = contours[0]
        
        area = cv2.contourArea(cnt)
        if area < 10: continue # Ignore very small artifacts
        
        perimeter = cv2.arcLength(cnt, True)
        equi_diameter = np.sqrt(4 * area / np.pi)
        radius = equi_diameter / 2
        volume_3d = (4/3) * np.pi * (radius**3)
        
        circularity = 4 * np.pi * (area / (perimeter**2)) if perimeter > 0 else 0
        
        # New metrics for shape classification
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h > 0 else 0

        stats = {
            "label_id": label,
            "area_pixels": area,
            "perimeter_pixels": perimeter,
            "equivalent_diameter_pixels": equi_diameter,
            "estimated_volume_3d": volume_3d,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio
        }
        stats["shape"] = classify_shape(stats)
        particle_stats.append(stats)
        
    return particle_stats

def main():
    parser = argparse.ArgumentParser(description='Analyze segmented nanoparticle masks.')
    parser.add_argument('--input', type=str, required=True, help='Folder containing predicted binary masks.')
    parser.add_argument('--output', type=str, default='results', help='Folder to save analysis results.')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    all_stats = []
    mask_files = [f for f in os.listdir(args.input) if f.endswith(('.png', '.jpg', '.tif'))]

    for mask_file in tqdm(mask_files, desc="Analyzing masks"):
        mask_path = os.path.join(args.input, mask_file)
        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if binary_mask is None: continue
        
        instance_mask = segment_instances(binary_mask)
        stats = analyze_particles(instance_mask)
        
        for stat in stats:
            stat['filename'] = mask_file
            all_stats.append(stat)
            
        vis_path = os.path.join(args.output, f"instance_{mask_file}")
        colored_labels = plt.cm.get_cmap('jet')(instance_mask / (np.max(instance_mask) if np.max(instance_mask) > 0 else 1))
        colored_labels[instance_mask == 0] = [0, 0, 0, 1]
        plt.imsave(vis_path, colored_labels)

    df = pd.DataFrame(all_stats)
    csv_path = os.path.join(args.output, "statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAnalysis complete. Statistics saved to {csv_path}")

    if not df.empty:
        plt.style.use('ggplot')
        
        # Plot shape distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df['shape'], order=df['shape'].value_counts().index)
        plt.title('Distribution des Formes de Nanoparticules')
        plt.xlabel('Nombre de particules')
        plt.ylabel('Forme Classifiée')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'shape_distribution.png'))
        plt.close()
        
        print(f"Plots saved in {args.output}")

if __name__ == "__main__":
    main()
