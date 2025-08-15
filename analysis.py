# analysis.py
# This script performs instance segmentation on predicted masks and computes statistics.

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
    """
    Separates touching objects in a binary mask using the Watershed algorithm.
    
    Args:
        binary_mask (np.array): A binary mask where foreground is 255.
        
    Returns:
        np.array: A label mask where each separated object has a unique integer ID.
    """
    # Compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel
    distance_map = ndimage.distance_transform_edt(binary_mask)
    
    # Find local maxima (peaks) in the distance map. These will be our markers.
    # The function now returns coordinates, not a boolean mask.
    # The 'labels' argument ensures peaks are only found within the mask area.
    coords = peak_local_max(distance_map, min_distance=5, labels=binary_mask)
    
    # Create an empty mask for the markers and draw them from the coordinates
    local_maxi_mask = np.zeros(distance_map.shape, dtype=bool)
    local_maxi_mask[tuple(coords.T)] = True
    
    # Perform a connected component analysis on the local maxima,
    # using 8-connectivity, and find the markers
    markers = ndimage.label(local_maxi_mask, structure=np.ones((3, 3)))[0]
    
    # Apply the Watershed algorithm
    labels = watershed(-distance_map, markers, mask=binary_mask)
    
    return labels

def analyze_particles(labels_mask):
    """
    Computes statistics for each labeled particle in the mask.
    
    Args:
        labels_mask (np.array): Mask where each particle has a unique ID.
        
    Returns:
        list: A list of dictionaries, where each dictionary contains
              stats for one particle.
    """
    particle_stats = []
    num_particles = len(np.unique(labels_mask)) - 1 # Exclude background
    
    if num_particles == 0:
        return []

    for label in np.unique(labels_mask):
        if label == 0:  # Skip background
            continue
            
        # Create a mask for the current particle
        particle_mask = np.zeros_like(labels_mask, dtype=np.uint8)
        particle_mask[labels_mask == label] = 255
        
        # Find contours
        contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
            
        cnt = contours[0]
        
        # Calculate properties
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Equivalent diameter: diameter of a circle with the same area
        equi_diameter = np.sqrt(4 * area / np.pi)
        
        # 3D Volume estimation (assuming spherical particle)
        radius = equi_diameter / 2
        volume_3d = (4/3) * np.pi * (radius**3)
        
        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
        else:
            circularity = 0
            
        particle_stats.append({
            "label_id": label,
            "area_pixels": area,
            "perimeter_pixels": perimeter,
            "equivalent_diameter_pixels": equi_diameter,
            "estimated_volume_3d": volume_3d,
            "circularity": circularity
        })
        
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
        
        if binary_mask is None:
            continue
        
        # Separate touching particles
        instance_mask = segment_instances(binary_mask)
        
        # Analyze particles in the instance mask
        stats = analyze_particles(instance_mask)
        
        for stat in stats:
            stat['filename'] = mask_file # Add filename for reference
            all_stats.append(stat)
            
        # Save a visualization of the instance segmentation
        # Generate a color for each label
        vis_path = os.path.join(args.output, f"instance_{mask_file}")
        # Using a colormap for visualization
        colored_labels = plt.cm.get_cmap('jet')(instance_mask / (np.max(instance_mask) if np.max(instance_mask) > 0 else 1))
        colored_labels[instance_mask == 0] = [0, 0, 0, 1] # Black background
        plt.imsave(vis_path, colored_labels)

    # Convert stats to a pandas DataFrame
    df = pd.DataFrame(all_stats)
    
    # Save statistics to CSV
    csv_path = os.path.join(args.output, "statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAnalysis complete. Statistics saved to {csv_path}")

    # --- Generate and save plots ---
    if not df.empty:
        plt.style.use('ggplot')
        
        # Plot distribution of particle areas
        plt.figure(figsize=(10, 6))
        sns.histplot(df['area_pixels'], kde=True, bins=30)
        plt.title('Distribution de l\'Aire des Nanoparticules')
        plt.xlabel('Aire (en pixels²)')
        plt.ylabel('Nombre de particules')
        plt.savefig(os.path.join(args.output, 'area_distribution.png'))
        plt.close()

        # Plot distribution of equivalent diameters
        plt.figure(figsize=(10, 6))
        sns.histplot(df['equivalent_diameter_pixels'], kde=True, bins=30)
        plt.title('Distribution du Diamètre Équivalent des Nanoparticules')
        plt.xlabel('Diamètre (en pixels)')
        plt.ylabel('Nombre de particules')
        plt.savefig(os.path.join(args.output, 'diameter_distribution.png'))
        plt.close()
        
        print(f"Plots saved in {args.output}")

if __name__ == "__main__":
    main()
