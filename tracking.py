# tracking.py
# This script performs particle tracking and shape classification on a video.

import cv2
import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance
import argparse
from tqdm import tqdm
import os

from models.unet import UNet
from analysis import segment_instances, analyze_particles
from predict import predict_single_image

class ParticleTracker:
    # ... (le contenu de la classe ParticleTracker reste identique)
    def __init__(self, max_disappeared=10, max_distance=30):
        self.next_object_id = 0
        self.objects = {}
        self.object_info = {} # NEW: Store shape and other info
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.tracks = []

    def register(self, centroid, stats):
        self.objects[self.next_object_id] = centroid
        self.object_info[self.next_object_id] = {'shape': stats.get('shape', 'unknown')}
        self.disappeared[self.next_object_id] = 0
        self.tracks.append([self.next_object_id] + list(centroid) + list(stats.values()))
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.object_info[object_id]
        del self.disappeared[object_id]

    def update(self, detected_particles):
        if len(detected_particles) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(detected_particles), 2), dtype="int")
        for i, p in enumerate(detected_particles):
            mask = np.zeros((256, 256), dtype=np.uint8) # Assuming size, should be dynamic
            mask[p['labels_mask'] == p['label_id']] = 255
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], detected_particles[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = distance.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols or D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.object_info[object_id]['shape'] = detected_particles[col].get('shape', 'unknown')
                self.disappeared[object_id] = 0
                self.tracks.append([object_id] + list(input_centroids[col]) + list(detected_particles[col].values()))
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            for col in unused_cols:
                self.register(input_centroids[col], detected_particles[col])

        return self.objects

def main():
    parser = argparse.ArgumentParser(description="Track nanoparticles in a video.")
    # ... (les arguments restent les mêmes)
    parser.add_argument('--video', required=True, help="Path to input video.")
    parser.add_argument('--model', required=True, help="Path to trained U-Net model.")
    parser.add_argument('--output-video', default="tracked_video.avi", help="Path for output video.")
    parser.add_argument('--output-csv', default="tracking_data.csv", help="Path for output CSV data.")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))
    model.eval()

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    tracker = ParticleTracker()
    
    for frame_idx in tqdm(range(total_frames), desc="Tracking Particles"):
        ret, frame = cap.read()
        if not ret: break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        temp_frame_path = "temp_frame.png"
        cv2.imwrite(temp_frame_path, gray_frame)
        binary_mask_pil = predict_single_image(model, temp_frame_path, DEVICE, height, width)
        binary_mask = np.array(binary_mask_pil)
        
        instance_mask = segment_instances(binary_mask)
        
        particles_stats = analyze_particles(instance_mask)
        for p in particles_stats:
            p['frame_id'] = frame_idx
            p['labels_mask'] = instance_mask

        tracked_objects = tracker.update(particles_stats)
        
        output_frame = frame.copy()
        for object_id, centroid in tracked_objects.items():
            shape = tracker.object_info.get(object_id, {}).get('shape', '')
            text = f"ID {object_id} ({shape})"
            cv2.putText(output_frame, text, (centroid[0] - 20, centroid[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.circle(output_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
        out_video.write(output_frame)
    
    if os.path.exists("temp_frame.png"): os.remove("temp_frame.png")
    cap.release()
    out_video.release()

    # Mettre à jour les colonnes pour le CSV
    df_columns = ['track_id', 'x', 'y', 'label_id', 'area_pixels', 'perimeter_pixels', 
                  'equivalent_diameter_pixels', 'estimated_volume_3d', 'circularity', 
                  'aspect_ratio', 'shape', 'frame_id', 'labels_mask']
    df = pd.DataFrame(tracker.tracks, columns=df_columns)
    df = df.drop(columns=['labels_mask'])
    df.to_csv(args.output_csv, index=False)
    
    print(f"\nTracking complete. Video saved to {args.output_video}, data saved to {args.output_csv}")

if __name__ == "__main__":
    main()
