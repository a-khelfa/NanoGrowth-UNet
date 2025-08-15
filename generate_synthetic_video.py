# generate_synthetic_video.py
# This script generates a synthetic video of moving and evolving nanoparticles with various shapes.

import cv2
import numpy as np
import argparse
from tqdm import tqdm

class Particle:
    """A class to represent a single synthetic particle with various shapes and dynamics."""
    def __init__(self, x, y, max_x, max_y):
        self.x = int(x)
        self.y = int(y)
        self.max_x = max_x
        self.max_y = max_y
        
        # Movement dynamics
        self.vx = np.random.uniform(-1, 1)
        self.vy = np.random.uniform(-1, 1)
        
        # Shape and appearance
        self.shape = np.random.choice(['spherical', 'ellipse', 'nanorod', 'cube', 'triangle', 'icosahedron'])
        self.angle = np.random.randint(0, 180)
        self.rotation_speed = np.random.uniform(-1, 1)
        self.intensity = np.random.randint(150, 255)
        
        # Size and growth dynamics
        self.size = np.random.randint(10, 20)
        self.growth_rate = np.random.uniform(-0.05, 0.1) # Can shrink or grow
        self.min_size = 5
        self.max_size = 30
        
        self.lifetime = np.random.randint(100, 300)

    def draw(self, frame):
        """Draw the particle on the frame based on its shape."""
        if self.shape == 'spherical':
            cv2.circle(frame, (self.x, self.y), int(self.size), self.intensity, -1)
        elif self.shape == 'ellipse':
            axes = (int(self.size * 1.5), int(self.size))
            cv2.ellipse(frame, (self.x, self.y), axes, self.angle, 0, 360, self.intensity, -1)
        elif self.shape == 'nanorod':
            axes = (int(self.size * 2.5), int(self.size / 2))
            cv2.ellipse(frame, (self.x, self.y), axes, self.angle, 0, 360, self.intensity, -1)
        elif self.shape == 'cube':
            half_size = int(self.size / np.sqrt(2))
            rect_points = cv2.boxPoints(((self.x, self.y), (half_size*2, half_size*2), self.angle))
            cv2.fillPoly(frame, [np.int0(rect_points)], self.intensity)
        elif self.shape == 'triangle':
            half_size = int(self.size)
            points = np.array([
                [0, -half_size],
                [-half_size, half_size],
                [half_size, half_size]
            ])
            rot_matrix = cv2.getRotationMatrix2D((0,0), self.angle, 1)
            rotated_points = (rot_matrix[:, :2] @ points.T).T
            translated_points = np.int0(rotated_points + [self.x, self.y])
            cv2.fillPoly(frame, [translated_points], self.intensity)
        elif self.shape == 'icosahedron': # Approximated as a hexagon in 2D
            half_size = int(self.size)
            points = np.array([
                [half_size, 0], [half_size/2, half_size*np.sqrt(3)/2],
                [-half_size/2, half_size*np.sqrt(3)/2], [-half_size, 0],
                [-half_size/2, -half_size*np.sqrt(3)/2], [half_size/2, -half_size*np.sqrt(3)/2]
            ])
            rot_matrix = cv2.getRotationMatrix2D((0,0), self.angle, 1)
            rotated_points = (rot_matrix[:, :2] @ points.T).T
            translated_points = np.int0(rotated_points + [self.x, self.y])
            cv2.fillPoly(frame, [translated_points], self.intensity)

    def update(self):
        """Update particle position, size, and state."""
        self.x += self.vx
        self.y += self.vy
        self.angle += self.rotation_speed
        self.size += self.growth_rate

        # Brownian motion
        self.vx = np.clip(self.vx + np.random.uniform(-0.2, 0.2), -1.5, 1.5)
        self.vy = np.clip(self.vy + np.random.uniform(-0.2, 0.2), -1.5, 1.5)

        # Bounce off walls
        if self.x <= self.size or self.x >= self.max_x - self.size: self.vx *= -1
        if self.y <= self.size or self.y >= self.max_y - self.size: self.vy *= -1
        
        # Reverse growth if size limits are reached
        if self.size >= self.max_size or self.size <= self.min_size:
            self.growth_rate *= -1
            
        self.lifetime -= 1

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic video of nanoparticles.")
    parser.add_argument('--frames', type=int, default=300, help="Number of frames in the video.")
    parser.add_argument('--width', type=int, default=512, help="Width of the video frames.")
    parser.add_argument('--height', type=int, default=512, help="Height of the video frames.")
    parser.add_argument('--max-particles', type=int, default=20, help="Maximum number of particles on screen.")
    parser.add_argument('--output', type=str, default="synthetic_video_shapes.avi", help="Output video file name.")
    args = parser.parse_args()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(args.output, fourcc, 30.0, (args.width, args.height), isColor=False)

    particles = []

    for _ in tqdm(range(args.frames), desc="Generating Video"):
        frame = np.zeros((args.height, args.width), dtype=np.float32)

        if len(particles) < args.max_particles and np.random.rand() > 0.9:
             particles.append(Particle(
                 np.random.randint(50, args.width - 50),
                 np.random.randint(50, args.height - 50),
                 args.width, args.height
             ))
        
        for p in particles:
            p.update()
            p.draw(frame)

        particles = [p for p in particles if p.lifetime > 0]
        
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        noise = np.random.normal(0, 25, frame.shape).astype(np.float32)
        final_frame = np.clip(frame_blurred + noise, 0, 255).astype(np.uint8)
        
        video_writer.write(final_frame)

    video_writer.release()
    print(f"\nSynthetic video with various shapes saved to {args.output}")

if __name__ == "__main__":
    main()
