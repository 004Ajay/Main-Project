
import os
import cv2
from glob import glob
from tqdm import tqdm

def process_frames(base_dir, frame_size, output_format='jpeg'):
    
    frame_folders = ['train/Fight', 'train/NonFight', 'val/Fight', 'val/NonFight']
    
    for folder in frame_folders:
        frame_files = glob(os.path.join(base_dir, folder, '*'))  # Assuming frame images are in various formats
        print(f"Processing folder: {folder}")  # Optional: print current folder being processed
        for frame_path in tqdm(frame_files, desc=f"Processing Frames in {folder}"):
            frame = cv2.imread(frame_path)
            resized_frame = cv2.resize(frame, (frame_size, frame_size))
            
            # Save processed frames
            frame_basename = os.path.basename(frame_path).split('.')[0]
            output_frame_path = os.path.join(base_dir, folder, f"{frame_basename}_processed.{output_format}")
            cv2.imwrite(output_frame_path, resized_frame)



frame_size = 224


