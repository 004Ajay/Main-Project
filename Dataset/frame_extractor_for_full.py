import os
import cv2

# Global counter for unique frame names
global_frame_counter = 0

def extract_frames(video_path, output_folder):
    global global_frame_counter
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get frames per second (fps) and total number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through each frame and save it as an image
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame as an image with continuous numbering
        frame_filename = f"{global_frame_counter:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        global_frame_counter += 1
    
    # Release the video capture object
    cap.release()
    
    # Delete the processed video file
    os.remove(video_path)

def process_folder(base_folder):
    # List all subdirectories in the base folder
    subdirectories = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    
    # Iterate through each subdirectory
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(base_folder, subdirectory)
        
        # List all files in the subdirectory
        files = os.listdir(subdirectory_path)
        
        # Iterate through each file in the subdirectory
        for file in files:
            # Check if the file is a video (avi format)
            if file.endswith(".avi"):
                video_path = os.path.join(subdirectory_path, file)
                extract_frames(video_path, subdirectory_path)


# Process the "train_small" folder
train_folder = "C:/Users/ajayt/OneDrive/Desktop/Main P/violenceDetection/train"
process_folder(train_folder)

# Process the "val_small" folder
val_folder = "C:/Users/ajayt/OneDrive/Desktop/Main P/violenceDetection/val"
process_folder(val_folder)
