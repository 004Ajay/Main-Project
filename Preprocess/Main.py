import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import shutil
from glob import glob

########################################################################
#                              MAIN FUCTION                
########################################################################

def main():
    parser = argparse.ArgumentParser(description='Process frame images and optionally save as npy files.')
    parser.add_argument('--base_dir', type=str, default="D:/Git-Uploads/Main Project/Main-Project/Dataset renamed", help='Base directory')
    parser.add_argument('--rename_files', action='store_true', help='Whether to rename files based on their category') #argument is present in the command line, its value will be set to True. else False.
    parser.add_argument('--split_to_frames', action='store_true', help='Whether to split videos into frames')
    parser.add_argument('--frame_size', type=int, default=224, help='Size to  resize')
    parser.add_argument('--process', action='store_true', help='Whether to process and resize frames')
    parser.add_argument('--save_as_npy', action='store_true', help='Whether to save frames as npy files and delete original folders')
    parser.add_argument('--dataset_root',type=str, default="D:/Git-Uploads/Main Project/Main-Project/dataset 224", help='npy storage loc')
    args = parser.parse_args()

    if args.rename_files:
        if  args.base_dir:
            rename_files(args.base_dir)
        else:
            print("Error: Please specify --base_dir arguments before using --rename_files.")
            return

    if args.split_to_frames:
        if args.base_dir:
            split_frames_to_train_val(args.base_dir, args.base_dir)
        else:
            print("Error: Please specify --base_dir  argument before using --split_to_frames.")
            return

    if args.save_as_npy:
        if args.base_dir:
            save_frames_as_npy_and_delete_folders(args.base_dir)
        else:
            print("Error: Please specify --base_dir argument before using --save_as_npy.")
            return

    if args.process:
        if args.base_dir:
            process_frames(args.base_dir, args.frame_size)
        else:
            print("Error: Please specify --base_dir argument.")
            return

########################################################################
#                            RENAME FILES               
########################################################################


def rename_files(path):
    entries = os.listdir(path)

    for entry in entries:
        entry_path = os.path.join(path, entry)
        
        if os.path.isdir(entry_path):
            sub_dirs = os.listdir(entry_path)

            for sub_dir in sub_dirs:
                sub_path = os.path.join(entry_path, sub_dir)

                if os.path.isdir(sub_path):
                    files = os.listdir(sub_path)

                    # Initialize the counter before the loop
                    i = 1

                    # Wrap the file iteration with tqdm for progress bar
                    for file_name in tqdm(files, desc=f"Renaming files in {sub_path}"):
                        file_path = os.path.join(sub_path, file_name)

                        # Constants
                        train = "train"
                        val = "val"
                        fight_keywords = ["Fight", "fight"]
                        no_fight_keywords = ["No_Fight", "no_fight"]

                        prefix = ""
                        if train in sub_path and any(keyword in sub_path for keyword in no_fight_keywords):
                            prefix = "No_Fight_"
                        elif train in sub_path and any(keyword in sub_path for keyword in fight_keywords):
                            prefix = "Fight_"
                        elif val in sub_path and any(keyword in sub_path for keyword in no_fight_keywords):
                            prefix = "no_fight_"
                        elif val in sub_path and any(keyword in sub_path for keyword in fight_keywords):
                            prefix = "fight_"
                        
                        if prefix:
                            new_file_name = f"{prefix}{i}.avi"
                            new_file_path = os.path.join(sub_path, new_file_name)
                            if not os.path.exists(new_file_path):
                                os.rename(file_path, new_file_path)
                            i += 1

path = r"D:/Git-Uploads/Main Project/Main-Project/dataset 224"  # path to your video dataset to be renamed

########################################################################
#                           SPLIT INTO FRAMES               
########################################################################


def split_frames_to_train_val(folder_path, output_folder):
    # Create train and val folders
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    categories = ["NonFight","Fight","nonfight","fight"]  # List of categories

    for split in ["train", "val"]:
        for category in categories:
            input_folder = os.path.join(folder_path, split, category)
            output_subfolder = os.path.join(output_folder, split, category.lower())

            os.makedirs(output_subfolder, exist_ok=True)

            videos = os.listdir(input_folder)

            for video in videos:
                video_path = os.path.join(input_folder, video)
                video_name = os.path.splitext(video)[0]
                video_capture = cv2.VideoCapture(video_path)
                frame_count = 1
                success = True

                while success:
                    success, image = video_capture.read()
                    if success:
                        output_video_folder = os.path.join(output_subfolder, video_name)
                        os.makedirs(output_video_folder, exist_ok=True)
                        output_path = os.path.join(output_video_folder, f"{video_name}_{frame_count}.jpg")
                        cv2.imwrite(output_path, image)
                        frame_count += 1

                video_capture.release()

                print(f"{video_name} Done\n")

video_folder_path = r"D:/Git-Uploads/Main Project/Main-Project/Violence data/violenceDetection"  # path to folder with video
output_folder = r"D:/Git-Uploads/Main Project/Main-Project/ Dataset Resized"  # path to store extracted frame pictures

########################################################################
 #                          RESIZE FRAMES             
########################################################################

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



#frame_size = 224


########################################################################
#                    CONVERT TO .npy AND SAVE           
########################################################################

def save_frames_as_npy_and_delete_folders(root_dir):
    # Iterate over 'train' and 'val' directories
    for data_type in ['train', 'val']:
        path = os.path.join(root_dir, data_type)
        if not os.path.exists(path):
            continue

        # Iterate over categories like 'Fight', 'No_Fight'
        for category in os.listdir(path):
            category_path = os.path.join(path, category)
            if not os.path.isdir(category_path):
                continue

            # Iterate over each event folder like 'Fight_0'
            for event_folder in os.listdir(category_path):
                event_path = os.path.join(category_path, event_folder)
                if not os.path.isdir(event_path):
                    continue

                # Read each image, sort to maintain the sequence
                images = []
                file_names = sorted(os.listdir(event_path), key=lambda x: int(x.split('.')[0].split('_')[-1]))
                for file_name in file_names:
                    file_path = os.path.join(event_path, file_name)
                    with Image.open(file_path) as img:
                        images.append(np.array(img))

                # Convert list of images to numpy array
                images_array = np.stack(images, axis=0)

                # Save to .npy file in the category directory, not in the event folder
                npy_file_path = os.path.join(category_path, f'{event_folder}.npy')
                np.save(npy_file_path, images_array)
                print(f'Saved {npy_file_path} with shape {images_array.shape}')

                # Remove the event folder after saving .npy file
                shutil.rmtree(event_path)
                print(f'Deleted folder {event_path}')

# Set the root directory of your dataset
# dataset_root = "D:\Git-Uploads\Main Project\Main-Project\dataset 224"