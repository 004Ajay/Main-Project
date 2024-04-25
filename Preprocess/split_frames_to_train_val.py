# The videos read here are actually 5s and 150 frames each.

import os
import cv2

def split_frames_to_train_val(folder_path, output_folder):
    # Create train and val folders
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    categories = ["NonFight"]  # List of categories

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


