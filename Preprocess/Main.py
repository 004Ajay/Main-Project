import os
import split_frames_to_train_val,rename_files,save_frames_as_npy_and_delete_folders,process_frames
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import shutil


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




