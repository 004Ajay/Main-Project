import os
import numpy as np
from PIL import Image
import shutil  # for directory removal

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
dataset_root = "D:\Git-Uploads\Main Project\Main-Project\dataset 224"
