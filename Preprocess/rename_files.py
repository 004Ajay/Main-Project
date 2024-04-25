import os
from tqdm import tqdm

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

path = r"D:/Git-Uploads/Main Project/Main-Project/dataset 224"



