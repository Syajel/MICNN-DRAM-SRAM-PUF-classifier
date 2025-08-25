import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np

def process_and_save_binary(file_path, dest_path, target_size=(512, 256)):
    """
    DRAM helper function to remove the headers and compress the files using bilinear interpolation
    """
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    if len(data) <= 9:
        print(f"{file_path} too small to process")
        return
    
    header = data[:9]
    data = data[9:]  # Remove 9 byte header

    # Compute original image shape before resizing
    original_shape = (4096, 2048) 
    expected_size = original_shape[0] * original_shape[1]

    # Ensure correct size before resizing
    if data.size < expected_size:
        data = np.pad(data, (0, expected_size - data.size), mode='constant')
    elif data.size > expected_size:
        data = data[:expected_size]

    # Reshape to original (4096, 2048)
    data = data.reshape(original_shape).astype(np.float32) / 255.0
    tensor_data = torch.tensor(data).unsqueeze(0).unsqueeze(0)  

    # Resize using Bilinear Interpolation to (512, 256)
    resized_tensor = F.interpolate(tensor_data, size=target_size, mode='bilinear', align_corners=False).squeeze()

    # Convert back to uint8 & save as binary
    resized_data = (resized_tensor.numpy() * 255).astype(np.uint8)
    with open(dest_path, 'wb') as f:
        f.write(resized_data.tobytes())

    print(f"Compressed: {dest_path}")



def split_dataset(raw_dir, train_dir, test_dir, test_files_count=2, is_sram=False):
    """
    Splits SRAM and DRAM data from `raw/` into `train/` and `test/`, while preserving folder structure.
    """
    if not os.path.exists(raw_dir):
        print(f"{raw_dir} does not exist.")
        return

    for category in os.listdir(raw_dir):  # DRAM sets (aa, em, fb) or SRAM boards (boardXXXX)
        category_path = os.path.join(raw_dir, category)
        if not os.path.isdir(category_path):
            continue

        # SRAM: Last 20 files in each board folder go to test
        if is_sram:
            all_files = sorted(os.listdir(category_path))  # Sort to maintain order
            all_files = [f for f in all_files if f.startswith("board")]  # Ensure valid files
            
            test_files = all_files[-test_files_count:]  # Last 20 files go to test
            train_files = all_files[:-test_files_count]  # Remaining files go to train

            for file_name in train_files:
                file_path = os.path.join(category_path, file_name)
                dest_path = os.path.join(train_dir, category, file_name)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(file_path, dest_path)

            for file_name in test_files:
                file_path = os.path.join(category_path, file_name)
                dest_path = os.path.join(test_dir, category, file_name)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(file_path, dest_path)

        # DRAM: Last 2 runs in each time interval go to test
        else:
            for addr_range in os.listdir(category_path):  
                addr_path = os.path.join(category_path, addr_range)
                if not os.path.isdir(addr_path):
                    continue

                for time_int in os.listdir(addr_path):  
                    time_path = os.path.join(addr_path, time_int)
                    if not os.path.isdir(time_path):
                        continue

                    all_files = sorted(os.listdir(time_path))  # Sort runs in order

                    test_files = all_files[-test_files_count:]  # Last 2 runs go to test
                    train_files = all_files[:-test_files_count]  # All other runs go to train

                    for file_name in train_files:
                        file_path = os.path.join(time_path, file_name)
                        dest_path = os.path.join(train_dir, category, addr_range, time_int, file_name)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        process_and_save_binary(file_path, dest_path)

                    for file_name in test_files:
                        file_path = os.path.join(time_path, file_name)
                        dest_path = os.path.join(test_dir, category, addr_range, time_int, file_name)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        process_and_save_binary(file_path, dest_path)
                        
                    print(f"Split {category}: Done")

# Define dataset paths

data_root = "data"

dram_raw = os.path.join(data_root, "dram", "raw")
dram_train = os.path.join(data_root, "dram", "train")
dram_test = os.path.join(data_root, "dram", "test")

sram_raw = os.path.join(data_root, "sram", "raw")
sram_train = os.path.join(data_root, "sram", "train")
sram_test = os.path.join(data_root, "sram", "test")

split_dataset(dram_raw, dram_train, dram_test, test_files_count=2, is_sram=False)
split_dataset(sram_raw, sram_train, sram_test, test_files_count=20, is_sram=True)

print("Done")
