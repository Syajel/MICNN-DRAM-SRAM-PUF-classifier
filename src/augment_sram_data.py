import os
import numpy as np

def corrupt_and_add_noise_sram(data_dir,i,j):
    """
    Corrupts responses 0 - i, adds noise to responses i - j
    """
    for root, _, files in os.walk(data_dir):
        files = sorted([f for f in files if f.startswith("board")])  # Only process board files

        # Corrupts 0 - i responses
        for file_name in files[:i]:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'rb') as f:
                data = f.read()
            corrupt_point = int(len(data) * 0.8)  # Sets last 20% to zeros
            corrupted_data = data[:corrupt_point] + b'\x00' * (len(data) - corrupt_point)
            with open(file_path, 'wb') as f:
                f.write(corrupted_data)

        # Adds Gaussian noise to responses i - j
        for file_name in files[i:j]:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            noise = np.random.normal(0, 0.05 ** 0.5, data.shape)    # Gaussian noise
            noisy_data = np.clip(data + noise * 255, 0, 255).astype(np.uint8)   # Adds the noise to the data while ensuring a max value of 255
            with open(file_path, 'wb') as f:
                f.write(noisy_data.tobytes())


# SRAM dataset paths
sram_dirs_train = ["data/sram/train"]
sram_dirs_test = ["data/sram/test"]

# Apply corruption and noise to SRAM dataset
for sram_dir in sram_dirs_train:
    corrupt_and_add_noise_sram(sram_dir,20,40)

for sram_dir in sram_dirs_test:
    corrupt_and_add_noise_sram(sram_dir,5,10)

print("Done")
