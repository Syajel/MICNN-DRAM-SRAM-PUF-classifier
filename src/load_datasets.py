import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# DRAM & SRAM Mapping
MEMORY_TYPES = {
    # DRAM board mapping
    "aa": 0, "em": 1, "fb": 2,

    # SRAM board mapping (Hexadecimal 000A-000F)
    "board000a": 3, "board000b": 4, "board000c": 5, "board000d": 6, "board000e": 7, "board000f": 8,

    # SRAM board mapping (Decimal 002-098) excluding removed boards (36, 38, 44, 92, 95, 63)
    "board0002": 9, "board0003": 10, "board0004": 11, "board0005": 12, "board0006": 13, "board0007": 14, "board0008": 15,
    "board0009": 16, "board0010": 17, "board0011": 18, "board0012": 19, "board0013": 20, "board0014": 21, "board0015": 22,
    "board0016": 23, "board0017": 24, "board0018": 25, "board0019": 26, "board0020": 27, "board0021": 28, "board0022": 29,
    "board0023": 30, "board0024": 31, "board0025": 32, "board0026": 33, "board0027": 34, "board0028": 35, "board0029": 36,
    "board0030": 37, "board0031": 38, "board0032": 39, "board0033": 40, "board0034": 41, "board0035": 42, "board0037": 43,
    "board0039": 44, "board0040": 45, "board0041": 46, "board0042": 47, "board0043": 48, "board0045": 49, "board0046": 50,
    "board0047": 51, "board0048": 52, "board0049": 53, "board0050": 54, "board0051": 55, "board0052": 56, "board0053": 57,
    "board0054": 58, "board0055": 59, "board0056": 60, "board0057": 61, "board0058": 62, "board0059": 63, "board0060": 64,
    "board0061": 65, "board0062": 66, "board0064": 67, "board0065": 68, "board0066": 69, "board0067": 70, "board0068": 71,
    "board0069": 72, "board0070": 73, "board0071": 74, "board0072": 75, "board0073": 76, "board0074": 77, "board0075": 78,
    "board0076": 79, "board0077": 80, "board0078": 81, "board0079": 82, "board0080": 83, "board0081": 84, "board0082": 85,
    "board0083": 86, "board0084": 87, "board0085": 88, "board0086": 89, "board0087": 90, "board0088": 91, "board0089": 92,
    "board0090": 93, "board0091": 94, "board0093": 95, "board0094": 96, "board0096": 97, "board0097": 98, "board0098": 99,

    # SRAM board mapping (Hexadecimal 001A-008F) excluding removed board 5C and 1C
    "board001a": 100, "board001b": 101, "board001d": 102, "board001e": 103, "board001f": 104,
    "board002a": 105, "board002b": 106, "board002c": 107, "board002d": 108, "board002e": 109, "board002f": 110,
    "board003a": 111, "board003b": 112, "board003c": 113, "board003d": 114, "board003e": 115, "board003f": 116,
    "board004a": 117, "board004b": 118, "board004c": 119, "board004d": 120, "board004e": 121, "board004f": 122,
    "board005a": 123, "board005b": 124, "board005d": 125, "board005e": 126, "board005f": 127,
    "board006a": 128, "board006b": 129, "board006c": 130, "board006d": 131, "board006e": 132, "board006f": 133,
    "board007a": 134, "board007b": 135, "board007c": 136, "board007d": 137, "board007e": 138, "board007f": 139,
    "board008a": 140, "board008b": 141, "board008c": 142, "board008d": 143, "board008e": 144, "board008f": 145
}



# DRAM address range mapping
ADDRESS_RANGES = {
    "c3c38": 0, "c4c48": 1, "c5c58": 2, "c6c68": 3, "c38": 4,
    "c4": 5, "c48c5": 6, "c58c6": 7, "c68c7": 8
}

# DRAM time interval mapping
TIME_INTERVALS = {
    "1min": 0, "2min": 1, "3min": 2, "4min": 3, "5min": 4, "6min": 5, "7min": 6, "8min": 7, "9min": 8, "10min": 9,
    "15min": 10, "20min": 11, "25min": 12, "30min": 13, "40min": 14, "50min": 15, "60min": 16, "10s": 17, "30s": 18
}

class DRAMSRAMBinaryDataset(Dataset):
    def __init__(self, root_dir, subset="train", memory_type="all",
                 dram_subdir="dram", sram_subdir="sram",
                 dram_image_size=(512, 256), sram_image_size=(512, 320), header_size=9):
        """
        Dataset class to load either DRAM or SRAM binary dumps.
        
        memory_type: "dram", "sram", or "all"
        """
        self.dram_dir = os.path.join(root_dir, dram_subdir, subset)  
        self.sram_dir = os.path.join(root_dir, sram_subdir, subset)  
        self.dram_image_size = dram_image_size
        self.sram_image_size = sram_image_size
        self.header_size = header_size
        self.files = []
        self.labels = []
        self.metadata = []

        if memory_type in ["dram", "all"]:
            self._load_dram_data(self.dram_dir)
        if memory_type in ["sram", "all"]:
            self._load_sram_data(self.sram_dir)

        print(f"Loaded {len(self.files)} {memory_type.upper()} files from {subset} dataset.")

    def _load_dram_data(self, dram_root):
        """Load DRAM dataset with metadata"""
        if not os.path.exists(dram_root):
            print(f"WARNING: {dram_root} does not exist. Skipping.")
            return
        for dram_set in os.listdir(dram_root):  
            dram_path = os.path.join(dram_root, dram_set)
            if not os.path.isdir(dram_path):
                continue
            for dram_device in os.listdir(dram_path):  
                device_path = os.path.join(dram_path, dram_device)
                if not os.path.isdir(device_path):
                    continue
                for addr_range in os.listdir(device_path):
                    addr_path = os.path.join(device_path, addr_range)
                    if not os.path.isdir(addr_path): continue

                    for time_int in os.listdir(addr_path):
                        time_path = os.path.join(addr_path, time_int)
                        if not os.path.isdir(time_path): continue

                        for file in os.listdir(time_path):
                            if not file.endswith(".bin"):  
                                continue  # Ensure only binary files are loaded

                            file_path = os.path.join(time_path, file)
                            self.files.append(file_path)

                            # Assign DRAM class label
                            dram_label = MEMORY_TYPES.get(dram_device.lower(), -1)
                            self.labels.append(dram_label)

                            # Store metadata (address range, time interval)
                            addr_num = ADDRESS_RANGES.get(addr_range, -1)
                            time_num = TIME_INTERVALS.get(time_int, -1)
                            self.metadata.append((addr_num, time_num))

    def _load_sram_data(self, sram_root):
        """Load SRAM dataset (no metadata)"""
        if not os.path.exists(sram_root):
            print(f"WARNING: {sram_root} does not exist. Skipping.")
            return

        for board_folder in os.listdir(sram_root):
            board_path = os.path.join(sram_root, board_folder)
            if not os.path.isdir(board_path):
                continue

            for file in os.listdir(board_path):
                if not file.startswith("board"):  # Ignore non-SRAM files
                    continue

                file_path = os.path.join(board_path, file)
                self.files.append(file_path)

                # Extract SRAM board name
                board_name = file[:9]  # Example: "board000A"
                sram_label = MEMORY_TYPES.get(board_name.lower(), -1)
                self.labels.append(sram_label)

                # SRAM has no metadata
                self.metadata.append((0, 0))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Load and preprocess a binary file into a grayscale image"""
        file_path = self.files[idx]
        label = self.labels[idx]

        # Determine if it's DRAM or SRAM
        memory_type = file_path.split(os.sep)[-4]  # "dram" or "sram"
        is_sram = memory_type == "sram"

        # Select correct image size
        image_size = self.sram_image_size if is_sram else self.dram_image_size

        # Read binary file
        with open(file_path, "rb") as f:
            #if not is_sram:
            #    f.seek(self.header_size)  # ✅ Skip header ONLY for DRAM
            image_data = np.frombuffer(f.read(), dtype=np.uint8)

        # Normalize and reshape
        image_data = image_data.reshape(image_size).astype(np.float32) / 255.0
        image_tensor = torch.tensor(image_data).unsqueeze(0)  # (1, Height, Width) for grayscale

        if is_sram:
            return image_tensor, torch.tensor(label, dtype=torch.long)
        else:
            metadata_tensor = torch.tensor(self.metadata[idx], dtype=torch.float32)
            return image_tensor, metadata_tensor, torch.tensor(label, dtype=torch.long)

