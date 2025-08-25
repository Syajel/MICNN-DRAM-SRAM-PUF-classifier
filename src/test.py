import torch
from torch.utils.data import DataLoader
from src.load_datasets import DRAMSRAMBinaryDataset
from src.model import MultiInputCNN_SRAM
from src.plot_results import plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

batch_size = 8 
test_dataset_dram = DRAMSRAMBinaryDataset(root_dir="data", subset="test", memory_type="dram")
test_dataset_sram = DRAMSRAMBinaryDataset(root_dir="data", subset="test", memory_type="sram")

test_loader_dram = DataLoader(test_dataset_dram, batch_size=batch_size, shuffle=False)
test_loader_sram = DataLoader(test_dataset_sram, batch_size=batch_size, shuffle=False)

model = MultiInputCNN_SRAM().to(device)
model.load_state_dict(torch.load("models/3metacompact4l_30x8.pth", map_location=device))            # Model to be loaded
model.eval()
print("Model loaded")

def evaluate_dram_and_sram(dram_loader, sram_loader, dram_total, sram_total):
    """
    Evaluates the test dataset and prints accuracy numbers
    DRAM: each of the three response types (intact, noisy, corrupt) are tested separately since a separate copy was made for each
    SRAM: the three types of responses are tested all once since there were enough responses to split rather than copy
    """
    def evaluate(loader, total_files, is_dram):
        correct_clean, correct_corrupt, correct_noise = 0, 0, 0
        total_clean, total_corrupt, total_noise = 0, 0, 0

        corrupt_range = int(total_files * 0.25)          # Corrupt files are the first 25% of files for SRAM
        noise_range = int(total_files * 0.5)            # Noisy files are the the second 25% of files for SRAM

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if is_dram:
                    images, metadata, labels = batch
                    metadata = metadata.to(device)
                else:
                    images, labels = batch
                    metadata = torch.zeros((labels.size(0), 2), dtype=torch.float32, device=device)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, metadata, is_dram)
                predictions = torch.argmax(outputs, dim=1)

                for i in range(len(labels)):
                    file_index = idx * batch_size + i
                    if file_index < corrupt_range:
                        total_corrupt += 1
                        correct_corrupt += (predictions[i] == labels[i]).item()
                    elif file_index < noise_range:
                        total_noise += 1
                        correct_noise += (predictions[i] == labels[i]).item()
                    else:
                        total_clean += 1
                        correct_clean += (predictions[i] == labels[i]).item()

        return correct_clean, correct_corrupt, correct_noise, total_clean, total_corrupt, total_noise

    dram_results = evaluate(dram_loader, dram_total, True)
    sram_results = evaluate(sram_loader, sram_total, False)

    def print_results(name, results):
        correct_clean, correct_corrupt, correct_noise, total_clean, total_corrupt, total_noise = results
        #print(f"{name} Clean Accuracy: {100 * correct_clean / total_clean:.2f}%")          # Only relevant for SRAM
        #print(f"{name} Corrupted Accuracy: {100 * correct_corrupt / total_corrupt:.2f}%")          # Only relevant for SRAM
        #print(f"{name} Noisy Accuracy: {100 * correct_noise / total_noise:.2f}%")          # Only relevant for SRAM
        print(f"{name} Total Accuracy: {100 * (correct_clean + correct_corrupt + correct_noise) / (total_clean + total_corrupt + total_noise):.2f}%")           # Relevant for both DRAM and SRAM

    print_results("DRAM", dram_results)
    print_results("SRAM", sram_results)

evaluate_dram_and_sram(test_loader_dram, test_loader_sram, len(test_dataset_dram), len(test_dataset_sram))
