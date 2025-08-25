import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.load_datasets import DRAMSRAMBinaryDataset
from src.model import MultiInputCNN_SRAM
from src.plot_results import plot

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

batch_size = 8          # Batch size

# Load datasets
train_dataset_dram = DRAMSRAMBinaryDataset(root_dir="data", subset="train", memory_type="dram")
train_dataset_sram = DRAMSRAMBinaryDataset(root_dir="data", subset="train", memory_type="sram")

# Separate DRAM & SRAM Loaders
train_loader_dram = DataLoader(train_dataset_dram, batch_size=batch_size, shuffle=True)
train_loader_sram = DataLoader(train_dataset_sram, batch_size=batch_size, shuffle=True)

# Initialize model
model = MultiInputCNN_SRAM().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(loader, is_dram=True):
    """
    Training function
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if is_dram:
            images, metadata, labels = batch
            metadata = metadata.to(device)
        else:
            images, labels = batch
            metadata = torch.zeros((labels.size(0), 2), dtype=torch.float32, device=device)
        
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images, metadata, is_dram)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

# Arrays for accuracy and loss numbers
dram_acc_res=[]
dram_loss_res=[]
sram_acc_res=[]
sram_loss_res=[]



# Main Training Loop
epochs = 30
for epoch in range(epochs):
    # Train DRAM
    dram_loss, dram_acc = train(train_loader_dram, is_dram=True)
    # Train SRAM
    sram_loss, sram_acc = train(train_loader_sram, is_dram=False)

    # Add accuracies and losses to respective arrays
    sram_acc_res.append(sram_acc)
    sram_loss_res.append(sram_loss)
    dram_acc_res.append(dram_acc)
    dram_loss_res.append(dram_loss)

    print(f"Epoch [{epoch+1}/{epochs}] - DRAM Loss: {dram_loss:.4f}, DRAM Acc: {dram_acc:.2f}% - SRAM Loss: {sram_loss:.4f}, SRAM Acc: {sram_acc:.2f}%")

# Save trained model
torch.save(model.state_dict(), "models/3metacompact4l_30x8test.pth")            # Set model name
print("Model saved")

# Plot training accuracy and loss for DRAM and SRAM
plot(epochs,dram_acc_res,sram_acc_res,"DRAM Acc","SRAM Acc","3metacompact4l_30x8_acctest","Epochs","Accuracy","Model Accuracy Over Epochs")
plot(epochs,dram_loss_res,sram_loss_res,"DRAM Loss","SRAM Loss","3metacompact4l_30x8_losstest","Epochs","Loss","Model Loss Over Epochs")

