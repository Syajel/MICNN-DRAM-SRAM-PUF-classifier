import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInputCNN_SRAM(nn.Module):
    def __init__(self, num_classes=146):
        super(MultiInputCNN_SRAM, self).__init__()

        # Convolution layers and pooling layers shared for both DRAM and SRAM
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)  # After Conv1

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)  # After Conv2

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)  # After Conv3

        # Flattened sizes calculation for DRAM and SRAM
        self.dram_flattened_size = 128 * 9 * 36  # DRAM Output Size
        self.sram_flattened_size = 128 * 9 * 44  # SRAM Output Size

        # Fully connected layer
        self.fc_dram = nn.Linear(self.dram_flattened_size, 512)
        self.fc_sram = nn.Linear(self.sram_flattened_size, 512)

        # Metadata layer
        self.fc_meta = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Fusion layer
        self.fc_fusion = nn.Sequential(
            nn.Linear(512 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, metadata=None, is_dram=True):
        # Each convolution followed by a ReLU activation function
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)

        if is_dram:
            x = F.relu(self.fc_dram(x))
            meta = self.fc_meta(metadata).squeeze(1)
        else:
            x = F.relu(self.fc_sram(x))
            meta = torch.zeros((x.shape[0], 16), device=x.device)       #Placeholder metadata values for SRAM

        combined = torch.cat((x, meta), dim=1)
        output = self.fc_fusion(combined)
        return output

