import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
rich_face_input_dir = './Final_Project_ML-main/richFaces_asia_focus_extract_green'
n_face_input_dir = './Final_Project_ML-main/poorFaces_asia_focus_extract_green'
# label_input_dur = './label'
rich_n = 0
normal_n = 0
img_list = []
n_list = []

class FacialWealthDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class FacialWealthCNN(nn.Module):
    def __init__(self):
        super(FacialWealthCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 25 * 25, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((256, 256)),
    transforms.Grayscale()
])

for filename in os.listdir(rich_face_input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = Image.open(os.path.join(rich_face_input_dir, filename))
        img_tensor = transform(image)
        img_list.append(img_tensor)
        rich_n += 1

for filename in os.listdir(n_face_input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = Image.open(os.path.join(n_face_input_dir, filename))
        resized_image = image.resize((200, 200))
        grayscale_image = resized_image.convert('L')
        np_image = np.array(grayscale_image)
        img_list.append(np_image)
        normal_n += 1

print(normal_n, rich_n, len(img_list))

img_list
label_n = np.ones(len(img_list))
label_n[:rich_n] = 0

train_data, remaining_data, train_labels, remaining_labels = train_test_split(img_list, label_n, test_size=0.2, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(remaining_data, remaining_labels, test_size=0.5, random_state=42)
train_dataset = FacialWealthDataset(train_data, train_labels)
val_dataset = FacialWealthDataset(val_data, val_labels)
test_dataset = FacialWealthDataset(test_data, test_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FacialWealthCNN()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0

    model.train()
    for inputs, labels in train_dataloader:
        # print(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

        val_loss /= len(val_dataloader)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

test_loss /= len(test_dataloader)
print(f"Test Loss: {test_loss:.4f}")