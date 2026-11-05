# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
The problem statement for this experiment is to develop an image classification model that can accurately distinguish between 'defect' and 'notdefect' semiconductor chip images. This is a binary classification task, where the goal is to leverage transfer learning using a pre-trained VGG19 model to effectively classify new, unseen chip images.


## Neural Network Model

<img width="1429" height="1100" alt="Fully Connected Neural Network Schematic" src="https://github.com/user-attachments/assets/3de8a905-0ea1-412d-a307-54d93d005de6" />


## DESIGN STEPS
### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 
Visualize sample images from the dataset.


### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.


### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.




## PROGRAM

### Name: SUJIN M L

### Register Number: 21222504035

```python

# NAME : SUJIN M L
# REGISTER NUMBER : 212225040435

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset_path = "C:\\Users\\acer\\Downloads\\chip_data\\dataset"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0) 
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()

show_sample_images(train_dataset)

print(f"Total number of training samples: {len(train_dataset)}")
print(f"Total number of testing samples: {len(test_dataset)}")
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
summary(model, input_size=(3, 224, 224))

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
    print("NAME : SUJIN M L \nREGISTER NUMBER :212225040435")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    print("NAME : SUJIN M L \nREGISTER NUMBER :212225040435")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print("NAME: SUJIN M L")
    print("REGISTER NUMBER : 212225040435")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

train_model(model, train_loader,test_loader)
test_model(model, test_loader)

def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _,pred=torch.max(output,1)
        prob = torch.sigmoid(pred)
        predicted = (prob > 0.5).int().item()

    class_names = class_names = dataset.classes
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

predict_image(model, image_index=55, dataset=test_dataset)

predict_image(model, image_index=25, dataset=test_dataset)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="492" height="210" alt="image" src="https://github.com/user-attachments/assets/aa148644-a0d6-4961-a5db-ebcb568141d6" />.

<img width="772" height="637" alt="image" src="https://github.com/user-attachments/assets/d7c98d5f-1fa6-41f4-9cab-93784341c204" />


## Confusion Matrix
<img width="691" height="662" alt="image" src="https://github.com/user-attachments/assets/561dca77-f0eb-485e-914f-7381cb03d7a7" />



## Classification Report
<img width="462" height="233" alt="image" src="https://github.com/user-attachments/assets/56f7ce47-19c5-4d0f-9ade-e4057e730804" />


### New Sample Data Prediction
<img width="363" height="401" alt="image" src="https://github.com/user-attachments/assets/b1718fea-3bf1-4b75-a552-d8e63c5efa1c" />

<img width="355" height="397" alt="image" src="https://github.com/user-attachments/assets/9a0c1a7c-4ab0-4800-86ae-d66e4d1641f9" />


## RESULT
Thus the python program to develop an image classification model using transfer learning with VGG19 architecture is executed successfully.
