import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import cv2

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
}

data_dir = "/home/user/Documents/pytorch_ws/data/hymenoptera_data"
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        data_transforms[x]
    ) for x in ['train', 'val']
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=4,
        shuffle=True,
        num_workers=4
    ) for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print("dataset_sizes: ", dataset_sizes)
print("class_names: ", class_names)
print("device: ", device)

inputs, classes = next(iter(dataloaders['train']))


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, model_name="resnet.pt"):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    
    

    torch.save(model.state_dict(), model_name)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), model_name)
                print(f"Model saved in {model_name}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print(f"Model saved in {model_name}")
    # load best model weights
    model.load_state_dict(torch.load(model_name))
    return model


def get_all_images_from_root_path(root_path):
    all_images = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".jpg"):
                all_images.append(os.path.join(root, file))
    return all_images

def visualize_model_predictions(model, img_folder):
    was_training = model.training
    model.eval()

    
    image_paths = get_all_images_from_root_path(img_folder)
    last_img = None

    for img_path in image_paths:

        img = Image.open(img_path)
        img = data_transforms['val'](img)
        img = img.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)

            print("Preds: \n", preds)

            model.train(mode=was_training)

            #cv2.imshow("Image", cv2.imread(img_path))
            #cv2.waitKey(0)
        last_img = img
    
    import torchsummary
    torchsummary.summary(model, input_size=(3, 224, 224), device="cuda")
    

def train(model_name="resnet.pt"):
    model_ft = models.resnet18(weights="IMAGENET1K_V1")
    for param in model_ft.parameters():
        param.requires_grad = False

    print("model attributes: \n", dir(model_ft))
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=25,
        model_name=model_name
    )


def model_load(model_name="resnet.pt"):
    model_ft = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.to(device)
    model_ft.load_state_dict(torch.load(model_name))
    model_ft.eval()

    visualize_model_predictions(
        model_ft,
        img_folder='/home/user/Documents/pytorch_ws/data/hymenoptera_data/val/bees'
    )


if __name__ == "__main__":
    model_name = "resnet.pt"
    #train(model_name=model_name)
    model_load(model_name=model_name)