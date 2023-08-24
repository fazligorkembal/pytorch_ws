from unet import UNet
from segmentation_dataloader_from_file import FileLoader
from dice_coef import dice_loss

from torch.utils.data import DataLoader
import torchvision
from torch import optim
import torch
from torch import nn
import torch.nn.functional as F

def train(model=None, train_loader=None):
    n_train = len(train_loader)
    train_loader = DataLoader(train_loader, batch_size=4, shuffle=True, num_workers=4)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.999, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False) #amp
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(1, 10):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            images, masks = batch
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks = masks.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=False): #amp
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred, masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), masks.float().squeeze(1), multiclass=False)
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
        print(f"Epoch {epoch} loss is {epoch_loss / n_train}")


if __name__ == "__main__":
    image_folder = "/home/user/Documents/pytorch_ws/data/PennFudanPed/PNGImages"
    mask_folder = "/home/user/Documents/pytorch_ws/data/PennFudanPed/PedMasks"

    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model = model.to(memory_format=torch.channels_last)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(300, 300), antialias=True),
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])
    mask_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(300, 300), antialias=True),
    ])
    
    train_loader = FileLoader(image_folder, mask_folder, image_transforms, mask_transforms)
    
    
    train(model, train_loader)