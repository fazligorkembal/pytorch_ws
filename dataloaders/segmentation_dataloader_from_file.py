import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import torchvision

class FileLoader(Dataset):
    def __init__(self, image_folder=None, mask_folder=None, image_trasnform=None, mask_transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_transform = image_trasnform
        self.mask_transform = mask_transform
        
        data_paths = self._get_all_files_from_folder(image_folder)
        
        
        self.data_paths = self._check_and_match_files(data_paths)
        print("Image Folder: ", image_folder)
        print("Mask Folder: ", mask_folder)
        print("Number of data: ", len(self.data_paths))

    def _get_all_files_from_folder(self, folder_path):
        files = []
        for dirname, _, filenames in os.walk(folder_path):
            for filename in filenames:
                files.append(os.path.join(dirname, filename))
        return files
    

    #Owerrite: Inherit and change this for different file named datasets ... !
    def _check_and_match_files(self, data_paths):
        data = {}
        for data_path in data_paths:
            if not os.path.exists(data_path):
                raise Exception(f"File {data_path} does not exist!")
            mask_name = data_path.split('/')[-1].split('.')[0] + '_mask.png'
            mask_path = os.path.join(self.mask_folder, mask_name)
            if not os.path.exists(mask_path):
                raise Exception(f"File {mask_path} does not exist!")
            
            data[data_path] = mask_path
        return data
            
    
    def __len__(self):
        return len(self.data_paths)
    

    def trasnform(self, image, mask):
        random = torch.rand(1)
        if random > 0.5:
            image = torchvision.transforms.functional.hflip(image)
            mask = torchvision.transforms.functional.hflip(mask)
        if self.image_transform:
            image = self.image_transform(image) / 255.0
        if self.mask_transform:
            mask = self.mask_transform(mask)    
        return image, mask


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_path = list(self.data_paths.keys())[idx]
        mask_path = list(self.data_paths.values())[idx]
        
        image = read_image(image_path)
        mask = read_image(mask_path)
        
        image, mask = self.trasnform(image, mask)
        return image, mask


if __name__ == "__main__":
    image_folder = "/home/user/Documents/pytorch_ws/data/PennFudanPed/PNGImages"
    mask_folder = "/home/user/Documents/pytorch_ws/data/PennFudanPed/PedMasks"


    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(300, 300)),
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])
    mask_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(300, 300)),
    ])

    dataset = FileLoader(image_folder, mask_folder, image_transforms, mask_transforms)
    
    import cv2

    for i in range(170):
        image, mask = dataset.__getitem__(i)
        cv2.imshow("image", image.permute(1, 2, 0).numpy())
        cv2.imshow("mask", mask.permute(1, 2, 0).numpy() * 255)
        cv2.waitKey(0)