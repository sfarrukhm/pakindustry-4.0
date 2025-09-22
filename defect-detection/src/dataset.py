import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CastDefectDataset(Dataset):
    """
    Custom Dataset for Cast Part Defect Detection
    Loads grayscale images, converts to 3-channel, applies transforms.
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])

        # Open as grayscale
        image = Image.open(img_path).convert("L")
        image = transforms.ToTensor()(image)
        image = image.repeat(3, 1, 1)  # expand to 3 channels

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label
