import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ButterflyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Map species to label indices
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.annotations['label'].unique()))}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.label2idx[self.annotations.iloc[index, 1]]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # To match Tanh activation in GANs
])