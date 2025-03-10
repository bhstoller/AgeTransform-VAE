import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os

class MorphII_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        file_path = row['filepath']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist. Check CSV paths.")
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Image at {file_path} could not be loaded.")
        # Convert from BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize age (assumed between 0 and 100) and cast gender to float.
        age = row['age']
        gender = row['gender']
        condition = torch.tensor([age / 100.0, float(gender)], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, condition