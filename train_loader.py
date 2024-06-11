import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os


# Define a custom dataset class
class TennisDataset(Dataset):
    def __init__(self, frames_dir, labels, transform=None):
        self.frames_dir = frames_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.frames_dir, f"frame_{idx}.jpg")
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# Function to create train_loader
def create_train_loader(video_path, labels, batch_size=32, shuffle=True):
    # Extract frames from video
    frames_dir = 'frames'
    frame_extractor.extract_frames(video_path, frames_dir)

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization parameters
    ])

    # Create dataset
    dataset = TennisDataset(frames_dir, labels, transform)

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader


def main():
    video_path = "/Users/cameroncamp/Downloads/Novak Djokovic v Rafael Nadal Full Match | Australian Open 2019 Final.mp4"
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_loader = create_train_loader(video_path, labels)
