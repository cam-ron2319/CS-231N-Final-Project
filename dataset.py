"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""
import torch
from torch.utils.data import Dataset
from PIL import Image


class VOCDataset(Dataset):
    def __init__(self, img_paths, labels, S=7, B=2, C=20, transform=None):
        """
        Args:
            img_paths (list): List of paths to the images.
            labels (list): List of labels, where each label is a list of bounding boxes.
                           Each bounding box is [class_label, x, y, width, height].
            S (int): Number of grid cells. Default is 7.
            B (int): Number of bounding boxes per grid cell. Default is 2.
            C (int): Number of classes. Default is 20.
            transform (callable, optional): A function/transform to apply to the images and labels.
        """
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Load image
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert("RGB")
        # Get the corresponding labels
        #boxes = self.labels[index]
        #boxes = torch.tensor(boxes)
        #print(index)
        #print(self.labels)
        box = self.labels[index]

        # Apply transformations if any
        if self.transform:
            image, box = self.transform(image, box)

        # Convert to cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        #for box in boxes:
            #print(box.tolist())
        class_label, x, y, width, height = box #.tolist()
        class_label = int(class_label)

         # i,j represents the cell row and cell column
        #print(x, y)
        i, j = int(self.S * y), int(self.S * x)
        #print(j, i)
        x_cell, y_cell = self.S * x - j, self.S * y - i

        # Calculating the width and height of cell of bounding box, relative to the cell
        width_cell, height_cell = (
            width * self.S,
            height * self.S,
        )

        # If no object already found for specific cell i,j
        # Note: This means we restrict to ONE object per cell!
        if label_matrix[i, j, 20] == 0:
            # Set that there exists an object
            label_matrix[i, j, 20] = 1

            # Box coordinates
            box_coordinates = torch.tensor(
                [x_cell, y_cell, width_cell, height_cell]
            )

            label_matrix[i, j, 21:25] = box_coordinates

            # Set one hot encoding for class_label
            label_matrix[i, j, class_label] = 1

        return image, label_matrix