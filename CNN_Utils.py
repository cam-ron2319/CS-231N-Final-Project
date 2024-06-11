import math
import os

import fitz
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn import metrics


def extract_data_from_pdf(pdf_path):
    '''Extracts the labeled data from the pdf file.'''
    doc = fitz.open(pdf_path)
    page = doc[0]  # Assuming the data is on the first page
    text = page.get_text()
    lines = text.split('\n')[1:]  # Skip the header line
    data = [line.split(',') for line in lines]
    df_columns = ["index", "rallyid", "frameid", "strokeid", "hitter", "receiver", "isserve", "serve", "type",
                  "stroke", "hitter_x", "hitter_y", "receiver_x", "receiver_y", "time"]
    df = pd.DataFrame(data, columns=df_columns)
    return df


def assign_shot(row):
    '''Assigns each shot class a numeric value according
    to the table below.'''
    # Shot-Index Map:
    # Top-spin Forehand: 0
    # Return Forehand: 1
    # Slice Forehand: 2
    # Volley Forehand: 3
    # Serve Forehand (serve): 4
    # Top-spin Backhand: 0 + 5 = 5
    # Return Backhand: 1 + 5 = 6
    # Slice Forehand: 2 + 5 = 7
    # Volley Forehand: 3 + 5 = 8

    shot_num = 0  # Initialized with "topspin".
    if row["type"] == "return":
        shot_num = 1
    elif row["type"] == "slice":
        shot_num = 2
    elif row["type"] == "volley":
        shot_num = 3
    elif row["type"] == "serve":
        shot_num = 4
    if row["stroke"] == "backhand":
        shot_num += 5
    return shot_num


def reverse_assign_shot(shot_num):
    '''Converts a numeric shot value back to its corresponding
    shot type and stroke.'''
    # Index-Shot Map:
    shot_types = ["topspin", "return", "slice", "volley", "serve"]
    strokes = ["forehand", "backhand"]

    if shot_num < 0 or shot_num > 8:
        raise ValueError("Invalid shot number")

    # Determine stroke
    if shot_num >= 5:
        stroke = strokes[1]  # backhand
        shot_num -= 5
    else:
        stroke = strokes[0]  # forehand

    # Determine shot type
    shot_type = shot_types[shot_num]

    return shot_type + " " + stroke


def preprocess_frame(frame):
    '''Reduce the dimensionality of the frame for efficiency
    in convolution.'''
    # Resize the frame to a lower resolution.
    resized_frame = cv2.resize(frame, (64, 64))

    # Convert the frame to float32 for normalization.
    normalized_frame = resized_frame.astype(np.float32)

    # Normalize RGB values by subtracting mean and dividing by standard deviation.
    mean = np.mean(normalized_frame, axis=(0, 1))
    std = np.std(normalized_frame, axis=(0, 1))
    normalized_frame = (normalized_frame - mean) / std
    return normalized_frame


def get_frames_around_timestamp(video_path, timestamp, margin):
    '''Gets frames one second before and one second after the timestamp in the video.
    Note: May need to pad with reiterations of frames to ensure correct dimensions (i.e. starts at 0 seconds)'''
    cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 25
    # target_frame = int(timestamp * fps)

    # Calculate frame indices for the time margin
    # start_frame = max(0, target_frame - int(fps * margin))
    # end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, target_frame + int(fps * margin))
    timestamp = math.ceil(timestamp)
    start_frame_timestamp = timestamp - int(fps * margin)
    end_frame_timestamp = timestamp + int(fps * margin)

    # Extract frames within the time margin
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_timestamp)
    for i in range(start_frame_timestamp, end_frame_timestamp + 1):
        ret, frame = cap.read()
        if ret:
            processed_frame = preprocess_frame(frame)
            frames.append(np.array(processed_frame))
        else:
            break
    cap.release()
    print("Block Extracted")
    return frames


def get_frames():
    '''Gets all the frames and their true classifications.'''
    pdf_path = "/Users/cameroncamp/PycharmProjects/tennis_CNN/tennis_data.pdf"
    video_path = "/Users/cameroncamp/Downloads/Novak Djokovic v Rafael Nadal Full Match | Australian Open 2019 Final.mp4"
    df = extract_data_from_pdf(pdf_path)
    # num_blocks = df.shape[0]  # The number of classifications in the training data (num_rows).
    num_blocks = 64
    frame_rate = 25
    timestamp_offset = 35.75
    block_size = 2 * frame_rate + 1  # Each labeled block of frames includes the frame with the label and each frame 1 sec before and after.
    frame_height = 64
    frame_width = 64
    frame_channels = 3
    num_shots = 9
    margin = 1
    frame_blocks = np.zeros((num_blocks, block_size, frame_height, frame_width, frame_channels))
    true = np.zeros((num_blocks, num_shots))
    for index, row in df.iterrows():
        print(index)
        if index == num_blocks:
            break
        if row["time"] is not None:
            probs = np.zeros(9)
            timestamp = float(row["time"]) + timestamp_offset
            frame_block = get_frames_around_timestamp(video_path, timestamp, margin)
            processed_frames = [preprocess_frame(frame) for frame in frame_block]
            frame_blocks[index] = np.array(processed_frames)
            shot_num = assign_shot(row)
            probs[shot_num] = 1
            true[index] = probs
    print("Frames Extracted!")
    return frame_blocks, true


class TennisDataset(Dataset):
    def __init__(self, frame_blocks, true_labels, transform=None):
        self.frame_blocks = frame_blocks
        self.true_labels = true_labels
        self.transform = transform

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, idx):
        frames = self.frame_blocks[idx]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        label = self.true_labels[idx]
        return frames, label


# Function to create train_loader
def create_train_loader(frame_blocks, true_labels, batch_size=32, shuffle=True):
    # Create dataset
    dataset = TennisDataset(frame_blocks, true_labels)

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader


class TennisBallDataset(Dataset):
    def __init__(self, image_paths, labels, S=7, B=2, C=1, img_size=448):
        self.image_paths = image_paths
        self.labels = labels
        self.S = S
        self.B = B
        self.C = C
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = self.labels[idx]
        target = self.encode_label(label)
        return img, target

    def encode_label(self, label):
        grid_size = self.img_size / self.S
        target = torch.zeros((self.S, self.S, self.C + self.B * 5))
        class_label, x_center, y_center, width, height = label
        x_center *= self.img_size
        y_center *= self.img_size
        width *= self.img_size
        height *= self.img_size

        grid_x = int(x_center / grid_size)
        grid_y = int(y_center / grid_size)

        x_cell = (x_center - grid_x * grid_size) / grid_size
        y_cell = (y_center - grid_y * grid_size) / grid_size
        w_cell = width / self.img_size
        h_cell = height / self.img_size
        if target[grid_y, grid_x, self.C] == 0:
            target[grid_y, grid_x, self.C] = 1
            target[grid_y, grid_x, self.C + 1:self.C + 5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
            target[grid_y, grid_x, :self.C] = torch.tensor(class_label)

        return target


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def compute_confusion_matrix(actual, predicted):
    max_values, _ = torch.max(predicted, dim=1, keepdim=True)
    prediction = torch.zeros_like(predicted)
    prediction[predicted == max_values] = 1
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    confusion_matrices = multilabel_confusion_matrix(actual, prediction)

    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    #cm_display.plot()
    #plt.show()

    # Visualize the confusion matrices
    for i, matrix in enumerate(confusion_matrices):
        plt.figure(figsize=(5, 5))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Class {i}')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['Predicted 0', 'Predicted 1'])
        plt.yticks(tick_marks, ['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
