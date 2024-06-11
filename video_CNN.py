import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split

import CNN_Utils
from matplotlib import pyplot as plt


# Define the 3D CNN model
class CNN3D(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN3D, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv3d(in_channels=input_shape[1], out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # 2nd convolutional layer
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # 3rd convolutional layer
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Fully connected layers
        self.fc1 = nn.Linear(49152, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten the 3D feature maps
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def train_model(self, train_loader, loss_function, optimizer, num_epochs=30):
        losses = []
        # Training loop
        for epoch in range(num_epochs):
            self.train()  # Set the model to train mode
            running_loss = 0.0

            for batch_idx, (data, targets) in enumerate(train_loader):
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(data.float())
                CNN_Utils.compute_confusion_matrix(targets, outputs)
                loss = loss_function(outputs, targets)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            losses.append(epoch_loss)  # Append the loss to the list of losses
            checkpoint = {"state_dict": self.state_dict(), "optimizer": optimizer.state_dict()}
            CNN_Utils.save_checkpoint(checkpoint, filename='model_weights.pth')
            print('Epoch %d Loss: %.3f' % (epoch + 1, epoch_loss))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss over Epochs')
        plt.show()

    def evaluate_model(self, test_loader, loss_function):
        total_loss = 0
        for batch_idx, (data, targets) in enumerate(test_loader):
            outputs = self(data.float())
            loss = loss_function(outputs, targets).item()
            total_loss += loss
        average_loss = total_loss / len(test_loader)
        print("Average Loss on Test Data: ", average_loss)


def train():
    frame_blocks, true = CNN_Utils.get_frames()
    frame_blocks = frame_blocks.transpose(0, 4, 1, 2, 3)
    batch_size = 32
    full_dataset = CNN_Utils.create_train_loader(frame_blocks, true, batch_size=batch_size)
    train_ratio = 0.8
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_loader, test_loader = random_split(full_dataset, [train_size, test_size])
    input_shape = np.array(frame_blocks.shape)
    print(input_shape)
    input_shape[0] -= train_size
    num_classes = 9

    # Create the 3D CNN model
    model = CNN3D(input_shape, num_classes)
    # model_weights = torch.load('model_weights.pth')
    # model.load_state_dict(model_weights)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    CNN_Utils.load_checkpoint(torch.load("model_weights.pth"), model, optimizer)
    print("Model Loaded")
    num_epochs = 100
    model.train_model(train_loader.dataset, loss_function, optimizer, num_epochs=num_epochs)
    model.evaluate_model(test_loader.dataset, loss_function)


train()
