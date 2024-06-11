import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import torch
from torch import optim

from model import Yolov1
from utils import load_checkpoint


def make_heatmap(img_path, model):
    # Load and preprocess the image
    img = cv2.imread(img_path)

    # Load the pre-trained model
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    load_checkpoint(torch.load("trained_model.pth.tar"), model, optimizer)
    print("Model Loaded")

    # Make predictions
    preds = model.predict(img_array)

    # Get the predicted class index
    class_idx = np.argmax(preds[0])

    # Get the last convolutional layer
    last_conv_layer = model.get_layer('block5_conv3')

    # Create a model that maps the input image to the activations of the last conv layer
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # Create a model that maps the activations of the last conv layer to the final predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ['block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # Get the activations of the last conv layer and make the gradient tape watch it
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        class_channel = preds[:, class_idx]

    # Compute the gradient of the class output with respect to the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool the gradients over all the axes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by the corresponding gradients
    last_conv_layer_output = last_conv_layer_output[0]
    pooled_grads = pooled_grads.numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()

    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Average the feature map to get the heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # Apply ReLU to the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Load the original image
    img = cv2.imread(img_path)

    # Resize the heatmap to the size of the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    overlay_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Display the overlay image
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()