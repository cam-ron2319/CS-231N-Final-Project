import numpy as np
import cv2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def apply_gradcam(model, image, target_layer_name):
    """
    Applies Grad-CAM to an image and generates a heatmap.

    Args:
        model: The trained 3D CNN model.
        image: The input image.
        target_layer_name: The name of the layer at which Grad-CAM is applied.

    Returns:
        heatmap: The generated heatmap.
    """
    # Set the model to evaluation mode
    model.eval()

    # Register hooks to get the gradients and activations from the target layer
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    target_layer = dict([*model.named_modules()])[target_layer_name]
    target_layer.register_forward_hook(forward_hook)

    # Forward pass
    output = model(image)

    # Get the index of the class with the highest score
    pred_class = output.argmax().item()

    # Backward pass
    model.zero_grad()
    target = output[0, pred_class]
    target.backward()

    # Get the gradients and activations
    gradients = gradients[0].cpu().data.numpy()[0]
    activations = activations[0].cpu().data.numpy()[0]

    # Make all values positive
    gradients = np.abs(gradients)
    activations = np.abs(activations)

    # print(gradients)
    # print(activations)

    # Compute the weights
    weights = np.mean(gradients, axis=(1, 2, 3))

    # Compute the Grad-CAM
    grad_cam = np.zeros(activations.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        #print(w * activations[i, 0, :, :])
        grad_cam += w * activations[i, 0, :, :]

    # Apply ReLU
    grad_cam = np.maximum(grad_cam, 0)

    # Normalize the Grad-CAM
    if grad_cam.max() != grad_cam.min():  # Avoid division by zero
        grad_cam -= grad_cam.min()
        grad_cam /= grad_cam.max()
    else:
        grad_cam.fill(0)  # If all values are the same, make the heatmap zero

    grad_cam = cv2.resize(grad_cam, (image.shape[3], image.shape[4]))
    return grad_cam


def generate_heatmap(image, grad_cam):
    """
    Generates a heatmap from the Grad-CAM and overlays it on the image.

    Args:
        image: The input image.
        grad_cam: The Grad-CAM.

    Returns:
        heatmap: The generated heatmap.
    """
    grad_cam = np.uint8(255 * grad_cam)
    heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) #/ 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def visualize_heatmap(image, heatmap):
    """
    Visualizes the heatmap.

    Args:
        image: The input image.
        heatmap: The generated heatmap.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image) #.transpose(1, 2, 0))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.title('Heatmap')
    plt.show()
