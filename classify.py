import cv2
import numpy as np
import torch

from heat_map import apply_gradcam, generate_heatmap, visualize_heatmap
from video_CNN import CNN3D
import CNN_Utils


def extract_frames(video_path, frame_block_size=16):
    """
    Extract frames from the video and return a list of frame blocks.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to RGB (cv2 reads frames in BGR format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    # Convert to numpy array and return as blocks of frames
    frames = np.array(frames)
    num_blocks = len(frames) // frame_block_size
    frame_blocks = np.array_split(frames[:num_blocks * frame_block_size], num_blocks)
    return frame_blocks


def predict_from_video(model, frame_blocks):

    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():
        for block in frame_blocks:
            block = torch.tensor(block).float().unsqueeze(0)  # Add batch dimension
            outputs = model(block)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())

    return predictions


def save_annotated_video(video_path, predictions, output_path, frame_block_size=51):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    block_index = 0

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display the prediction on the frame
        if frame_count % ((block_index + 1) * frame_block_size) > 0 and (block_index + 1) < len(predictions):
            block_index += 1

        # if block_index < len(predictions):
        label = f"Prediction: {predictions[block_index]}"

        # Add text overlay to the frame
        cv2.putText(frame, label, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert frame back to BGR for writing
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


def main():
    orig_vid_path = '/Users/cameroncamp/Desktop/player_labeled_output_with_ball.mp4'
    video_path = '/Users/cameroncamp/PycharmProjects/pythonProject/player_labeled_output.mp4'
    frame_blocks = extract_frames(video_path, frame_block_size=51)
    frame_blocks = np.array(frame_blocks)
    frame_blocks = frame_blocks.transpose(0, 4, 1, 2, 3)  # Convert to (num_blocks, channels, depth, height, width)
    input_shape = frame_blocks.shape
    num_classes = 9

    # Create the 3D CNN model
    model = CNN3D(input_shape, num_classes)
    # Load model weights if available
    CNN_Utils.load_checkpoint(torch.load("model_weights.pth"), model, torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0))
    # Predict from video
    predictions = predict_from_video(model, frame_blocks)
    labels = []
    for pred in predictions:
        label = CNN_Utils.reverse_assign_shot(pred)
        labels.append(label)
    print("Predictions from video: ", labels)

    # Save video with predictions
    output_path = 'classified_video.mp4'
    save_annotated_video(orig_vid_path, labels, output_path)

    # Load an image
    image_path = '/Users/cameroncamp/Desktop/Screenshot 2024-06-10 at 8.47.31 PM.png'
    # image_path = "/Users/cameroncamp/Desktop/Screenshot 2024-06-10 at 8.48.33 PM.png"
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    #image = np.transpose(image, (2, 0, 1))
    #image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    image = np.array(image)
    # Convert to (num_blocks, channels, depth, height, width)
    reshaped_image = np.expand_dims(np.expand_dims(image, axis=3), axis=4)
    reshaped_image = reshaped_image.transpose(4, 2, 3, 0, 1)
    block = np.tile(reshaped_image, (1, 1, 51, 1, 1))
    block = torch.tensor(block, dtype=torch.float32)

    # Apply Grad-CAM
    #target_layer = model.conv3  # Change this to the layer you want to apply Grad-CAM to
    #target_layer = "Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))"
    target_layer = 'conv3'
    grad_cam = apply_gradcam(model, block, target_layer)
    single_batch = block[0]  # Shape: (3, 51, 64, 64)

    # Select a specific depth slice, e.g., the 8th frame (index 7)
    depth_index = 0
    single_image = single_batch[:, depth_index, :, :]  # Shape: (3, 64, 64)
    # print(single_image.shape)

    # Generate heatmap
    heatmap = generate_heatmap(single_image.numpy().transpose(1, 2, 0), grad_cam)
    # Visualize heatmap
    #single_image = torch.tensor(single_image, dtype=torch.float32)
    visualize_heatmap(image, heatmap)


main()
