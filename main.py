import json
import math

import numpy as np
import cv2
import torch
from torch import optim

from loss import YoloLoss
from model import Yolov1
from utils import load_checkpoint


def choose_box_from_trajectory(bboxes, trajectory_path, max_value_ids, frame_index):
    with open(trajectory_path, 'r') as file:
        trajectories = json.load(file)
    if not trajectories:
        return None

    best_bbox = None
    best_distance = float('inf')

    y_diff, x_diff = trajectories[frame_index]
    # traj_angle = math.atan2(y_diff, x_diff)
    for id in max_value_ids:
        bbox = bboxes[id]
        x_min, y_min, x_max, y_max = bbox
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

        # bbox_angle = math.atan2(y_center - y_min, x_center - x_min)  # Fix angle calculation
        # angle_diff = abs(traj_angle - bbox_angle)

        # Calculate the squared distance from the trajectory to the bounding box center
        squared_distance = math.sqrt((x_diff - x_center) ** 2 + (y_diff - y_center) ** 2)

        # Avoid division by zero
        # if squared_distance == 0:
        # squared_distance = 1e-10

        # Calculate inverse squared distance
        # inverse_squared_distance = 1 / squared_distance

        # Normalize the angle difference
        # normalized_angle_diff = angle_diff / math.pi  # Angle difference normalization

        # Calculate the weighted score
        # weighted_score = 0.75 * inverse_squared_distance + 0.25 * (1 - normalized_angle_diff)
        # Normalize the angle difference and squared distance

        # if weighted_score < best_weighted_score:
        #     best_weighted_score = weighted_score
        #     best_bbox = bbox

        if squared_distance < best_distance:
            best_distance = squared_distance
            best_bbox = bbox

    return best_bbox, best_distance


# get the webcam video stream
# webcam_video_stream = cv2.VideoCapture(0)
file_video_stream = cv2.VideoCapture('/Users/cameroncamp/Desktop/labeled_output.mp4')
output_video_path = "player_labeled_output.mp4"
trajectory_path = "/Users/cameroncamp/PycharmProjects/ball_tracking/ball_trajectory.json"

# Get video properties
frame_width = int(file_video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(file_video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_frame_size = (frame_width // 16, frame_height // 16)
fps = int(file_video_stream.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (64, 64))

# while(file_video_stream.isOpened):
while file_video_stream.isOpened():
    frame_index = 0
    # get the current frame from video stream
    # ret,current_frame = webcam_video_stream.read()
    ret, current_frame = file_video_stream.read()
    if not ret:
        break
    # use the video current frame instead of image
    img_to_detect = current_frame

    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]

    # convert to blob to pass into model
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)
    # recommended by yolo authors, scale factor is 0.003922=1/255, width,height of blob is 320,320
    # accepted sizes are 320×320,416×416,609×609. More size means more accuracy but less speed

    yolo_model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    optimizer = optim.Adam(
        yolo_model.parameters(), lr=2e-5, weight_decay=0
    )
    yolo_loss = YoloLoss()

    load_checkpoint(torch.load("trained_player_model.pth.tar"), yolo_model, optimizer)

    # Get all layers from the yolo network
    # Loop and find the last layer (output layer) of the yolo network
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[yolo_layer - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

    # input preprocessed blob into model and pass through the model
    yolo_model.setInput(img_blob)
    # obtain the detection layers by forwarding through till the output layer
    obj_detection_layers = yolo_model.forward(yolo_output_layer)

    # initialization for non-max suppression (NMS)
    # declare list for [class id], [box center, width & height[], [confidences]
    class_ids_list = []
    boxes_list = []
    confidences_list = []
    class_labels = ["Person"]

    # loop over each of the layer outputs
    for object_detection_layer in obj_detection_layers:
        # loop over the detections
        for object_detection in object_detection_layer:

            # obj_detections[1 to 4] => will have the two center points, box width and box height
            # obj_detections[5] => will have scores for all objects within bounding box
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]

            # take only predictions with confidence more than 20%
            if predicted_class_id == 0 and prediction_confidence > 0.20:
                # get the predicted label
                predicted_class_label = class_labels[predicted_class_id]
                # obtain the bounding box co-oridnates for actual image from resized image size
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))

                # save class id, start x, y, width & height, confidences in a list for nms processing
                # make sure to pass confidence as float and width and height as integers
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])

    # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes
    # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    box, dist = choose_box_from_trajectory(boxes_list, trajectory_path, max_value_ids, frame_index)
    # loop through the final set of detections remaining after NMS and draw bounding box and write text
    # for max_valueid in max_value_ids:
    #     max_class_id = max_valueid
    #     box = boxes_list[max_class_id]
    start_x_pt = box[0]
    start_y_pt = box[1]
    box_width = box[2]
    box_height = box[3]

    # get the predicted class id and label
    # predicted_class_id = class_ids_list[max_class_id]
    # predicted_class_label = class_labels[predicted_class_id]
    # prediction_confidence = confidences_list[max_class_id]
    ############## NMS Change 3 END ###########

    end_x_pt = start_x_pt + box_width
    end_y_pt = start_y_pt + box_height

    # label_size, _ = cv2.getTextSize(str(dist), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    # label_width, label_height = label_size

    # Make sure the label does not go out of the image
    # y_label = start_y_pt - 10 if start_y_pt - 10 > 10 else start_y_pt + label_height + 10
    # x_label = start_x_pt
    frame_index += 1

    # draw rectangle and text in the image
    cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0, 255, 0), 1)
    # cv2.putText(img_to_detect, str(dist), (x_label, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #out.write(current_frame)
    # Crop the frame to the bounding box
    cropped_frame = current_frame[start_y_pt:end_y_pt, start_x_pt:end_x_pt]
    cropped_frame = cv2.resize(cropped_frame, (64, 64))
    #cv2.imshow('image', cropped_frame)

    # Wait indefinitely until any key is pressed
    # cv2.waitKey(0)
    #
    # # Destroy the window
    # cv2.destroyAllWindows()

    # Write the cropped frame to the output video
    out.write(cropped_frame)

# releasing the stream and the camera
# close all opencv windows
# webcam_video_stream.release()
file_video_stream.release()
out.release()
cv2.destroyAllWindows()
