"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from utils import cellboxes_to_boxes

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


def _create_fcs(split_size, num_boxes, num_classes):
    S, B, C = split_size, num_boxes, num_classes
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024 * S * S, 496),
        nn.Dropout(0.0),
        nn.LeakyReLU(0.1),
        nn.Linear(496, S * S * (C + B * 5)),
    )


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = _create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def classify_and_extract_bounding_boxes(self, frame, output_dir, detections, frame_count):
        os.makedirs(output_dir, exist_ok=True)

        for i, box in enumerate(detections):
            class_id, confidence, x, y, w, h = box
            left, top, right, bottom = map(int, [x - w / 2, y - h / 2, x + w / 2, y + h / 2])

            left = max(0, left)
            top = max(0, top)
            right = min(frame.shape[1], right)
            bottom = min(frame.shape[0], bottom)

            if right > left and bottom > top:
                roi = frame[top:bottom, left:right]
                if roi.size != 0:
                    file_path = os.path.join(output_dir, f'frame_{frame_count}_bbox_{i}_class_{class_id}.jpg')
                    cv2.imwrite(file_path, roi)
                else:
                    print(f"Skipping empty ROI for frame {frame_count}, bbox {i}")
            else:
                print(
                    f"Invalid bounding box for frame {frame_count}, bbox {i}: left={left}, top={top}, right={right}, bottom={bottom}")

    def detect_objects(self, frame, confidence_threshold=0.5, nms_threshold=0.4):
        self.eval()

        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (448, 448))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        with torch.no_grad():
            predictions = self.forward(torch.tensor(img).float())

        # Convert predictions to bounding boxes
        print(predictions)
        bboxes = cellboxes_to_boxes(predictions)
        print(bboxes)
        bboxes = bboxes[0]  # Since we're processing one image

        # Filter out low confidence boxes
        bboxes = [bbox for bbox in bboxes if bbox[1] > confidence_threshold]

        # Perform Non-Maximum Suppression (NMS)
        bboxes = self.nms(bboxes, nms_threshold)

        return bboxes

    def nms(self, bboxes, nms_threshold=0.4):
        if len(bboxes) == 0:
            return []
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
        keep_boxes = []
        while bboxes:
            chosen_box = bboxes.pop(0)
            keep_boxes.append(chosen_box)
            bboxes = [box for box in bboxes if
                      self.iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])) < nms_threshold]
        return keep_boxes

    def iou(self, box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[
            1] + box1[3] / 2
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2] / 2, box2[
            1] + box2[3] / 2

        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area