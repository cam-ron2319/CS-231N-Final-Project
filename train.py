import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from model import Yolov1
from dataset import VOCDataset
from utils import mean_average_precision, get_bboxes, save_checkpoint, load_checkpoint, load_yolo_dataset, \
    process_images_and_annotations, visualize_detections
from loss import YoloLoss
import matplotlib.pyplot as plt

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
# DEVICE = "cuda" if torch.cuda.is_available else "cpu"
DEVICE = torch.device('cpu')
BATCH_SIZE = 16  # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "trained_model.pth.tar"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    return sum(mean_loss) / len(mean_loss)


def test_fn(test_loader, model, loss_fn):
    loop = tqdm(test_loader, leave=True)
    mean_loss = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())

            # update progress bar
            loop.set_postfix(loss=loss.item())

    return sum(mean_loss) / len(mean_loss)


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    yolo_loss = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    images, labels = process_images_and_annotations("/Users/cameroncamp/Downloads/people_detection/images", "/Users/cameroncamp/Downloads/people_detection/annotations.xml")
    full_dataset = VOCDataset(images, labels, transform=transform)
    train_ratio = 0.8
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    map_values = []
    epochs = []
    losses = []
    for epoch in range(EPOCHS):
        print(f"Epoch:{epoch}")
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        print(pred_boxes, target_boxes)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Train mean average precision (MAP) for Epoch {epoch}: {mean_avg_prec}")

        if mean_avg_prec > 0.2:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        map_values.append(mean_avg_prec)
        epochs.append(epoch)

        mean_loss = train_fn(train_loader, model, optimizer, yolo_loss)
        losses.append(mean_loss)
        print(f"Mean Loss after Epoch {epoch}: {mean_loss}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, map_values, label='MAP')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Average Precision (MAP)')
    plt.title('MAP vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses, label='Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate on the test set
    test_loss = test_fn(test_loader, model, yolo_loss)
    print(f"Test Loss: {test_loss}")

    # pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4)
    # test_map = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
    # print(f"Test mean average precision (MAP): {test_map}")
    #
    # # Visualization of detections
    # images, _ = zip(*[test_dataset[i] for i in range(len(test_dataset))])
    # visualize_detections(images, pred_boxes, class_names, 'visualized_detections')
    #
    # # Confusion Matrix
    # y_true = [target[-1] for target in target_boxes]
    # y_pred = [pred[-1] for pred in pred_boxes]
    # #plot_confusion_matrix(y_true, y_pred, class_names, 'confusion_matrix.png')


if __name__ == "__main__":
    main()
