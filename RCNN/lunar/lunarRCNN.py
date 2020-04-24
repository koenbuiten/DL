import os
import numpy as np
import torch
import pandas as pd
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import engine
import transforms as T
from torchvision.transforms import Normalize

import time
import utils
from engine import train_one_epoch
import matplotlib.pyplot as plt

lunar_loc = '/media/koenbuiten/8675c03f-5bb1-4466-8581-8f042a79029b/koenbuiten/Datasets/artificial-lunar-rocky-landscape-dataset'
# lunar_loc = '/data/s3861023/lunarDataset'

class LunarDataset(object):
    def __init__(self, data_loc, transforms):
        self.data_loc = data_loc
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(data_loc, "images/render"))))
        self.masks = list(sorted(os.listdir(os.path.join(data_loc, "images/clean"))))

        self.boxes = pd.read_csv(os.path.join(data_loc, 'bounding_boxes.csv'))
        self.boxes = self.boxes.values
        boxes = []
        images = []
        j = 0

        # Create database with only the images with a bounding box
        datasize = len(self.imgs)
        datasize = 1000
        for i in range(1, datasize):
            if self.boxes[j, 0] != i:
                continue
            else:
                boxes.append([])
                images.append(self.imgs[i - 1])
                while self.boxes[j, 0] == i:
                    # convert to xMin, yMin, xMax, yMax.
                    # Coordinate system starts at top left
                    temp_box = [0, 0, 0, 0]
                    temp_box[0] = self.boxes[j, 1]
                    temp_box[1] = self.boxes[j, 2]
                    temp_box[2] = self.boxes[j, 1] + self.boxes[j, 3]
                    temp_box[3] = self.boxes[j, 2] + self.boxes[j, 4]
                    j = j + 1

                    boxes[len(boxes) - 1].append(temp_box)

        self.boxes = boxes
        self.imgs = images


    def __getitem__(self, idx):
        # load images and boxes
        img_path = os.path.join(self.data_loc, "images/render", self.imgs[idx])
        mask_path = os.path.join(self.data_loc, "images/clean", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        boxes = self.boxes[idx]

        num_objs = len(boxes)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # # convert everything into a torch.Tensor

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = self.imgs[idx]
        image_id = int(image_id[6:10])
        image_id = torch.tensor(image_id)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # transforms.append(torchvision.transforms.Normalize(2, 0.5, inplace=False))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Function to do detection from the generated model onto a random test image
# It plot the ground truth bounding boxes in the image (single plot)
#
def detect_single_image(model, data_loader):
    model.eval()
    images = list(data_loader)
    image = images[0]

    id = image[1][0]['image_id'].numpy()
    id = str(id)
    while (len(id) != 4):
        id = '0' + id

    print(id)

    output = model(images[0][0])
    predBoxes = output[0]['boxes'].detach().numpy()

    img_path = os.path.join(lunar_loc, "images/render", 'render' + id + '.png')
    img = Image.open(img_path).convert("RGB")

    gt_boxes = image[1][0]['boxes'].detach().numpy()

    no_intersection, intersectionBoxes = engine.seperate_boxes(predBoxes, gt_boxes)
    FP = len(no_intersection)

    engine.evaluate_single(gt_boxes,intersectionBoxes,0.5)

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots(1, 2, figsize=(16, 9))
    utils.draw_image_boxes(intersectionBoxes, ax2[0], img)
    utils.draw_image_boxes(gt_boxes, ax, img)
    #
    best_scores, best_boxes = engine.evaluate_single_best(gt_boxes, intersectionBoxes)
    #
    utils.draw_image_boxes(best_boxes, ax2[1], img)
    #
    plt.show()
    #

def main():
    # train on the GPU or on the CPU, if a GPU is not available

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # --------- Model/data initialization ---------
    # For use two classes only - background and rock
    num_classes = 2
    # use our dataset and defined transformations
    dataset = LunarDataset(lunar_loc, get_transform(train=True))
    dataset_test = LunarDataset(lunar_loc, get_transform(train=False))
    # print(len(dataset))
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.05,
                                momentum=0.9, weight_decay=0.05)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # ------------ Training ------------
    num_epochs = 1
    stats = {'epoch': [], 'precision': [], 'recall': []}
    for epoch in range(num_epochs):
        startT = time.time()
        print('start epoch ', epoch)
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        print('first training done')
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluation = engine.evaluate(model, data_loader_test,device)
        # Create stats dict
        stats['epoch'].append(epoch)
        stats['precision'].append(evaluation[0])
        stats['recall'].append(evaluation[1])
        endT = time.time()
        duration = endT-startT
        duration = utils.convertTime(duration)
        print('time: ' + duration )
    # Save model and stats
    torch.save(model, './models/model1000.pt')
    stats = pd.DataFrame(stats)
    with open('stats.txt', 'w') as csv_file:
        stats.to_csv(path_or_buf=csv_file)
    return eval

    print("That's it!")

# main()

model = torch.load('./models/model1000.pt')

# use our dataset and defined transformations
dataset = LunarDataset(lunar_loc, get_transform(train=True))
dataset_test = LunarDataset(lunar_loc, get_transform(train=False))
# print(len(dataset))
# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

detect_single_image(model,data_loader_test)