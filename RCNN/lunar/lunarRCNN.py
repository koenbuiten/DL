import os
import numpy as np
import torch
import pandas as pd
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import engine
import transforms as T

import time
import utils
from engine import train_one_epoch
import matplotlib.pyplot as plt

# --- Goal ----
# Detecting rocks in images from the lunar moon lander.

# ---- Dataset ----
# The following dataset from Kaggle is used:
# https://www.kaggle.com/romainpessia/artificial-lunar-rocky-landscape-dataset
# The dataset consist of render moon images, mask images with big rocks, small rocks,
# air and ground for each rendered image. There is a clean and a normal version of the
# masks and images, the clean version consist of the mask with only rock of a certain size,
# these rocks can be considered 'useful'. We made a new clean version for the masks and images,
# in this new version, all the images which do not have any useful rocks are discarded.

# ---- Model ----
# The program uses a pre trained fast RCNN model from pytorch.


# Koen's peregrine data location
# lunar_loc = '/data/s3861023/lunarDataset'

# Koen's local data location
lunar_loc = '/media/koenbuiten/8675c03f-5bb1-4466-8581-8f042a79029b/koenbuiten/Datasets/artificial-lunar-rocky-landscape-dataset'


# Make sure image name has 4 integers
def createName(name):
    while len(name) != 4:
        name = '0' + name
    return nameI


# Create and save a new dataset with only 'useful' rocks.
# Folder structure:
# Parent: artificial-lunar-rocky-landscape-dataset
# First children: images, real_moon_images, bounding_boxes.csv
# images children: clean, clean2, ground, render, render2
def create_only_rocks_dataset(data_loc):
    img_list = list(sorted(os.listdir(os.path.join(data_loc, "images/render"))))
    mask_list = list(sorted(os.listdir(os.path.join(data_loc, "images/clean"))))
    imgs = []
    masks = []

    jdx = 1
    for idx in range(len(mask_list)):
        mask_path = os.path.join(data_loc, "images/clean", mask_list[idx])
        imgMask = Image.open(mask_path)
        mask = np.array(imgMask)

        if (True in ((mask == [0, 255, 0]).all(-1))) or (True in (mask == [0, 0, 255]).all(-1)):
            imageName = createName(str(jdx))
            imgMask.save(os.path.join(data_loc, "./images/clean2/clean" + imageName + ".png"))
            masks.append(mask_list[idx])

            img_path = os.path.join(data_loc, "images/render", img_list[idx])
            image = Image.open(img_path)
            image.save(os.path.join(data_loc, "images/render2/render" + imageName + ".png"))
            imgs.append(img_list[idx])

            jdx = jdx + 1

# Loads in dataset
class LunarDataset(object):
    def __init__(self, data_loc, transforms):
        self.data_loc = data_loc
        self.transforms = transforms

        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(data_loc, "images/render2"))))
        self.masks = list(sorted(os.listdir(os.path.join(data_loc, "images/clean2"))))

    #
    def __getitem__(self, idx):
        # load images and boxes
        img_path = os.path.join(self.data_loc, "images/render2", self.imgs[idx])
        mask_path = os.path.join(self.data_loc, "images/clean2", self.masks[idx])

        # Open image
        img = Image.open(img_path)

        # Open the mask image and convert to numpy array
        imgMask = Image.open(mask_path)
        imgMask = np.array(imgMask)

        # Obtain masks, boxes and labels from the mask image
        masks, boxes, labels = engine.label_objects(imgMask, idx)

        # convert everything into a torch.Tensor

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# get_model_instance segmentation and get_transform are fully copied from the tutorial:
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
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
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Function which put an image from the test set into model.
# The output consist of the ground truth image, the boxes found by the model
# and the best matching box(if any) for every ground truth box.
def detect_single_image(model, imgId):
    # Set model to evaluation mode
    model.eval()

    # Load in test dataset and get image and target(ground truth)
    dataset = LunarDataset(lunar_loc, get_transform(train=False))
    img, target = dataset.__getitem__(imgId-1)
    img = [img]

    # Put image into model
    output = model(img)


    predBoxes = output[0]['boxes'].detach()
    gt_boxes = target['boxes']

    # Seperate the boxes which do not intersect with any of the ground truth boxes
    # from the ones which do intersect
    no_intersection, intersectionBoxes = engine.seperate_boxes(predBoxes, gt_boxes)
    best_scores, best_boxes = engine.evaluate_single_best(gt_boxes, intersectionBoxes)
    precision, recall = engine.evaluate_single(gt_boxes, predBoxes, 0.5)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))

    # Show image with ground truth boxes, predicted boxes and best boxes.
    imgId = createName(str(imgId))
    img_path = os.path.join(lunar_loc, "images/render2", 'render' + imgId + '.png')
    img = Image.open(img_path).convert('RGB')

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots(1, 2, figsize=(16, 9))
    utils.draw_image_boxes(gt_boxes, ax[0], img)
    utils.draw_image_boxes(intersectionBoxes, ax2[1], img)
    utils.draw_image_boxes(best_boxes, ax2[2], img)

    plt.show()

def main(evaluate):
    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # --------- Model/data initialization ---------
    # Select numer of classes: Big rock, small rocks and the rest
    num_classes = 3
    # Use dataset and defined transformations
    dataset = LunarDataset(lunar_loc, get_transform(train=True))
    dataset_test = LunarDataset(lunar_loc, get_transform(train=False))

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
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # ------------ Training ------------
    num_epochs = 1
    # Threshold for evalution with iou, an iou above the thresholds defines a good detection.
    threshold = 0.5
    stats = {'epoch': [], 'precision': [], 'recall': []}
    for epoch in range(num_epochs):
        startT = time.time()
        print('start epoch ', epoch)
        # train for one epoch, printing every 50 iterations (batch training)
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        print('first training done')
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # I created the evaluation function myself to be more in controll.
        # Planning on comparing it with the coco evaluation from the tutorial
        if evaluate:
            evaluation = engine.evaluate(model, data_loader_test, threshold, device)
            # create stats dict
            stats['epoch'].append(epoch)
            stats['precision'].append(evaluation[0])
            stats['recall'].append(evaluation[1])
        endT = time.time()
        duration = endT - startT
        duration = utils.convertTime(duration)
        print('time: ' + duration)
    # Save model and stats
    torch.save(model, './models/modelAll_10.pt')
    stats = pd.DataFrame(stats)
    with open('stats.csv', 'w') as csv_file:
        stats.to_csv(path_or_buf=csv_file)
    return eval

    print("That's it!")


main(False)
