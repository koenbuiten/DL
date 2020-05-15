import os
import time
import sys

import engine
import numpy as np
import pandas as pd
import torch
from torch.optim.adagrad import Adagrad
from torch.optim.adadelta import Adadelta
from torch.optim.adam import Adam
import torchvision
import transforms as T
import utils
from PIL import Image
from engine import train_one_epoch
from evaluate import evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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
lunar_loc = '/data/s3861023/lunarDataset'

# Koen's local data location
# lunar_loc = '/media/koenbuiten/8675c03f-5bb1-4466-8581-8f042a79029b/koenbuiten/Datasets/artificial-lunar-rocky-landscape-dataset'


# This class load the dataset, this function is based on the pytorch tutorial on obejct detection: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# We change a few things line paths and made a seperate function in engine.py to extract masks and bounding boxes.
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
        maskArr = np.array(imgMask)

        # Obtain masks, boxes and labels from the mask image
        masks, boxes, labels = engine.label_objects(maskArr)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        num_objs = len(boxes)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "masks": masks, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# get_model_instance segmentation and get_transform are fully copied from the tutorial:
# same holds for get_transform
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

def get_mask_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Function to get different optimizers
def get_optimizer(params, settings):
    lrs = False
    if settings['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params, lr=settings['lr'],
                                    momentum=settings['momentum'],
                                    weight_decay=settings['wd'])
        lrs = True
    elif settings['optimizer'] == 'Adagrad':
        optimizer = Adagrad(params, lr=settings['lr'],
                            lr_decay=0,
                            weight_decay=settings['wd'],
                            initial_accumulator_value=0,
                            eps=1e-10)
    elif settings['optimizer'] == 'Adadelta':
        optimizer = Adadelta(params, lr=1.0,
                             rho=0.9,
                             eps=1e-06,
                             weight_decay=settings['wd'])
    elif settings['optimizer'] == 'Adam':
        optimizer = Adam(params,lr=settings['lr'],
                          betas=(0.9, 0.999),
                          eps=1e-08,
                          weight_decay=0,
                          amsgrad=False)
        lrs = True
    else:
        print('optimizer name invalid, using default SGD')
        optimizer = torch.optim.SGD(params, 0.005,
                                    momentum=0.9,
                                    weight_decay=0.0005)
    return optimizer, lrs

# The main program which trains and evaluates the model
# The input ds is True when we want to use an exisisting dataLoader, it is false when we want to create a new one
# From the command line the name of a text file with settings is read in
# The models checks if there is already a model for these settings, if so it will further train this model
def main(ds):
    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # --------- Model/data initialization ---------
    # Select numer of classes: Big rocks, small rocks and the rest
    num_classes = 3
    # If ds is True, use an exisiting dataloader, otherwise make a new one.
    if not ds:
        print('creating dataLoader')
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

        torch.save(data_loader, './dataLoaders/dataLoader_train2.pth')
        torch.save(data_loader_test, './dataLoaders/dataLoader_test2.pth')
    else:
        data_loader = torch.load('./dataLoaders/dataLoader_train2.pth')
        data_loader_test = torch.load('./dataLoaders/dataLoader_test2.pth')

    # get settings from the the command line argument
    settings = utils.get_settings(sys.argv[1])

    # Extract the model name
    modelName = sys.argv[1][0:len(sys.argv[1])-4]
    modelsList = list(sorted(os.listdir("./models")))
    print(modelName)

    # If the model already exist, load the model so further training can be done.
    # It is necessary to also use the same dataloader, otherwise there is a big change of overfitting
    if modelName + '.pt' in modelsList:
        print('Using existing model')
        statsFile = pd.read_csv("./stats/eval_" + modelName + '.csv')
        epochs = statsFile.get('epoch')
        epochs = np.max(epochs)+1
        model = torch.load("./models/" + modelName + '.pt',map_location='cpu')
    else:
        epochs = 0
        model = get_mask_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # Define parameters for the model and get optimzer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer, lrs = get_optimizer(params, settings)

    # and a learning rate scheduler
    if lrs:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

    # ------------ Training ------------
    num_epochs = 5
    # Threshold for non-mamxium suspression, when to predicted boxes have an IOU above the threshold,
    # the one with the lowest score is removed.
    nmsThreshold = 0.25
    for epoch in range(epochs, epochs + num_epochs):
        startT = time.time()
        print('start epoch ', epoch)
        # train for one epoch, printing every 50 iterations (batch training)
        train_one_epoch(model, optimizer, data_loader, device, epoch, 50, modelName)

        # update the learning rate, if the optimizer uses a learning rate schedular, make the step.
        if lrs:
            lr_scheduler.step()

        # evaluate on the test dataset
        evaluation = evaluate(model, data_loader_test, num_classes, nmsThreshold, device, epoch)
        evaluation = pd.DataFrame(evaluation)

        # Save statistics to an existing or a new file.
        if os.path.exists('./stats/eval_' + modelName + '.csv'):
            with open('./stats/eval_' + modelName + '.csv', 'a') as csv_file:
                evaluation.to_csv(path_or_buf=csv_file,header=False)
        else:
            with open('./stats/eval_' + modelName + '.csv', 'w') as csv_file:
                evaluation.to_csv(path_or_buf=csv_file)

        endT = time.time()
        duration = endT - startT
        duration = utils.convertTime(duration)
        print('time: ' + duration)

    # Save model
    torch.save(model, './models/' + modelName + '.pt')

    return

main(False)

