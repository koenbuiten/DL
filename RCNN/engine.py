import math
import os
import sys
import cv2

import numpy as np
import pandas as pd
import utils
from skimage import measure

#------ Engine.py -------
# The engine of the program has two main function for the training of the model.

# Train_one_epoch:
# This function trains the model for one epoch, this means it loops over the train dataset,
# puts batched image in the model, calculates the loss en makes a learning step.
# The function is a slightly changed version of the one from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
# For logging to a csv file the modelname variable is added, therefore we also don't return the metric logger.
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, modelname):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        print('warm up')
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        print('warm up done')

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_dict_reduced['loss'] = losses_reduced

        metric_logger.update(**loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        utils.losses_to_file('./stats/loss_' + modelname + '.csv', loss_dict_reduced,epoch)

# label_objects:
# Label object is the function which makes the preprocessing step for the data.
# A maskImg is inserted, from this mask images, the different kind of rocks(big and small) are extracted.
# For each of these rocks, the bounding box is calculated.
# The boxes, masks and labels are returned by the function
# This funciton is based on the kernel from kaggle: https://www.kaggle.com/romainpessia/understanding-and-using-the-dataset-wip
def label_objects(maskImg):
    # Red pixels are air, green pixels are small rocks and blue pixels are big rock
    bigRocks = maskImg[:, :, 2]
    smallRocks = maskImg[:,:,1]

    masks = []
    labels = []
    boxes = []

    # For every object class, make the pixel 1 if it is the class color, otherwise make it 0
    h, w = smallRocks.shape
    for y in range(0, h):
        for x in range(0, w):
            smallRocks[y, x] = 1 if smallRocks[y, x] != 0 else 0

    h, w = bigRocks.shape
    for y in range(0, h):
        for x in range(0, w):
            bigRocks[y, x] = 1 if bigRocks[y, x] != 0 else 0

    # Code from the above mentioned kernel which remove small rocks in big rocks.
    # This is redundent when using the clean version of the dataset.
    kernel = np.ones((15, 15), np.uint8)
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    smallRocks = cv2.morphologyEx(smallRocks, cv2.MORPH_OPEN, kernel_circle)

    objects = [bigRocks, smallRocks, air]

    # Loop over the 3 object classes retrieve objects in a class with measure.label
    # Measure.label creates connected components and give all the pixels in a connected components a different integer
    # The about of unique integers is the amount of objects for the object class.
    for objClass in range(0,len(objects)):
        objectsInClass = measure.label(objects[objClass])
        objNums = len(np.unique(objectsInClass))

        # Loop over all the objects in a class, set true or false when for the pixels in the mask image which belong to the object.
        # Create a new mask where all the pixels are zero, excepte for the pixels where a singel object is located.
        # Extract the bounding box by taking the maximum and mimimum of the x and y.
        # Check if it is actually a box an not a line, if so, put the mask, box and corresponding class label in the array.
        for objIdx in range(1,objNums):
            object = np.where(objectsInClass == objIdx)
            mask = np.zeros((len(objectsInClass),len(objectsInClass[0])))
            mask[object[0],object[1]] = 1

            box = []

            box.append(min(object[1]))
            box.append(min(object[0]))
            box.append(max(object[1]))
            box.append(max(object[0]))
            if not ((box[0] == box[2]) or (box[1] == box[3])):
                masks.append(mask)
                boxes.append(box)
                labels.append(objClass+1)
    return masks, boxes, labels
