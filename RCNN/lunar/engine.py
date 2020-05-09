import math
import os
import sys

import numpy as np
import pandas as pd
import utils
from skimage import measure


def losses_to_file(filename, dict,epoch):
    for key in dict:
        dict[key] = [dict[key].detach()]
        
        # dict[key] = [dict[key].cpu()]
    # print(dict)
    dict = pd.DataFrame(dict,index=[epoch])
    headerVal = False

    with open(filename, 'a') as csv_file:
        if os.stat(filename).st_size == 0:
            headerVal = True
        dict.to_csv(path_or_buf=filename, mode='a',header= headerVal)

    # print(dict)
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, modelname):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # print(loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        # stats.append(loss_dict_reduced)
        loss_dict_reduced['loss'] = losses_reduced
        # loss_dict_reduced.numpy()


        # print(loss_dict_reduced)
        # print(losses_reduced)
        metric_logger.update(**loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        losses_to_file('./stats/loss_' + modelname + '.csv', loss_dict_reduced,epoch)

        # print(stats)



def intersection_over_union(gtBox, predBox):
    xMax = max(gtBox[0], predBox[0])
    yMax = max(gtBox[1], predBox[1])
    xMin = min(gtBox[2], predBox[2])
    yMin = min(gtBox[3], predBox[3])
    # print('new box')
    inter = [xMin, yMin, xMax, yMax]
    interArea = abs((xMax - xMin) * (yMax - yMin))
    # print(interArea)
    gtArea = (gtBox[2] - gtBox[0]) * (gtBox[3] - gtBox[1])
    # print(gtArea)
    predArea = (predBox[2] - predBox[0]) * (predBox[3] - predBox[1])
    # print(predArea)

    iou = interArea / ((gtArea + predArea) - interArea)

    return iou


def check_intersect(boxA, boxB):
    if (boxA[2] < boxB[0]) or (boxA[3] < boxB[1]) or (boxB[2] < boxA[0]) or (boxB[3] < boxA[1]):
        return False
    else:
        return True


def seperate_boxes(predBoxes, gtBoxes):
    noIntersect = []
    intersectPredBoxes = []
    for predBox in predBoxes:
        intersect = False
        for gtBox in gtBoxes:
            if not check_intersect(predBox, gtBox):
                continue
            else:
                intersectPredBoxes.append(predBox)
                intersect = True
                # print('break')
                break
        if intersect == False:
            noIntersect.append(predBox)

    return noIntersect, intersectPredBoxes


def connected_components_boxes(boxes, boxIdx,connectedBoxes):
    if boxIdx == len(boxes)-1:
        return connectedBoxes
    for boxJdx in range(boxIdx+1,len(boxes)):
        if check_intersect(boxes[boxIdx],boxes[boxJdx]):
            connectedBoxes.append(boxes[boxJdx])
            connected_components_boxes(boxes, boxJdx,connectedBoxes)
    return connectedBoxes


def label_objects(maskImg):
    # ground = (maskImg == [0,0,0]).all(-1)
    # air = (maskImg == [255, 0, 0]).all(-1)
    smallRocks = (maskImg == [0, 255, 0]).all(-1)
    bigRocks = (maskImg == [0, 0, 255]).all(-1)



    # objects = [ground,bigRocks, smallRocks,air]
    objects = [bigRocks, smallRocks]
    masks = []
    labels = []
    boxes = []

    # objLabels 1,2 and 3, because 0 is for the air

    for objClass in range(len(objects)):
        objectsInClass = measure.label(objects[objClass])

        # print(objectsInClass)
        objNums = len(np.unique(objectsInClass))
        for objIdx in range(1,objNums):
            object = np.where(objectsInClass == objIdx)
            mask = np.zeros((len(objectsInClass),len(objectsInClass[0])))
            mask[object[0],object[1]] = 1

            box = []

            box.append(min(object[1]))
            box.append(min(object[0]))
            box.append(max(object[1]))
            box.append(max(object[0]))

            masks.append(mask)

            boxes.append(box)
            labels.append(objClass+1)

    return masks, boxes, labels


def get_intersection(maskA, maskB):
    intersection = maskA + maskB
    # print((intersection == 2).sum())
    intersection[intersection != 2] = 0
    intersection[intersection == 2] = 1


    return intersection




def iou_masks(maskA, maskB, intersection):
    areaA = maskA.sum()
    areaB = maskB.sum()
    areaI = intersection.sum()

    iou = areaI / ((areaA+areaB)-areaI)

    return iou



