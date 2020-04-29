import math
import sys

import numpy as np

import utils
from skimage import measure

import matplotlib.pyplot as plt

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
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

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


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


def evaluate_single_best(gtBoxes, boxes_out):
    best_scores = []
    best_boxes = []
    i = 0
    for gtBox in gtBoxes:
        best_iou = 0
        best_box = []
        for predBox in boxes_out:
            if check_intersect(gtBox, predBox):
                iou_tmp = intersection_over_union(gtBox, predBox)
                if (iou_tmp > best_iou):
                    best_iou = iou_tmp
                    best_box = predBox
        if best_iou != 0:
            best_scores.append(best_iou)
            best_boxes.append(best_box)
        i += 1

    return best_scores, best_boxes


def evaluate_single(gtBoxes, predBoxes, threshold):
    scores = []
    # print(gtBoxes[0][0])
    for predIdx in range(len(predBoxes)):
        best_iou = [0, 0, 0]
        for gtIdx in range(len(gtBoxes)):
            if check_intersect(gtBoxes[gtIdx], predBoxes[predIdx]):
                iou_tmp = intersection_over_union(gtBoxes[gtIdx], predBoxes[predIdx])
                scores.append([iou_tmp, predBoxes, gtIdx])
                if (iou_tmp > best_iou[0]):
                    best_iou = [iou_tmp, predIdx, gtIdx]

    predLen = len(predBoxes)

    if scores != []:

        tp = 0
        fp = 0
        fn = 0
        for score in scores:
            if score[0] > threshold:
                tp = tp + 1
            else:
                fp = fp + 1

        for i in range(len(gtBoxes)):
            # i = torch.tensor([i])

            # for cpu use below
            # if not isinstance(score[0],int):
            #     score[0] = score[0].detach()

            detect = False
            for score in scores:
                if score[2] >i:
                    if score[0] > threshold:
                        detect = True
                        break
            if detect == False:
                fn = fn + 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        recall2 = tp / predLen
    # print ('precision: ' + str(precision))
    # print('recall: ' + str(recall))

        return precision, recall, recall2
    return 0,0

def connected_components_boxes(boxes, boxIdx,connectedBoxes):
    if boxIdx == len(boxes)-1:
        return connectedBoxes
    for boxJdx in range(boxIdx+1,len(boxes)):
        if check_intersect(boxes[boxIdx],boxes[boxJdx]):
            connectedBoxes.append(boxes[boxJdx])
            connected_components_boxes(boxes, boxJdx,connectedBoxes)
    return connectedBoxes


def evaluate(model, dataLoader, threshold, device):
    model.eval()
    precision = []
    recall = []
    # print(len(dataLoader))
    i = 0
    for images, targets in dataLoader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # image.cuda()
        # image = image[0].to(device=device)
        # print(image)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        gtBoxes = targets[0]['boxes']
        # print(gtBoxes)
        output = model(images)
        # print(predBoxes)
        predBoxes = output[0]['boxes']
        # print(predBoxes)
        prec, recc = evaluate_single(gtBoxes, predBoxes, threshold)
        precision.append(prec)
        recall.append(recc)
        # print("precision: " + str(prec))
        # print("recall: " + str(recc)+"\n")

        if i == 2:
            break;
        i = i+1
    meanPrec = np.mean(precision)
    meanRecc = np.mean(recall)

    print("Mean precision: " + str(meanPrec))
    print("Mean recall: " + str(meanRecc))

    return meanPrec, meanRecc

def label_objects(maskImg, idx):
    # ground = (maskImg == [0,0,0]).all(-1)
    # air = (maskImg == [255, 0, 0]).all(-1)
    smallRocks = (maskImg == [0, 255, 0]).all(-1)
    bigRocks = (maskImg == [0, 0, 255]).all(-1)
    # print(bigRocks)

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
            objectsInClass = measure.label(objects[objClass])

            # First object is everything else
            object = np.where(objectsInClass == objIdx)
            # print(object)
            box = []

            box.append(min(object[1]))
            box.append(min(object[0]))
            box.append(max(object[1]))
            box.append(max(object[0]))
            mask = objectsInClass


            mask[mask == objIdx] = 1
            mask[mask != objIdx] = 0

            masks.append(mask)

            boxes.append(box)
            labels.append(objClass+1)

    # if len(boxes) == 0:
    #     print('no boxes: ' + str(idx))
        # masks = [[0,0]]
        # boxes = [[0, 0, 0, 0]]
        # labels = [0]
    return masks, boxes, labels



