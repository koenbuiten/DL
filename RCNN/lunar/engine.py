import math
import sys
import torch

import numpy as np

import utils


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
                if (iou_tmp > best_iou[0]):
                    best_iou = [iou_tmp, predIdx, gtIdx]
        scores.append(best_iou)

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
    # print ('precision: ' + str(precision))
    # print('recall: ' + str(recall))

        return precision, recall
    return 0,0  


# def evaluate(model, dataLoader, threshold,device):
#     model.eval()
#     precision = []
#     recall = []
#     print(len(dataLoader))
#     i = 0
#     for image, target in dataLoader:
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         gtBoxes = target[0]['boxes'].numpy()
#         # print(gtBoxes)
#         predBoxes = model(image)
#         predBoxes = predBoxes[0]['boxes'].detach().numpy()
#         prec, recc = evaluate_single(gtBoxes, predBoxes, threshold)
#         precision.append(prec)
#         recall.append(recc)
#         # print("precision: " + str(prec))
#         # print("recall: " + str(recc)+"\n")
#
#         if i == 2:
#             break;
#         i = i+1
#     meanPrec = np.mean(precision)
#     meanRecc = np.mean(recall)
#
#     print("Mean precision: " + str(meanPrec))
#     print("Mean recall: " + str(meanRecc))
#
#     return meanPrec, meanRecc

def evaluate(model, dataLoader, threshold, device):
    model.eval()
    precision = []
    recall = []
    print(len(dataLoader))
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