import sys

import numpy as np

# data_loc = '/media/koenbuiten/8675c03f-5bb1-4466-8581-8f042a79029b/koenbuiten/Datasets/artificial-lunar-rocky-landscape-dataset'
lunar_loc = '/data/s3861023/lunarDataset'


def createName(name):
    while len(name) != 4:
        name = '0' + name
    return name


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
        # print('no intersection')
        # print(boxA, boxB)
        return False
    else:
        return True


def iou_box(boxA, boxB):
    iou = 0
    if check_intersect(boxA, boxB):
        iou = intersection_over_union(boxA, boxB)
    return iou


# init boxId = 0 connectedBoxes = []

def get_mask_intersection(maskA, maskB):
    intersection = maskA + maskB
    # print((intersection == 2).sum())
    intersection[intersection != 2] = 0
    intersection[intersection == 2] = 1

    return intersection


def iou_mask(maskA, maskB):
    iou = 0
    intersection = get_mask_intersection(maskA, maskB)
    if intersection.sum() != 0:
        areaA = maskA.sum()
        areaB = maskB.sum()
        areaI = intersection.sum()

        iou = areaI / ((areaA + areaB) - areaI)

    return iou


def sort(arr1, arr2, sortTo):
    zipped = zip(arr1, arr2)
    sortedZip = sorted(zipped, key=lambda x: x[sortTo], reverse=True)
    # Unzip
    sortedArr = [[], []]
    for i in range(len(sortedZip)):
        sortedArr[0].append(sortedZip[i][0])
        sortedArr[1].append(sortedZip[i][1])
    # print(sortedArr)

    return sortedArr[0], sortedArr[1]


def non_maximum_suspression(arr, threshold, scores, boxMask):
    filteredArr = []
    filteredScores = []
    scoresCp = scores.copy()  # have to copy due to properties of pop
    if boxMask == 'box':
        while arr != []:
            box = arr.pop(0)
            filteredArr.append(box)
            filteredScores.append(scoresCp.pop(0))
            i = 0
            while i < len(arr) - 1:
                iouVal = iou_box(box, arr[i])
                if iouVal > threshold:
                    scoresCp.pop(i)
                    arr.pop(i)
                i = i + 1
    elif boxMask == 'mask':
        while arr != []:
            mask = arr.pop(0)
            filteredArr.append(mask)
            filteredScores.append(scoresCp.pop(0))
            i = 0
            while i < len(arr) - 1:
                iouVal = iou_mask(mask, arr[i])
                if iouVal > threshold:
                    scoresCp.pop(i)
                    arr.pop(i)
                i = i + 1
    else:
        sys.exit('Wrong method, must be box, or mask')

    return filteredArr, filteredScores


def maskThreshold(masks, threshold):
    newMasks = []
    for mask in masks:
        mask[mask > threshold] = 1
        mask[mask < threshold] = 0
        newMasks.append(mask)
    return newMasks


def calculate_ious(predSet, gtSet, boxMask):
    ious = []
    if boxMask == 'box':
        ious = {'iou': [], 'gtidx': []}
        for predIdx in range(len(predSet)):
            ious['iou'].append(0)
            ious['gtidx'].append(0)
            for gtIdx in range(len(gtSet)):
                if check_intersect(predSet[predIdx], gtSet[gtIdx]):
                    iou = intersection_over_union(predSet[predIdx], gtSet[gtIdx])
                    if ious['iou'][predIdx] < iou:
                        ious['iou'][predIdx] = iou
                        ious['gtidx'][predIdx] = (gtIdx)

    if boxMask == 'mask':
        ious = {'iou': [], 'gtidx': []}
        for predIdx in range(len(predSet)):
            ious['iou'].append(0)
            ious['gtidx'].append(0)
            for gtIdx in range(len(gtSet)):
                if get_mask_intersection(predSet[predIdx], gtSet[gtIdx]).sum() != 0:
                    iou = iou_mask(predSet[predIdx], gtSet[gtIdx])
                    if ious['iou'][predIdx] < iou:
                        ious['iou'][predIdx] = iou
                        ious['gtidx'][predIdx] = (gtIdx)

    for gtIdx in range(len(gtSet)):
        if np.count_nonzero(ious['gtidx'] == gtIdx) > 1:
            ious['iou'][ious['gtidx'] == gtIdx] = 0

    return ious['iou']


def average_precision(scores, ious, threshold, gts):
    scores, ious = sort(scores, ious, 0)
    # ranked = { "scores": scores, "ious": ious,"precision": [],"recall": []}
    precisions = []
    recalls = []

    tps = 0
    fps = 0
    # gts = int(10*(1-threshold))
    if gts == 0:
        return 0, 0

    for i in range(len(scores)):
        if ious[i] > threshold:
            tps = tps + 1
            precision = tps / (fps + tps)
            recall = tps / gts
        else:
            fps = fps + 1
            precision = tps / (fps + tps)
            recall = tps / gts
        precisions.append(precision)
        recalls.append(recall)
        # print("%.2f | %.2f" % (precision, recall))

    # epi = eleven_point_inter(precisions, recalls)
    precisions = smoothing(precisions)

    return precisions, recalls


def smoothing(arr):
    idx = len(arr) - 1
    while idx - 1 != 0:
        if arr[idx] >= arr[idx - 1]:
            arr[idx - 1] = arr[idx]
        idx = idx - 1
    return arr


def calculate_ap(scores, ious, method, gts):
    areas = []
    if method == 'AP{.50}':
        precisions, recalls = average_precision(scores, ious, 0.5, gts)
        ap = np.mean(precisions)
    elif method == 'AP{.75}':
        precisions, recalls = average_precision(scores, ious, 0.75, gts)
        ap = np.mean(precisions)
    elif method == 'AP{.50:.95:.05}':
        for iouThreshold in range(50, 95, 5):
            precisions, recalls = average_precision(scores, ious, iouThreshold / 100, gts)
            if precisions != 0:
                area = np.mean(precisions)
                # plt.show()
                areas.append(area)
        ap = np.mean(areas)
    else:
        print('not a valid method')
        return 0

    return ap


def evaluate(model, dataLoader, numclasses, nmsThreshold, device, epoch):
    model.eval()
    boxStats = []
    maskStats = []
    gtCount = 0
    for label in range(1, numclasses):
        boxStats.append({'ious': [], 'scores': []})
        maskStats.append({'ious': [], 'scores': []})
    i = 0
    for images, targets in dataLoader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        gtBoxes = targets[0]['boxes'].cpu().numpy()
        gtMasks = targets[0]['masks'].cpu().numpy()
        gtLabels = targets[0]['labels'].cpu().numpy()

        gtCount = gtCount + len(gtBoxes)
        # print(images)
        # image_id = targets[0]['image_id'].cpu()
        # image_id = image_id.numpy()+1
        # image_path = os.path.join(data_loc, "images/render2/", 'render' + createName(str(image_id[0])) + '.png')
        # image = Image.open(image_path)
        # ax.imshow(images[0][0])

        output = model(images)
        # print("output ready")
        predBoxes = output[0]['boxes'].detach().cpu().numpy()
        predMasks = output[0]['masks'].detach().cpu().numpy()
        predLabels = output[0]['labels'].detach().cpu().numpy()
        scores = output[0]['scores'].detach().cpu().numpy()

        labels = np.unique(gtLabels)
        for label in labels:
            scoreSet = scores[np.where(predLabels == label)]
            scoreSet = list(scoreSet)
            gtBoxset = gtBoxes[np.where(gtLabels == label)]
            gtMaskset = gtMasks[np.where(gtLabels == label)]

            predBoxset = predBoxes[np.where(predLabels == label)]

            # fig, ax = plt.subplots(1,2,figsize=(16,9))
            # utils.draw_image_boxes(gtBoxset, ax[0], image)
            # utils.draw_masks(gtMaskset,ax[0])
            predBoxset = predBoxset
            predBoxset = list(predBoxset)  # convert to numpy to use pop
            predBoxset, boxScores = non_maximum_suspression(predBoxset, nmsThreshold, scoreSet, 'box')

            predMaskset = predMasks[np.where(predLabels == label)]
            predMaskset = list(predMaskset)
            predMaskset = maskThreshold(predMaskset, 0.8)

            # utils.draw_image_boxes(predBoxset,ax[1],image)

            predMaskset, maskScores = non_maximum_suspression(predMaskset, nmsThreshold, scoreSet, 'mask')

            iouBoxSet = calculate_ious(predBoxset, gtBoxset, 'box')
            iouMaskSet = calculate_ious(predMaskset, gtMaskset, 'mask')
            boxStats[label - 1]['ious'].extend(iouBoxSet)
            boxStats[label - 1]['scores'].extend(boxScores)
            maskStats[label - 1]['ious'].extend(iouMaskSet)
            maskStats[label - 1]['scores'].extend(maskScores)

    eval = {'epoch': [], 'boundary': [], 'class': [], 'AP{.50:.95:.05}': [], 'AP{.50}': [], 'AP{.75}': []}
    methods = ['AP{.50:.95:.05}', 'AP{.50}', 'AP{.75}']
    for labelIdx in range(numclasses - 1):
        eval['epoch'].append(epoch)
        eval['epoch'].append(epoch)
        eval['boundary'].append('box')
        eval['class'].append(labelIdx + 1)
        eval['boundary'].append('mask')
        eval['class'].append(labelIdx + 1)
        for method in methods:
            apBox = calculate_ap(boxStats[labelIdx]['scores'], boxStats[labelIdx]['ious'], method, gtCount)
            apMask = calculate_ap(maskStats[labelIdx]['scores'], maskStats[labelIdx]['ious'], method, gtCount)
            eval[method].append(apBox)
            eval[method].append(apMask)

    return eval
