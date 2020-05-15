import sys
import numpy as np

# Calculate the intersection over uninion between two bounding boxes.
# First calculate the maximum and minium x and y values of the intersection
# Then calcuate the area of the intersection and boxes.
# Calculate the iou with these areas.
def intersection_over_union(gtBox, predBox):
    xMax = max(gtBox[0], predBox[0])
    yMax = max(gtBox[1], predBox[1])
    xMin = min(gtBox[2], predBox[2])
    yMin = min(gtBox[3], predBox[3])

    interArea = abs((xMax - xMin) * (yMax - yMin))
    gtArea = (gtBox[2] - gtBox[0]) * (gtBox[3] - gtBox[1])
    predArea = (predBox[2] - predBox[0]) * (predBox[3] - predBox[1])

    iou = interArea / ((gtArea + predArea) - interArea)

    return iou

# Check if two bounding boxes intersect, they do when no maximum value of one box is smaller than the minium of the other box.
def check_intersect(boxA, boxB):
    if (boxA[2] < boxB[0]) or (boxA[3] < boxB[1]) or (boxB[2] < boxA[0]) or (boxB[3] < boxA[1]):
        return False
    else:
        return True

# Calculate the intersection over uninion between two bounding boxes, first check if the boxes intersect.
def iou_box(boxA, boxB):
    iou = 0
    if check_intersect(boxA, boxB):
        iou = intersection_over_union(boxA, boxB)
    return iou

# Check if two masks intersect, they do when there are pixels values of 2 in the sum of the masks.
def get_mask_intersection(maskA, maskB):
    intersection = maskA + maskB
    # print((intersection == 2).sum())
    intersection[intersection != 2] = 0
    intersection[intersection == 2] = 1

    return intersection

# Calculate the intersection of union of two masks, first check if they intersect.
# If they do not, return an iou of zero.
# When they do, calcuate the areas of the masksa and intersection.
# Calculate the iou with these areas.
def iou_mask(maskA, maskB):
    iou = 0
    intersection = get_mask_intersection(maskA, maskB)
    if intersection.sum() != 0:
        areaA = maskA.sum()
        areaB = maskB.sum()
        areaI = intersection.sum()

        iou = areaI / ((areaA + areaB) - areaI)

    return iou

# Sorts to array in decending order, the sortTo variable refers to the array to which the arrays have to be sorted.
def sort(arr1, arr2, sortTo):
    zipped = zip(arr1, arr2)
    sortedZip = sorted(zipped, key=lambda x: x[sortTo], reverse=True)
    sortedArr = [[], []]
    for i in range(len(sortedZip)):
        sortedArr[0].append(sortedZip[i][0])
        sortedArr[1].append(sortedZip[i][1])

    return sortedArr[0], sortedArr[1]

# Non maximum suspression removes boxes and masks which overlaps, and chooses the one with the best score.
# The input should be sorted in ascending order of the scores.
# The function first pops the box/mask with the highest score and put it in the filtered boxes/masks.
# It calculates the iou between this box/mask  with all the other boxes/masks.
# If the iou is above the set threshold, the box/mask is removed from the array and thus discarded.
# It continous to do this untile the array is empty. Which leaves the filter array as the new boxes/masks
def non_maximum_suspression(arr, threshold, scores, boxMask):
    filteredArr = []
    filteredScores = []
    scoresCp = scores.copy()  # have to copy due to properties of pop

    if boxMask == 'box':
        while arr != []:
            box = arr.pop(0)
            filteredArr.append(box)
            score = scoresCp.pop(0)
            filteredScores.append(score)
            i = 0
            while i < len(arr):
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
            while i < len(arr):
                iouVal = iou_mask(mask, arr[i])
                # print(iouVal)
                if iouVal > threshold:
                    scoresCp.pop(i)
                    arr.pop(i)
                i = i + 1
    else:
        sys.exit('Wrong method, must be box, or mask')

    return filteredArr, filteredScores

# Masks which come out of the model are boxes, where pixels further outward have a lower value.
# To obtain the shape of the object, at threshold is set for which pixels denote the object.
def maskThreshold(masks, threshold):
    newMasks = []
    for mask in masks:
        mask[mask > threshold] = 1
        mask[mask < threshold] = 0
        newMasks.append(mask)
    return newMasks

# Function which loops over the ground trurh and predicted bounding boxes or masks.
# For every prediction, the ground truth is found and the iou is calculated.
# If a prediction has no ground truth, the idx of the ground truth corresponding to the prediction is set to zero
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

# Calculates the average precision gives scores, iou, threshold and the amount of ground truth object
def average_precision(scores, ious, threshold, gts):
    scores, ious = sort(scores, ious, 0)
    precisions = []
    recalls = []

    tps = 0
    fps = 0

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

    precisions, recalls = smoothing(precisions,recalls)

    return precisions, recalls

# Smoothing/interpolation of the precision
def smoothing(arr,arr2):

    if len(arr) > 1:
        idx = len(arr) - 1
        while idx - 1 != 0:
            if arr[idx] >= arr[idx - 1]:
                arr[idx - 1] = arr[idx]
                arr2[idx - 1] = arr2[idx]
            idx = idx - 1

    return arr,arr2

# Calculates the ap according to the iou method given
def calculate_ap(scores, ious, method, gts):
    if method == 'AP{.50}':
        precisions, recalls = average_precision(scores, ious, 0.5, gts)
        ap = np.mean(precisions)
        ar = np.mean(recalls)
    elif method == 'AP{.75}':
        precisions, recalls = average_precision(scores, ious, 0.75, gts)
        ap = np.mean(precisions)
        ar = np.mean(recalls)
    elif method == 'AP{.50:.95:.05}':
        multiAps = []
        multiArs = []
        for iouThreshold in range(50, 95, 5):
            precisions, recalls = average_precision(scores, ious, iouThreshold / 100, gts)
            if precisions != 0:
                ap = np.mean(precisions)
                ar = np.mean(recalls)
                # plt.show()
                multiAps.append(ap)
                multiArs.append(ar)
        ap = np.mean(multiAps)
        ar = np.mean(multiArs)
    else:
        print('not a valid method')
        return 0

    return ap

# loops over the images from the test dataset and calculates the ap and ar for the masks and box method seperatly
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

        output = model(images)

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

            predBoxset = predBoxset
            predBoxset = list(predBoxset)  # convert to numpy to use pop
            predBoxset, boxScores = non_maximum_suspression(predBoxset, nmsThreshold, scoreSet, 'box')
            predBoxset, boxScores = non_maximum_suspression(predBoxset, nmsThreshold, boxScores, 'box')

            predMaskset = predMasks[np.where(predLabels == label)]
            predMaskset = list(predMaskset)
            predMaskset = maskThreshold(predMaskset, 0.8)

            predMaskset, maskScores = non_maximum_suspression(predMaskset, nmsThreshold, scoreSet, 'mask')
            predMaskset, maskScores = non_maximum_suspression(predMaskset, nmsThreshold, maskScores, 'mask')

            iouBoxSet = calculate_ious(predBoxset, gtBoxset, 'box')
            iouMaskSet = calculate_ious(predMaskset, gtMaskset, 'mask')
            boxStats[label - 1]['ious'].extend(iouBoxSet)
            boxStats[label - 1]['scores'].extend(boxScores)
            maskStats[label - 1]['ious'].extend(iouMaskSet)
            maskStats[label - 1]['scores'].extend(maskScores)

    eval = {'epoch': [], 'boundary': [], 'class': [],
            'AP{.50:.95:.05}': [], 'AP{.50}': [], 'AP{.75}': []}
    methods = ['AP{.50:.95:.05}', 'AP{.50}', 'AP{.75}']
    for labelIdx in range(numclasses - 1):
        eval['epoch'].append(epoch)
        eval['epoch'].append(epoch)
        eval['boundary'].append('box')
        eval['class'].append(labelIdx + 1)
        eval['boundary'].append('mask')
        eval['class'].append(labelIdx + 1)
        for method in methods:
            apBox  = calculate_ap(boxStats[labelIdx]['scores'], boxStats[labelIdx]['ious'], method, gtCount)
            apMask = calculate_ap(maskStats[labelIdx]['scores'], maskStats[labelIdx]['ious'], method, gtCount)
            eval[method].append(apBox)
            eval[method].append(apMask)

    return eval
