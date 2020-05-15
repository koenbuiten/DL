import torch
import numpy as np
import os
from PIL import Image
import engine
import evaluate
import utils
from matplotlib import pyplot as plt
import time

# Koen's peregrine data location
# lunar_loc = '/data/s3861023/lunarDataset'

# Koen's local data location
lunar_loc = '/media/koenbuiten/8675c03f-5bb1-4466-8581-8f042a79029b/koenbuiten/Datasets/artificial-lunar-rocky-landscape-dataset'

# The dataset class has to be in the file for the loading of the dataloaders to work
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

# Function makes a figure with the dected masks compared to the ground truth
def showMaskDetection(maskAp,maskAr, idx, type):
    figure, ax = plt.subplots(1, 2, figsize=(12, 6))
    imgId = maskAp[idx]['image_id'][0][0] + 1

    imgName = 'render' + utils.createName(str(imgId)) + '.png'
    print('mask, ', imgName)
    img_path = os.path.join(lunar_loc, "images/render2", imgName)
    img = Image.open(img_path)
    ax[0].imshow(img)
    ax[1].imshow(img)
    utils.draw_masks(maskAp[idx]['gtMasks'], ax[0], 'g')
    utils.draw_masks(maskAp[idx]['predMasks'], ax[1], 'g')

    if type == 'double':
        ap = (maskAp[idx]['AP{.50:.95:.05}'] + maskAp[idx - 1]['AP{.50:.95:.05}']) / 2
        ar = (maskAr[idx]['AP{.50:.95:.05}'] + maskAr[idx - 1]['AP{.50:.95:.05}']) / 2
        utils.draw_masks(maskAp[idx - 1]['gtMasks'], ax[0], 'b')
        utils.draw_masks(maskAp[idx - 1]['predMasks'], ax[1], 'b')
    else:
        ap = maskAp[idx]['AP{.50:.95:.05}']
        ar = maskAr[idx]['AP{.50:.95:.05}']

    # print(maskAp[idx]['gtMasks'])

    ax[0].set_title('Grounth truth', fontsize=12)
    ax[1].set_title('Predicted', fontsize=12)
    figure.suptitle('Rock detection\n' + 'AP{.50:.95:.05}:' + str(ap) + '\nAR{.50:.95:.05}: ' + str(ar), fontsize=18)
    plt.tight_layout(w_pad=3, h_pad=3)
    figure.savefig('example_outputs/mask_' + 'Rock_detection_' + 'AP{.50:.95:.05}:' + str(ap) + '.jpg')

# Function makes a figure with the dectected boxes compared to the ground truth
def showBoxDetection(boxAp, boxAr, idx, type):
    # print(boxAp[idx]['gtboxs'])

    figure, ax = plt.subplots(1, 2, figsize=(12, 6))
    imgId = boxAp[idx]['image_id'][0][0] + 1

    imgName = 'render' + utils.createName(str(imgId)) + '.png'
    print(imgName)
    img_path = os.path.join(lunar_loc, "images/render2", imgName)
    img = Image.open(img_path)

    ax[0].imshow(img)
    ax[1].imshow(img)

    utils.draw_boxes(boxAp[idx]['gtBoxes'], ax[0], 'g')
    utils.draw_boxes(boxAp[idx]['predBoxes'], ax[1], 'g')
    ax[0].set_title('Grounth truth', fontsize=12)
    ax[1].set_title('Predicted', fontsize=12)
    if type == 'double':
        ap = (boxAp[idx]['AP{.50:.95:.05}'] + boxAp[idx - 1]['AP{.50:.95:.05}']) / 2
        ar = (boxAr[idx]['AP{.50:.95:.05}'] + boxAr[idx - 1]['AP{.50:.95:.05}']) / 2
        utils.draw_boxes(boxAp[idx - 1]['gtBoxes'], ax[0], 'b')
        utils.draw_boxes(boxAp[idx - 1]['predBoxes'], ax[1], 'b')
    else:
        ap = boxAp[idx]['AP{.50:.95:.05}']
        ar = boxAr[idx]['AP{.50:.95:.05}']

    figure.suptitle('Rock detection\n' + 'AP{.50:.95:.05}:' + str(ap) + '\nAR{.50:.95:.05}: ' + str(ar), fontsize=18)
    plt.tight_layout(w_pad=3, h_pad=3)
    figure.savefig('example_outputs/box_' + 'Rock_detection_' + 'AP{.50:.95:.05}:' + str(ap) + '.jpg')

# Main function which loads the dataloader, the model and run the evaluate per image
# It loops over the detection methods (masks/boxes) and makes figures from the show functions.
def main():
    data_loader = torch.load('./dataLoaders/dataLoader_train.pth')
    data_loader_test = torch.load('./dataLoaders/dataLoader_test.pth')
    modelName = 'SGD_0.01_0.9_0.0001_old'

    device = torch.device('cpu')
    model = torch.load("./models/old/" + modelName + '.pt', map_location='cpu')
    maskAp, boxAp, maskAr,boxAr = evaluate_per_image(model,data_loader_test,0.05,device)

    idx = len(maskAp)-1
    while idx > 0:
        if (maskAp[idx]['image_id'] == maskAp[idx-1]['image_id']):
            showMaskDetection(maskAp, maskAr, idx,'double')
            idx = idx - 2
        else:
            showMaskDetection(maskAp, maskAr, idx,'single')
            idx = idx -1

    idx = len(boxAp) - 1
    while idx > 0:
        if (boxAp[idx]['image_id'] == boxAp[idx - 1]['image_id']):
            showBoxDetection(boxAp, boxAr, idx,'double')
            idx = idx - 2
        else:
            showBoxDetection(boxAp, boxAr, idx, 'single')
            idx = idx - 1

# The function calculated the AP and AR for every separate image and for boxes and masks separate
def evaluate_per_image(model, dataLoader, nmsThreshold, device):
    # Set model to eval mode
    model.eval()
    durations = []
    maskAp = []
    boxAp = []
    maskAr = []
    boxAr = []
    imageIds = []

    for image, target in dataLoader:
        image = list(img.to(device) for img in image)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]

        gtBoxes = target[0]['boxes'].cpu().numpy()
        gtMasks = target[0]['masks'].cpu().numpy()
        gtLabels = target[0]['labels'].cpu().numpy()
        imgId = target[0]['image_id'].cpu().numpy()
        imageIds.append(imgId[0])

        startT = time.time()
        output = model(image)
        endT = time.time()
        duration = endT - startT
        durations.append(duration)
        totaltime = utils.convertTime(duration)
        print("time: ", totaltime)
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
            gtCount = len(gtBoxset)
            predBoxset = predBoxes[np.where(predLabels == label)]

            predBoxset = predBoxset
            predBoxset = list(predBoxset)  # convert to numpy to use pop

            predBoxset, boxScores = evaluate.non_maximum_suspression(predBoxset, nmsThreshold, scoreSet, 'box')
            predBoxset, boxScores = evaluate.non_maximum_suspression(predBoxset, nmsThreshold, boxScores, 'box')

            predMaskset = predMasks[np.where(predLabels == label)]
            predMaskset = list(predMaskset)
            predMaskset = evaluate.maskThreshold(predMaskset, 0.8)

            predMaskset, maskScores = evaluate.non_maximum_suspression(predMaskset, nmsThreshold, scoreSet, 'mask')
            predMaskset, maskScores = evaluate.non_maximum_suspression(predMaskset, nmsThreshold, maskScores, 'mask')

            iouBoxSet = evaluate.calculate_ious(predBoxset, gtBoxset, 'box')
            iouMaskSet = evaluate.calculate_ious(predMaskset, gtMaskset, 'mask')
            methods = ['AP{.50:.95:.05}', 'AP{.50}', 'AP{.75}']
            maskAp.append({'image_id': [imgId], 'class': [label], 'AP{.50:.95:.05}': [], 'AP{.50}': [], 'AP{.75}': [], 'predMasks': predMaskset, 'gtMasks': gtMaskset})
            boxAp.append({'image_id': [imgId], 'class': [label], 'AP{.50:.95:.05}': [], 'AP{.50}': [], 'AP{.75}': [], 'predBoxes': predBoxset, 'gtBoxes': gtBoxset})
            maskAr.append({'image_id': [imgId], 'class': [label], 'AR{.50:.95:.05}': [], 'AR{.50}': [], 'AR{.75}': [], 'predMasks': predMaskset, 'gtMasks': gtMaskset})
            boxAr.append({'image_id': [imgId], 'class': [label], 'AR{.50:.95:.05}': [], 'AR{.50}': [], 'AR{.75}': [], 'predBoxes': predBoxset, 'gtBoxes': gtBoxset})

            for method in methods:
                apBox,arBox = evaluate.calculate_ap(boxScores, iouBoxSet, method, gtCount)
                apMask,arMask = evaluate.calculate_ap(maskScores, iouMaskSet, method, gtCount)

                maskAp[len(maskAp) - 1][method] = apMask
                boxAp[len(boxAp) - 1][method] = apBox
                maskAr[len(maskAp) - 1][method] = arMask
                boxAr[len(boxAr) - 1][method] = arBox

    print('average duration: ', utils.convertTime(np.mean(durations)))
    return maskAp, boxAp, maskAr, boxAr

main()
