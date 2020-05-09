import os

import engine
import matplotlib.pyplot as plt
import numpy as np
import torch
import transforms as T
import utils
from PIL import Image

# --- Goal ----
# Evaluation a single image on the trained model

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
    return name

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
        masks, boxes, labels = engine.label_objects(imgMask)


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        # print(masks)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "masks": masks}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Function which put an image from the test set into model.
# The output consist of the ground truth image, the boxes found by the model
# and the best matching box(if any) for every ground truth box.
def detect_single_image(model, imgId, iouThreshold, maskThreshold):
    # Set model to evaluation mode
    model.eval()
    device = torch.device('cpu')
    # Load in test dataset and get image and target(ground truth)
    dataset = LunarDataset(lunar_loc, get_transform(train=False))
    img, targets = dataset.__getitem__(imgId-1)
    img = img.to(device)
    model.to(device)
    img = [img]

    # Put image into model
    output = model(img)


    predBoxes = output[0]['boxes'].detach()
    # print(output)
    predMasks = output[0]['masks'].detach()
    gtBoxes = targets['boxes']
    gtMasks = targets['masks']

    gtLabels = targets['labels']
    predLabels = output[0]['labels']
    # print('gtMasks')
    # print(gtMasks)
    # print('predmasks')
    # print(predMasks[0,:][0])

    # print(predMasks)



    # Seperate the boxes which do not intersect with any of the ground truth boxes
    # from the ones which do intersect

    precision, recall = engine.evaluate_single(gtBoxes, predBoxes, gtLabels, predLabels, iouThreshold)
    maskPrecision, maskRecall = engine.evaluate_single_mask(gtMasks, predMasks, gtLabels, predLabels, iouThreshold, maskThreshold)


    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('Mask precision: ' + str(maskPrecision))
    print('Mask recall: ' + str(maskRecall))
    # print('Recall2: ' + str(recall2))

    # Show image with ground truth boxes, predicted boxes and best boxes.
    imgId = createName(str(imgId))
    img_path = os.path.join(lunar_loc, "images/render2", 'render' + imgId + '.png')
    img = Image.open(img_path).convert('RGB')


    fig, ax = plt.subplots(1, 2, figsize=(16, 9))

    utils.draw_image_boxes(gtBoxes, ax[0], img)
    utils.draw_masks(gtMasks, ax[0])
    utils.draw_image_boxes(predBoxes, ax[1], img)
    utils.draw_masks(predMasks, ax[1])
    plt.savefig("plots/masksPlot1.png")
    plt.show()

# dataset = LunarDataset(lunar_loc, get_transform(train=True))
# dataset.__getitem__(12)
model = torch.load(os.path.join(lunar_loc , "models/modelmasks_1.pt"), map_location='cpu')
# device = torch.device('cpu')
# model.to(device)
detect_single_image(model,11,0.5,0.8)


