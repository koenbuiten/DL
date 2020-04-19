import os
import numpy as np
import torch
import pandas as pd
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
from torchvision.transforms import Normalize

import utils
from utils import show_image_box
from engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt
import matplotlib.patches as pat

lunar_loc = '/media/koenbuiten/8675c03f-5bb1-4466-8581-8f042a79029b/koenbuiten/Datasets/artificial-lunar-rocky-landscape-dataset'


class LunarDataset(object):
    def __init__(self, data_loc, transforms):
        self.data_loc = data_loc
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(data_loc, "images/render"))))
        self.masks = list(sorted(os.listdir(os.path.join(data_loc, "images/clean"))))


        self.boxes = pd.read_csv(os.path.join(data_loc, 'bounding_boxes.csv'))
        self.boxes = self.boxes.values
        # print(self.boxes[0:20])
        boxes = []
        images =[]
        noAnno = 0
        j = 0
        # Make all the images without a bounding box have a bounding box of 0,0,0,0
        # Puts multiple bounding boxes of one images on the index of the image.
        # datasize = len(self.imgs)
        datasize = 100
        for i in range(1,datasize):
            temp_box = [0, 0, 0, 0]
            if self.boxes[j,0] != i:
                noAnno = noAnno +1
                continue
                # boxes.append([])
                # boxes[i-1].append(temp_box)
            else:
                boxes.append([])
                images.append(self.imgs[i-1])
                while self.boxes[j,0] == i:
                    temp_box[0] = self.boxes[j,1]
                    temp_box[1] = self.boxes[j, 2] - self.boxes[j, 4]
                    temp_box[2] = self.boxes[j, 1] + self.boxes[j, 3]
                    temp_box[3] = self.boxes[j, 2] + self.boxes[j, 4]

                    boxes[len(images)-1].append(temp_box)

                    # print(boxes[i-1])
                    #Convert to x0, y0, x1, x2
                    # boxes[i-1] = boxes[i-1][1]-boxes[i-1][3]
                    # boxes[i - 1][2] = boxes[i-1][0] + boxes[i-1][2]
                    # boxes[i - 1][3] = boxes[i-1][1] + boxes[i-1][3]
                    j=j+1
        # print(noAnno)
        self.boxes = boxes
        self.imgs = images
        # print(boxes[0:20])

    def __getitem__(self, idx):
        # load images and boxes
        img_path = os.path.join(self.data_loc, "images/render", self.imgs[idx])
        mask_path = os.path.join(self.data_loc, "images/clean", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # print(img)
        # print('%s\n',img_path)
        mask = np.array(mask)

        # convert the PIL Image into a numpy array

        # instances are encoded as different colors

        # mask_Brocks = (mask == [0, 0, 255]).all(-1)
        # mask_Srocks = (mask == [0,255,0]).all(-1)
        # mask_ground = (mask == [0,0,0]).all(-1)
        #
        # mask_air = (mask == [255,0,0]).all(-1)
        # print(len(mask_Brocks))
        # # blue_mask = (mask_blue == 255)
        # fig, ax = plt.subplots(1, 5, figsize=(16,9))
        # ax[0].axis('off')
        # ax[0].imshow(img)
        # ax[1].axis('off')
        # ax[1].imshow(mask_Brocks)
        # ax[2].axis('off')
        # ax[2].imshow(mask_Srocks)
        # ax[3].axis('off')
        # ax[3].imshow(mask_ground)
        # ax[4].axis('off')
        # ax[4].imshow(mask_air)
        #

        boxes = self.boxes[idx]

        # show_image_box(img,boxes)


        num_objs = len(boxes)
        # print(boxes)
        if (len(boxes) == 1) & (boxes[0][2] == 0):
            print('no boxes')
            iscrowd = torch.ones((num_objs,), dtype=torch.int64)
        else:
            # print('there are boxes')
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # # convert everything into a torch.Tensor

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # print(boxes)

        # # print(boxes)
        # # # there is only one class
        #


        # print(num_objs)


        labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        #
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # # suppose all instances are not crowd

        #
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        # target["masks"] = masks
        #
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        print(target)
        return img, target

    def __len__(self):
        return len(self.imgs)



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
    # transforms.append(torchvision.transforms.Normalize(2, 0.5, inplace=False))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available

    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')

    device = torch.device('cpu')
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = LunarDataset(lunar_loc, get_transform(train=True))
    dataset_test = LunarDataset(lunar_loc, get_transform(train=False))
    # print(len(dataset))
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

    # let's train it for 10 epochs
    num_epochs = 1
    eval = []
    for epoch in range(num_epochs):
        print('start epoch ', epoch)
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        print('first training done')
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluation = evaluate(model, data_loader_test, device=device)
        eval.append(evaluation)
    return eval
    print("That's it!")

# dataset = LunarDataset(lunar_loc, get_transform(train=True))
# [img,target] =  dataset.__getitem__(3)
# print(img)
main()

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# dataset = LunarDataset(lunar_loc, get_transform(train=True))
#
# data_loader = torch.utils.data.DataLoader(
#  dataset, batch_size=2, shuffle=True, num_workers=4,
#  collate_fn=utils.collate_fn)
# # For Training
# images,targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images,targets)   # Returns losses and detections
# print(output)
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)           # Returns predictions