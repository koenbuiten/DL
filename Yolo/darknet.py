from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2

from yoloutils import *

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parsecfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x
        
        return detections
                    

# Empty layer class
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
    
    def forward(self, x):
        padded_x = F.pad(x, (0,self.pad,0,self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x

# Detection Layer for Yolo layer
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

# Reads through the cfg file and creates a list of dictionaries
# The dictionaries content each layers parameters
def parsecfg(configfile):
    file = open(configfile,'r')
    
    lines = file.read().split('\n')             # Lisf of lines of the cfg file
    lines = [x for x in lines if len(x) > 0]    # Removes the white lines
    lines = [x for x in lines if x[0] != '#']   # Removes the comments
    lines = [x.strip() for x in lines]          # Removes blank spaces left and right of the line

    block = {}
    layers_list = []
    
    for line in lines:
        if line[0] == "[":                      # layers in the cfg file start with "["
            if len(block) != 0:                 # If block is not empty, implies it is storing values of previous block.
                layers_list.append(block)            
                block = {}                      # Re-init the block
            block["type"] = line[1:-1].rstrip() # Get layer name/type (removes [])   
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip() # Adds the parameter to the dictionary, removing white spaces
    layers_list.append(block)

    return layers_list  

# 
def create_modules(layers_list):
    net_info = layers_list[0]       # The first "layer" in yolo.cfg contains info regarding the net. Input size and such
    module_list = nn.ModuleList()   # Empty list of nn.modules
    prev_filters = 3                # Used for depth of the previous input
    output_filters = []             # Keeps track of the depth of every layer


    for index,x in enumerate(layers_list[1:]): # Skips first element since it's the [Net] parameters
        module = nn.Sequential()
        
        # Convolutional layers
        if (x["type"] == "convolutional"):
            
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
                
            filters = int(x["filters"]) # Parse the dictionary elements into layer parameters
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
                
            # Add the Convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            
            # Add the Batch Norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            # Check the activation. 
            # It is either Linear or a Leaky ReLU for YOLO
            # Conv layers have only leaky ReLU
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        
        # Upsampling layers
        # We use Bilinear2dUpsampling
        
        elif (x["type"] == "upsample"):

            stride = int(x["stride"])
            # upsample = Upsample(stride)
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
        
        # Route layers
        elif (x["type"] == "route"):

            

            x["layers"] = x["layers"].split(',')
            
            # Start  of a route
            start = int(x["layers"][0])
            
            # End, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
                
            # Positive anotation
            if start > 0: 
                start = start - index
            
            if end > 0:
                end = end - index
          
            route = EmptyLayer()

            module.add_module("route_{0}".format(index), route)
            
            if end < 0:
                filters = output_filters[index + start] #+ output_filters[index + end]
            else:
                filters = output_filters[index + start]
                        
            
        
        # Shortcut layers
        # Skip connections
        elif x["type"] == "shortcut":
            #from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
            
        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            
            module.add_module("maxpool_{}".format(index), maxpool)
        
        # Yolo layers
        # They are the detection layers
        elif x["type"] == "yolo":

            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
            
            
        else:
            print("Error: Can't parse layer")
            assert False


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)  
    
    return (net_info, module_list)




def get_test_input():
    img = cv2.imread("Yolo/dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = Darknet("Yolo/yolo.cfg")
inp = get_test_input()
model.to(device)
pred = model(inp, torch.cuda.is_available())
print(pred)