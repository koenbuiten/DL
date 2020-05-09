from __future__ import print_function

import pandas as pd
import datetime
import errno
import os
import pickle
import time
from collections import defaultdict, deque

import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image


# Make sure image name has 4 integers
def createName(name):
    while len(name) != 4:
        name = '0' + name
    return name


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

# made by Koen, not from Pytorch tut
def draw_image_boxes(boxes,ax,img):
    # fig, ax = plt.subplots(1, figsize=(16, 9))
    # print(len(boxes))
    ax.axis('off')
    ax.imshow(img)
    for i in boxes:
        # print(i)
        rect = pat.Rectangle((i[0], i[1]), i[2] - i[0], i[3] - i[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.plot([i[0]], [i[1]], '-o')
        ax.add_patch(rect)

def draw_boxes(boxes,ax):
    # fig, ax = plt.subplots(1, figsize=(16, 9))
    # print(len(boxes))
    ax.axis('off')
    for i in boxes:
        # print(i)
        rect = pat.Rectangle((i[0], i[1]), i[2] - i[0], i[3] - i[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.plot([i[0]], [i[1]], '-o')
        ax.add_patch(rect)

def convertTime(duration):
    #convert time to hours:minutes:seconds:milliseconds
    mins = duration // 60
    secs = int(duration % 60)
    ms = int(((duration%60)-secs)*1000)
    hours = int(mins // 60)
    mins = int(mins % 60)
    duration = str(hours)+':'+str(mins)+':'+str(secs)+':'+str(ms)
    return(duration)

def draw_masks(masks,ax):
    # print(len(masks))
    # zerosMasks = mask[mask ==
    # print(len(masks))
    # print(masks)
    # plt.imshow(masks[0][0])

    for mask in masks:
        # print(len(mask))
        if len(mask) == 1:
            maskPos = np.where(mask[0] == 1)
        else:
            maskPos = np.where(mask == 1)
        # print(len(np.unique(mask)))
        # print(maskPos)
        ax.plot(maskPos[1], maskPos[0])
    return

def create_only_rocks_dataset(data_loc):
    img_list = list(sorted(os.listdir(os.path.join(data_loc, "images/render"))))
    mask_list = list(sorted(os.listdir(os.path.join(data_loc, "images/clean"))))
    imgs = []
    masks = []

    jdx = 1
    for idx in range(len(mask_list)):
        mask_path = os.path.join(data_loc, "images/clean", mask_list[idx])
        imgMask = Image.open(mask_path)
        mask = np.array(imgMask)

        if (True in ((mask == [0, 255, 0]).all(-1))) or (True in (mask == [0, 0, 255]).all(-1)):
            imageName = createName(str(jdx))
            imgMask.save(os.path.join(data_loc, "./images/clean2/clean" + imageName + ".png"))
            masks.append(mask_list[idx])

            img_path = os.path.join(data_loc, "images/render", img_list[idx])
            image = Image.open(img_path)
            image.save(os.path.join(data_loc, "images/render2/render" + imageName + ".png"))
            imgs.append(img_list[idx])

            jdx = jdx + 1

# Function which put an image from the test set into model.
# The output consist of the ground truth image, the boxes found by the model
# and the best matching box(if any) for every ground truth box.
def detect_single_image(model, imgId):
    # Set model to evaluation mode
    model.eval()
    device = torch.device('cpu')
    # Load in test dataset and get image and target(ground truth)
    dataset = LunarDataset(lunar_loc, get_transform(train=False))
    img, target = dataset.__getitem__(imgId-1)
    img = img.to(device)
    model.to(device)
    img = [img]

    # Put image into model
    output = model(img)


    predBoxes = output[0]['boxes'].detach()
    # print(output)
    predMasks = output[0]['masks'].detach()
    gtBoxes = target['boxes']
    gtMasks = target['masks']
    # print('gtMasks')
    # print(gtMasks)
    # print('predmasks')
    # print(predMasks[0,:][0])

    # print(predMasks)



    # Seperate the boxes which do not intersect with any of the ground truth boxes
    # from the ones which do intersect
    # no_intersection, intersectionBoxes = engine.seperate_boxes(predBoxes, gtBoxes)
    # best_scores, best_boxes = engine.evaluate_single_best(gtBoxes, intersectionBoxes)
    # precision, recall, recall2 = engine.evaluate_single(gtBoxes, predBoxes, 0.5)
    # maskPrecision, maskRecall = engine.evaluate_single_mask(gtMasks,predMasks,0.5)
    # print('Precision: ' + str(precision))
    # print('Recall: ' + str(recall))
    # print('Mask precision: ' + str(maskPrecision))
    # print('Mask recall: ' + str(maskRecall))
    # print('Recall2: ' + str(recall2))

    # Show image with ground truth boxes, predicted boxes and best boxes.
    imgId = createName(str(imgId))
    img_path = os.path.join(lunar_loc, "images/render2", 'render' + imgId + '.png')
    img = Image.open(img_path).convert('RGB')

    fig, ax = plt.subplots()
    ax.imshow(img)
    utils.draw_masks(gtMasks, ax)

    fig2, ax2 = plt.subplots()
    ax2.imshow(img)
    utils.draw_masks(predMasks, ax2)
    plt.show()



    # utils.draw_masks(predMasks, ax2)
    #
    # fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    # utils.draw_image_boxes(gtBoxes, ax[0], img)
    # utils.draw_image_boxes(predBoxes, ax[1], img)
    # utils.draw_image_boxes(best_boxes, ax[2], img)
    # plt.savefig("plots/masksPlot1.png")
    plt.show()

def save_setting_to_file(settings):
    seperator = '_'
    modelName = seperator.join([settings['optimizer'], str(settings['lr'][0]), str(settings['momentum'][0]), str(settings['wd'][0])])
    settings = pd.DataFrame(settings)
    file = './settings/' + modelName + '.txt'

    with open(file, 'w') as csv_file:
        settings.to_csv(path_or_buf=file)

def save_all_settings(optimizers, learning_rates, momentums, weight_decays):
    # fileid = 1
    # settingsList = list(sorted(os.listdir("./settings")))
    # fileid = len(settingsList)+1
    for optimizer in optimizers:
        for lr in learning_rates:
            for momentum in momentums:
                for wd in weight_decays:
                    settings = {'optimizer': optimizer, "lr": [lr], "momentum": [momentum], "wd":[wd]}
                    save_setting_to_file(settings)
                    # fileid = fileid + 1

def get_settings(file_id):
    settings = pd.read_csv('./settings/'+file_id)
    settings = settings.to_dict('records')[0]
    del settings['Unnamed: 0']
    return settings

def make_settings_list():
    file = open('./settings/settingsList.in', "w")
    settingsList = list(sorted(os.listdir("./settings")))
    for settings in settingsList:
        if settings != 'settingsList.in':
            file.write(settings+'\n')
