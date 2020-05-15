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

# The functions below up till line 317 are all directly copied from: https://github.com/pytorch/vision/blob/master/references/detection/utils.py
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
    return dist.get_world_size



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

# ---------- Functions not copied from tutorial --------
# The below function are all helper function for the model to work or to visualize the results.

# Converts and id to a four digit id
def createName(name):
    while len(name) != 4:
        name = '0' + name
    return name

def draw_boxes(boxes,ax,color):
    ax.axis('off')
    j = 0
    for i in boxes:
        # print(i)
        rect = pat.Rectangle((i[0], i[1]), i[2] - i[0], i[3] - i[1], linewidth=1, edgecolor=color, facecolor='none')
        ax.plot([i[0]], [i[1]], '-o',color = color)
        ax.text(i[0], i[1],str(j), color = 'w')
        j = j + 1
        ax.add_patch(rect)

#convert time to hours:minutes:seconds:milliseconds
def convertTime(duration):

    mins = duration // 60
    secs = int(duration % 60)
    ms = int(((duration%60)-secs)*1000)
    hours = int(mins // 60)
    mins = int(mins % 60)
    duration = str(hours)+':'+str(mins)+':'+str(secs)+':'+str(ms)
    return(duration)

# Draws given masks on the plot(ax) in a certain color
def draw_masks(masks,ax,color):
    ax.axis('off')
    i = 0
    for mask in masks:
        # print(len(mask))
        if len(mask) == 1:
            maskPos = np.where(mask[0] == 1)
        else:
            maskPos = np.where(mask == 1)
        # print(len(np.unique(mask)))
        # print(maskPos)
        ax.plot(maskPos[1], maskPos[0], color = color)
        i = i + 1
    return

# Creates a new dataset, where all the images which do not include any rocks are removed.
def create_only_rocks_dataset(data_loc):
    if not os.path.exists(os.path.join(data_loc, "images/clean2")):
        os.mkdir(os.path.join(data_loc, "images/clean2"))
    if not os.path.exists(os.path.join(data_loc, "images/render2")):
        os.mkdir(os.path.join(data_loc, "images/render2"))

    mask_list = list(sorted(os.listdir(os.path.join(data_loc, "images/clean"))))
    imgs = []
    masks = []

    jdx = 1
    for idx in range(len(mask_list)):
        mask_path = os.path.join(data_loc, "images/clean", mask_list[idx])
        imgMask = Image.open(mask_path)
        mask = np.array(imgMask)
        if mask_list[idx][5:9] != '0028': # Image is missing in the dataset
            if not isinstance((mask == [0, 255, 0]), bool):
                if (True in (mask == [0, 255, 0])) or (True in (mask == [0, 0, 255])):
                    imageName = createName(str(jdx))
                    imgMask.save(os.path.join(data_loc, "images/clean2/clean" + imageName + ".png"))
                    masks.append(mask_list[idx])
                    img_path = os.path.join(data_loc, "images/render/render" + mask_list[idx][5:9] + ".png")
                    image = Image.open(img_path)
                    image.save(os.path.join(data_loc, "images/render2/render" + imageName + ".png"))
                    imgs.append("render" + mask_list[idx][5:9] + ".png")

                    jdx = jdx + 1


# Saves given losses from the dict to a csv file with the indicated epoch as index
def save_setting_to_file(settings):
    seperator = '_'
    modelName = seperator.join([settings['optimizer'], str(settings['lr'][0]), str(settings['momentum'][0]), str(settings['wd'][0])])
    settings = pd.DataFrame(settings)
    file = './settings/' + modelName + '.txt'

    with open(file, 'w') as csv_file:
        settings.to_csv(path_or_buf=file)

# Function which save the settings for a model into a text file
def save_all_settings(optimizers, learning_rates, momentums, weight_decays):
    for optimizer in optimizers:
        for lr in learning_rates:
            for momentum in momentums:
                for wd in weight_decays:
                    settings = {'optimizer': optimizer, "lr": [lr], "momentum": [momentum], "wd":[wd]}
                    save_setting_to_file(settings)

# Given a setting file name, it extracts the setting and puts them in a dictionary
def get_settings(file_id):
    settings = pd.read_csv('./settings/'+file_id)
    settings = settings.to_dict('records')[0]
    del settings['Unnamed: 0']
    return settings

# Makes a text file with all the setting in the settings folder
# This list can be used to run a batch job on the peregrine cluster, where every run has different settings.
def make_settings_list():
    file = open('./settings/settingsList.in', "w")
    settingsList = list(sorted(os.listdir("./settings")))
    for settings in settingsList:
        if settings != 'settingsList.in':
            file.write(settings+'\n')
