import time
import utils
import engine
import torch
import pandas as pd

from lunarRCNN import LunarDataset
from lunarRCNN import get_transform

# lunar_loc = unar_loc = '/data/s3861023/lunarDataset'
lunar_loc = '/media/koenbuiten/8675c03f-5bb1-4466-8581-8f042a79029b/koenbuiten/Datasets/artificial-lunar-rocky-landscape-dataset'

startT = time.time()
device = torch.device('cuda')
model2 = torch.load('models/model1000.pt')
model2.to(device)

dataset = LunarDataset(lunar_loc, get_transform(train=True))

dataset_test = LunarDataset(lunar_loc, get_transform(train=False))
indices = torch.randperm(len(dataset_test)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

# detect_single_image(model2, data_loader)
print("start eval")
evaluation = engine.evaluate(model2,data_loader,0.5,device)
print('eval done')
stats = {'epoch': [], 'precision': [], 'recall': []}
stats['epoch'].append(0)
stats['precision'].append(evaluation[0])
stats['recall'].append(evaluation[1])

stats = pd.DataFrame(stats)

with open('csv_data.txt', 'w') as csv_file:
    stats.to_csv(path_or_buf=csv_file)

endT = time.time()
duration = endT - startT
print(duration)