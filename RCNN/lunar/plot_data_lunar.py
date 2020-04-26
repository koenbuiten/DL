from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv('./stats.txt' )
# print(data['precision'])
plt.plot(data['epoch'], data['precision'])
plt.plot(data['epoch'], data['recall'])
plt.show()