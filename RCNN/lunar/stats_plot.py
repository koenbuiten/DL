import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
import os

def plot_loss(lossFile, mean_over_iters,loc):
    newLoss = {}

    modelName = lossFile[len(loc)+4:len(lossFile) - 4]

    # The loss have to be parsed and are saved in newLoss
    with open(lossFile) as loss_file:
        csv_reader = csv.reader(loss_file, delimiter=',')
        columns = next(csv_reader)
        columns[0] = 'epoch'
        # del columns[6]
        for column in columns:
            newLoss[column] = []
        # line_count = 0
        for row in csv_reader:
            newLoss['epoch'].append(int(row[0]))
            for columnIdx in range(1,len(columns)):
                newLoss[columns[columnIdx]].append(float(row[columnIdx][7:13]))


    # Set some variables for the amount of plotted points
    epochs = max(np.unique(newLoss['epoch']))+1
    total_iters = len(newLoss['epoch'])
    iter_per_epoch = int(total_iters/epochs)
    fig, ax = plt.subplots(figsize=(8, 5))

    if mean_over_iters == 0:
        mean_over_iters = iter_per_epoch
        ax.set_xlabel('epoch')
    else:
        ax.set_xlabel('Mean of ' + str(mean_over_iters) +' iterations')
    ax.set_ylabel('loss value')

    # Loop over the different instances of loss
    for column in columns:
        losses = []
        if column != 'epoch':
            for iteration in range(len(newLoss[column])):
                # print(iteration)
                if (iteration % mean_over_iters == 0) & (iteration !=0):
                    mean = np.mean(newLoss[column][iteration-mean_over_iters:iteration])
                    # print(mean)
                    losses.append(mean)
                    # break
            # plt.plot(newLoss['loss'])
            #     print(losses)
            ax.plot(losses,'-o')
    ax.legend(columns[1:])
    plt.show()
    return fig, 'loss_' + modelName


def createTitle(boundaryClass):
    if boundaryClass[1] == 1:
        title = boundaryClass[0] + ' detection for big rocks'
    else:
        title = boundaryClass[0] + ' detection for small rocks'
    return title

# lossFile = './stats/loss_SGD_0.005_0.8_0.0005.csv'
# apFile = './stats/eval_SGD_0.005_0.8_0.0005.csv'

def plot_ap(apFile,loc):
    # Read in the ap file
    statsFile = pd.read_csv(apFile)
    # Get the different column names
    columns = statsFile.columns[4:]
    modelName = apFile[len(loc)+5:len(apFile)-4]
    # Get the amount of epochs
    epochs = np.max(statsFile.get('epoch'))+1
    # Get the amount of different ap instances (mask, box, small rocks, big rocks)
    aps_per_epoch = int(len(statsFile.get('epoch'))/epochs)

    statsFile = statsFile.values

    # intialize figure with 4 plots
    fig, ax = plt.subplots(2, 2,figsize = (16,9))
    x,y = 0,0

    # loop over different ap instances, epochs and ap values(
    for apIdx in range(aps_per_epoch):
        stats = {columns[0]: [], columns[1]: [], columns[2]: []}
        for epoch in range(epochs):
            for keyIdx in range(len(columns)):
                stats[columns[keyIdx]].append(statsFile[apIdx+epoch][keyIdx+4])

        for columnIdx in range(len(columns)):
            ax[x][y].plot(stats[columns[columnIdx]],'-o')
            ax[x][y].set_title(createTitle(statsFile[apIdx][2:4]))
            ax[x][y].legend(columns)
        if x == 0:
            x = 1
        else:
            x = 0
            y = 1


    fig.suptitle(modelName,fontsize = 16)
    print(modelName)
    return fig, 'eval_' + modelName


# Save figures for every stats file
loc = './single_run_stat/'
plot_loss('./single_run_stat/loss_SGD_0.01_0.9_0.0001_nt.csv',10,loc)
# statsList = list(sorted(os.listdir(loc)))
# for statfile in statsList:
#     # print(statfile)
#     if statfile[0:4] == 'eval':
#         fig, name = plot_ap(loc + statfile,loc)
#         print(name)
#         fig.savefig('./figures/' + name + '.jpg')
#     elif statfile[0:4] == 'loss':
#
#         fig, name = plot_loss(loc + statfile,0,loc)
#         fig.savefig('./figures/' + name + '.jpg')
#     else:
#         print('wrong stats file: ' + statfile)