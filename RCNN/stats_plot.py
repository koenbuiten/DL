import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
import os

# Plot the losses from a loss file made by the lunarRCNN program
# mean_over_iters set the about of iteration the loss is averaged.
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
    # plt.show()
    return fig, 'loss_' + modelName


def createTitle(boundaryClass):
    if boundaryClass[1] == 1:
        title = boundaryClass[0] + ' detection for big rocks'
    else:
        title = boundaryClass[0] + ' detection for small rocks'
    return title

# Plots the average precision from a evaluation file made by the lunarRCNN program
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

    allAps = {columns[0]: [], columns[1]: [], columns[2]: []}
    for apIdx in range(aps_per_epoch):
        stats = {columns[0]: [], columns[1]: [], columns[2]: []}
        for epoch in range(epochs):
            for keyIdx in range(len(columns)):
                stats[columns[keyIdx]].append(statsFile[apIdx+epoch][keyIdx+4])
                allAps[columns[keyIdx]].append(statsFile[apIdx+epoch][keyIdx+4])
        for columnIdx in range(len(columns)):
            ax[x][y].plot(stats[columns[columnIdx]],'-o')
            ax[x][y].set_xlabel('Epoch\n')
            ax[x][y].set_ylabel('Average precision')
            ax[x][y].set_title(createTitle(statsFile[apIdx][2:4]))
            ax[x][y].legend(columns)
        if x == 0:
            x = 1
        else:
            x = 0
            y = 1
    fig.subplots_adjust(hspace = 0.25)
    mAp = {columns[0]: [np.mean(allAps[columns[0]])], columns[1]: [np.mean(allAps[columns[1]])], columns[2]: [np.mean(allAps[columns[2]])]}
    fig.suptitle("Average precision per class, object indicator and evaluation method",fontsize = 24 )
    plt.figtext(0.325,0.92,'\nM' + columns[0] + ': ' + str(round(mAp[columns[0]][0],4)) + '  M' + columns[1] + ': ' + str(round(mAp[columns[1]][0],4)) + '  M' + columns[2] + ': ' + str(round(mAp[columns[2]][0],4)),fontsize = 12 )
    print(modelName)
    return fig, 'eval_' + modelName


# Save figures for every stats file in the folder
loc = './stats/'
statsList = list(sorted(os.listdir(loc)))
print(statsList)
for statfile in statsList:
    print(statfile)
    if statfile[0:4] == 'eval':
        fig, name = plot_ap(loc + statfile,loc)
        print(name)
        fig.savefig('./figures/' + name + '.jpg')
    if statfile[0:4] == 'loss':

        fig, name = plot_loss(loc + statfile,50,loc)
        fig.savefig('./figures/' + name + '.jpg')
    else:
        print('wrong stats file: ' + statfile)