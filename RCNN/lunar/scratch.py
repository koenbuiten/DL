import sys
import os
import pandas as pd
import numpy as np
import csv
# print(str(sys.argv))

# def main(argvList):
#     print(str(sys.argv[1]))
#     # print (str(sys.argv))
# main(sys.argv)
def createName(name):
    while len(name) != 3:
        name = '0' + name
    return name

def save_setting_to_file(fileId, settings):
    settings = pd.DataFrame(settings)
    settings.reset_index(drop=True, inplace=False)
    fileId = createName(str(fileId))
    file = './settings/settings' + fileId + '.txt'

    with open(file, 'w') as csv_file:
        settings.to_csv(path_or_buf=file)

def save_all_settings(optimizers, learning_rates, momentums, weight_decays):
    fileid = 1
    for optimizer in optimizers:
        for lr in learning_rates:
            for momentum in momentums:
                for wd in weight_decays:
                    settings = {'optimizer': optimizer, "lr": [lr], "momentum": [momentum], "wd":[wd]}
                    save_setting_to_file(fileid, settings)
                    fileid = fileid + 1

optimizers = ['SGD', 'Adagrad', 'Adadelta']
learning_rates = [0.005, 0.01]
momentums = [0.8,0.9]
weight_decays = [0.0005,0.001]

save_all_settings(optimizers, learning_rates, momentums, weight_decays)

# print(settings_list)
# print(np.sort(settings_list))
def get_settings(file_id):
    settings = pd.read_csv('./settings/settings' + createName(str(file_id)) + '.txt')
    settings = settings.to_dict('records')[0]
    del settings['Unnamed: 0']
    return settings



get_settings(1)
def main():
    settings = get_settings(sys.argv[1])
    print(settings)

main()



