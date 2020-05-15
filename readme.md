Readme file for the DL project: Rock detection for the Lunar Surface

running the programm: python lunarRCNN SGD_0.01_0.9_0.0001.txt (The text refers to certain settings used for the model, the settings folder contains several settings files for reference). The dataLoader folder should be made to be able to run the code.

There are 7 files:
- lunarRCNN.py is the main file, this file should be run with a name of a settings file like: "python 		lunarRCNN SGD_0.01_0.9_0.0001.txt". The settings folder contains several settings files for reference. The program is constructed this way to be able to run batch jobs on the peregrine hpc from the rug. The program save the model in the models folder, the evalation and loss statistics in the stats folder.
- evaluation.py is the file which has all the function to evaluate the model
- utils.py includes all kinds of function which are used by multiple other files.
- engine.py contain the function need for pre-processing and training the model
- transforms.py have function for transformations on the images.
- model_visualization.py can be run for the evaluation of a model and to visualize the output. For every image in the test dataset, the programs makes a figure with the ground truths and predictions. See example_outputs for results from running this program. (In the programm file the specific model and test dataloader should be specified to run)
- stats_plot.py make plots for the loss data and average precision data which is saved by lunarRCNN.py in the stats folder. When the program is run: python stats_plot.py, the program looks for all the files in the stats folder and saves figures the plots in the figures folder


