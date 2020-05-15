# Explanation of the various files
- **lunarRCNN.py** is the main file, this file should be run with the name of a settings file as the first parameter, e.g.  "python lunarRCNN.py SGD_0.01_0.9_0.0001.txt". The **settings** folder contains several of these files for reference. The need for the settings file is to be able to run batch jobs on the peregrine hpc from the RUG. The program saves the model in the **models** folder, the evalation and loss statistics are put in the **stats** folder.
- **evaluation.py** contains the functions to evaluate the model.
- **utils.py** contains miscellaneous function which are used throughout the project.
- **engine.py** contains the functions for pre-processing and training the model.
- **transforms.py** contains the functions for the transformations on the images.
- **model_visualization.py** can be run for the evaluation of a model and to visualize the output. For every image in the test dataset, the programs makes a figure with the ground truths and predictions. See **example_outputs** for results from running this program. To run this yourself, change the **lunar_loc**, **data_loader, data_loader_test** and **modelName** on line 15, 127, 128 and 129 to your own data folder, dataloaders and model.
- **stats_plot.py** makes plots for the loss and average precision data which is saved by **lunarRCNN.py** in the **stats** folder. When the program is ran, this will looks for the files in the **stats** folder and outputs the plots in the **figures** folder.

# To make the program run
1. Open your python environment: python or python3 (in most cases)
2. type **import utils** enter
3. type **utils.create_only_rocks_dataset(lunar_loc)** enter, where **lunar_loc** is the path to the **artificial-lunar-rocky-landscape-dataset** folder, i.e. **utils.create_only_rocks_dataset('home/DL/Datasets/artificial-lunar-rocky-landscape-dataset')**
4. type **exit()** enter
5. Go to **lunarRCNN.py** and change line 40 to your **artificial-lunar-rocky-landscape-dataset** folder
6. run **python lunarRCNN.py SGD_0.01_0.9_0.0001.txt**


