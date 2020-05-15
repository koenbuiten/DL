import glob
import os
import numpy as np
import sys
current_dir = "Yolo/data/images"
split_pct = 10  # 10% validation set
file_train = open("Yolo/data/train.txt", "w")  
file_val = open("Yolo/data/val.txt", "w")  
counter = 1  
index_test = round(100 / split_pct)  
for fullpath in glob.iglob(os.path.join(current_dir, "*.PNG")):  
  title, ext = os.path.splitext(os.path.basename(fullpath))
  if counter == index_test:
    counter = 1
    file_val.write(current_dir + "/" + title + '.PNG' + "\n")
  else:
    file_train.write(current_dir + "/" + title + '.PNG' + "\n")
    counter = counter + 1
file_train.close()
file_val.close()