import os ,shutil

data_dir = 'put the path to the GTSRB folder here'
output_dir = "put the path to the output folder here"
os.makedirs(os.path.join(output_dir, "GTSRB_Final_Test_GT"),exist_ok=True)
os.makedirs(os.path.join(output_dir, "GTSRB_Final_Test_Images"),exist_ok=True)
shutil.copytree(os.path.join(data_dir, "GTSRB_Final_Test_GT"), os.path.join(output_dir, "GTSRB_Final_Test_GT"),dirs_exist_ok = True)

shutil.copytree(os.path.join(data_dir, "GTSRB_Final_Test_Images"), os.path.join(output_dir, "GTSRB_Final_Test_Images"),dirs_exist_ok=True)

os.makedirs(os.path.join(output_dir, "GTSRB_Final_Train"), exist_ok = True)
train_path = os.path.join(data_dir, "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/")
for class_str in os.listdir(train_path) : 
    for file in os.listdir(os.path.join(train_path, class_str)):
        if ".csv" not in file:
            shutil.copy(os.path.join(train_path, class_str, file), os.path.join(output_dir, "GTSRB_Final_Train", str(int(class_str))+"@"+file))