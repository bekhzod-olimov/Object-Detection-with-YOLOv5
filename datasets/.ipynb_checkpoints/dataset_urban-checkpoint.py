import os, shutil
from sklearn.model_selection import train_test_split
from glob import glob

# Read images and annotations
root = "/home/ubuntu/workspace/dataset/crowd"
ims = sorted(glob(f"{root}/ims/*/*.jpg"))
gts = sorted(glob(f"{root}/gts/*/*.txt"))

# print(ims[:10])
# print(gts[:10])
print(len(gts))
print(len(ims))

#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    os.makedirs(destination_folder, exist_ok = True)
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print("Data already exists skipping..."); continue
            
train_images, val_images, train_annotations, val_annotations = train_test_split(ims, gts, test_size = 0.2, random_state = 2023)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 2023)
            
# Move the splits into their folders
move_files_to_folder(train_images, f'{root}/object_detection/images/train')
move_files_to_folder(val_images, f'{root}/object_detection/images/val/')
move_files_to_folder(test_images, f'{root}/object_detection/images/test/')
move_files_to_folder(train_annotations, f'{root}/object_detection/labels/train/')
move_files_to_folder(val_annotations, f'{root}/object_detection/labels/val/')
move_files_to_folder(test_annotations, f'{root}/object_detection/labels/test/')