import os, shutil
from sklearn.model_selection import train_test_split

# Read images and annotations
root = "/home/ubuntu/workspace/dataset/road-sign-detection"

images = sorted([os.path.join(root, 'images', x) for x in os.listdir(f'{root}/images')])
annotations = sorted([os.path.join(root, 'annotations', x) for x in os.listdir(f'{root}/annotations') if x[-3:] == "txt"])

print(images[:10])
print(annotations[:10])

#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    os.makedirs(destination_folder, exist_ok = True)
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print("Data already exists skipping..."); continue
            assert False

train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 2023)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 2023)
            
# Move the splits into their folders
move_files_to_folder(train_images, f'{root}/object_detection/images/train/')
move_files_to_folder(val_images, f'{root}/object_detection/images/val/')
move_files_to_folder(test_images, f'{root}/object_detection/images/test/')
move_files_to_folder(train_annotations, f'{root}/object_detection/labels/train/')
move_files_to_folder(val_annotations, f'{root}/object_detection/labels/val/')
move_files_to_folder(test_annotations, f'{root}/object_detection/labels/test/')