import os, numpy as np, shutil
from make_dataset import *
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

# Dictionary that maps class names to IDs
class_name_to_id_mapping = get_class_mapping()
annotations = get_txt_files()
class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def sample_im(image, annotation_list, randint, save_path):
    
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        font = ImageFont.truetype("arial-unicode-ms.ttf", 30)
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))], font = font)
    
    plt.imshow(np.array(image))
    plt.axis("off")
    plt.savefig(f"{save_path}/sample_im_{randint}.jpg")
    
# Get any random annotation file 
def check_data(save_path, annotations):
    
    if os.path.isdir(save_path): shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok = True)
    
    randints = np.random.randint(0, len(annotations), 5)

    for randint in randints:
        annotation_file = annotations[int(randint)]
        with open(annotation_file, "r") as file:
            annotation_list = file.read().split("\n")[:-1]
            annotation_list = [x.split(" ") for x in annotation_list]
            annotation_list = [[float(y) for y in x ] for x in annotation_list]

        #Get the corresponding image file
        image_file = annotation_file.replace("gts", "ims").replace("txt", "jpg")
        assert os.path.exists(image_file)

        #Load the image
        image = Image.open(image_file)

        #Plot the Bounding Box
        sample_im(image, annotation_list, randint, save_path)
        
check_data("data_samples", annotations)