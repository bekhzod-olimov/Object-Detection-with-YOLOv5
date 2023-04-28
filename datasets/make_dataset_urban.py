import xml.etree.ElementTree as ET
import os, numpy as np, json, glob
from tqdm import tqdm

# Function to get the data from XML Annotation

def extract_info_from_json(json_file):
    
    final_li = []
    js_data = json.load(open(f"{json_file}"))
    for i, data in enumerate(js_data['annotations']["image"]):
        info_dict = {}
        info_dict['bboxes'] = []
        if "box" not in data: 
            print("No bounding box found, skipping!")
            fname = data["@name"]
            dirname = os.path.splitext(fname)[0].split("_")[2]
            os.remove(f"/home/ubuntu/workspace/dataset/crowd/ims/{dirname}/{fname}")
            print(f"/home/ubuntu/workspace/dataset/crowd/ims/{dirname}/{fname} is deleted!")
            continue
        info_dict['filename'] = data["@name"]
        info_dict['image_size'] = int(data["@width"]), int(data["@height"]), 3
        
        if isinstance(data["box"], dict):
            bbox = {}
            bbox["class"] = data["box"]["@label"]
            bbox["xmin"] = round(float(data["box"]["@xtl"]))
            bbox["ymin"] = round(float(data["box"]["@ytl"]))
            bbox["xmax"] = round(float(data["box"]["@xbr"]))
            bbox["ymax"] = round(float(data["box"]["@ybr"]))
            info_dict['bboxes'] = [bbox]
        
        elif isinstance(data["box"], list):
            bounding_boxes = []
            for bboxes in data["box"]:
                bbox = {}
                bbox["class"] = bboxes["@label"]
                bbox["xmin"] = round(float(bboxes["@xtl"]))
                bbox["ymin"] = round(float(bboxes["@ytl"]))
                bbox["xmax"] = round(float(bboxes["@xbr"]))
                bbox["ymax"] = round(float(bboxes["@ybr"]))
                bounding_boxes.append(bbox)
            info_dict['bboxes'] = bounding_boxes
        # if i == 5: break
        final_li.append(info_dict)
    return final_li

# annotations = sorted(glob.glob("/home/ubuntu/workspace/dataset/crowd/gts/*/*.json"))
# for i, ann in enumerate(annotations):
#     if i == 10: break
#     print(f"\n{extract_info_from_json(ann)}\n")
    
# extract_info_from_json('/home/ubuntu/workspace/dataset/crowd/gts/202010170830/F18011_2_202010170830.json')
# print(extract_info_from_json('/home/ubuntu/workspace/dataset/crowd/gts/202010170830/F18011_2_202010170830.json'))
# extract_info_from_json('/home/ubuntu/workspace/dataset/crowd/gts/202010161800/F18011_2_202010161800.json')
# print(extract_info_from_json('/home/ubuntu/workspace/dataset/crowd/gts/202010161800/F18011_2_202010161800.json'))

# a = extract_info_from_json('/home/ubuntu/workspace/dataset/crowd/gts/202010161800/F18011_2_202010161800.json')

# ttt = []

# for data in a:
#     for info in data["bboxes"]:
#         if info["class"] not in ttt:
#             ttt.append(info["class"])
# print(ttt)


# Dictionary that maps class names to IDs
def get_class_mapping(): return {"사람": 0, "이륜차": 1, "자전거": 2}
# def get_class_mapping(): return {"person": 0, "wtf": 1, "bicycle": 2}

class_name_to_id_mapping = get_class_mapping()

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(root, final_li):
    
    for idx, data in enumerate(final_li):
        print_buffer = []
        fname = data["filename"] 
        dirname = os.path.splitext(fname)[0].split("_")[2]
        # if idx == 2: break
        # For each bounding box
        for bbox in data["bboxes"]:
            try:
                class_id = class_name_to_id_mapping[bbox["class"]]
            except KeyError:
                print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

            # Transform the bbox co-ordinates as per the format required by YOLO v5
            bbox_center_x = (bbox["xmin"] + bbox["xmax"]) / 2 
            bbox_center_y = (bbox["ymin"] + bbox["ymax"]) / 2
            bbox_width    = (bbox["xmax"] - bbox["xmin"])
            bbox_height   = (bbox["ymax"] - bbox["ymin"])

            # Normalise the co-ordinates by the dimensions of the image
            image_w, image_h, image_c = data["image_size"]  
            bbox_center_x /= image_w 
            bbox_center_y /= image_h 
            bbox_width    /= image_w 
            bbox_height   /= image_h 

            # print("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, bbox_center_x, bbox_center_y, bbox_width, bbox_height))
            #Write the bbox details to the file 
            print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, bbox_center_x, bbox_center_y, bbox_width, bbox_height))
        # Name of the file which we have to save 
        save_file_name = os.path.join(root, dirname, fname.replace("jpg", "txt"))
        # Save the annotation to disk
        print("\n".join(print_buffer), file = open(save_file_name, "w"))
    
def get_txt_files():
    
    root = "/home/ubuntu/workspace/dataset/crowd/gts"
    annotations = sorted(glob.glob(f"{root}/*/*.json"))
    for idx, ann in tqdm(enumerate(annotations)):
        # if idx == 1: break
        info_dict = extract_info_from_json(ann)
        convert_to_yolov5(root, info_dict)
    annotations = sorted(glob.glob(f"{root}/*/*.txt"))
    
    return annotations

get_txt_files()

