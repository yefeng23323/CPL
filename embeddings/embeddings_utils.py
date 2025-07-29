import torch
import pickle
import os
from mmfewshot.detection.datasets.voc import VOC_SPLIT
import pdb
from filelock import FileLock


PASCAL_CALASS_NAMES = [
            "background",  # class 0
            "airplane",  # class 1
            "bicycle",  # class 2
            "bird",  # class 3
            "boat",  # class 4
            "bottle",  # class 5
            "bus",  # class 6
            "car",  # class 7
            "cat",  # class 8
            "chair",  # class 9
            "cow",  # class 10
            "table",  # class 11
            "dog",  # class 12
            "horse",  # class 13
            "motorbike",  # class 14
            "person",  # class 15
            "potted_plant",  # class 16
            "sheep",  # class 17
            "sofa",  # class 18
            "train",  # class 19
            "tv",  # class 20
        ]


new_to_original_map = {
    'aeroplane': 'airplane',
    'bicycle': 'bicycle',
    'boat': 'boat',
    'bottle': 'bottle',
    'car': 'car',
    'cat': 'cat',
    'chair': 'chair',
    'diningtable': 'table',
    'dog': 'dog',
    'horse': 'horse',
    'person': 'person',
    'pottedplant': 'potted_plant',
    'sheep': 'sheep',
    'train': 'train',
    'tvmonitor': 'tv',
    'bird': 'bird',
    'bus': 'bus',
    'cow': 'cow',
    'motorbike': 'motorbike',
    'sofa': 'sofa'
}

COCO_CLASS_NAMES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic", "light", # 0-10
    "fire_hydrant", "stop", "sign", "parking", "meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", # 11-20
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee" # 21-30
    "skis", "snowboard", "sports", "ball", "kite", "baseball", "bat", "baseball", "glove", "skateboard", "surfboard", "tennis_racket", "bottle" # 31-40
    "wine", "glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange" # 41-50
    "broccoli", "carrot", "hot", "dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant", "bed" # 51-60
    "dining", "table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven" # 61-70
    "toaster", "sink", "fridge", "book", "clock", "vase", "scissors", "teddy_bear", "hair", "drier", "toothbrush" # 71-80
]

COCO_CLASS_NAMES_clip=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


new_to_original_map_clip_coco = {
    'person':'person', 'bicycle':'bicycle', 'car':'car', 'motorcycle':'motorcycle', 'airplane':'airplane', 'bus':'bus',
    'train':'train', 'truck':'truck', 'boat':'boat', 'traffic light':'traffic light', 'fire hydrant':'fire hydrant',
    'stop sign':'stop sign', 'parking meter':'parking meter', 'bench':'bench', 'bird':'bird', 'cat':'cat', 'dog':'dog',
    'horse':'horse', 'sheep':'sheep', 'cow':'cow', 'elephant':'elephant', 'bear':'bear', 'zebra':'zebra',
    'giraffe':'giraffe', 'backpack':'backpack', 'umbrella':'umbrella', 'handbag':'handbag', 'tie':'tie',
    'suitcase':'suitcase', 'frisbee':'frisbee', 'skis':'skis', 'snowboard':'snowboard', 'sports ball':'sports ball',
    'kite':'kite', 'baseball bat':'baseball bat', 'baseball glove':'baseball glove', 'skateboard':'skateboard',
    'surfboard':'surfboard', 'tennis racket':'tennis racket', 'bottle':'bottle', 'wine glass':'wine glass', 'cup':'cup',
    'fork':'fork', 'knife':'knife', 'spoon':'spoon', 'bowl':'bowl', 'banana':'banana', 'apple': 'apple',
    'sandwich':'sandwich', 'orange':'orange', 'broccoli':'broccoli', 'carrot':'carrot', 'hot dog':'hot dog',
    'pizza':'pizza', 'donut':'donut', 'cake':'cake', 'chair':'chair', 'couch':'couch', 'potted plant':'potted plant',
    'bed':'bed', 'dining table':'dining table', 'toilet':'toilet', 'tv':'tv', 'laptop':'laptop', 'mouse':'mouse',
    'remote':'remote', 'keyboard':'keyboard', 'cell phone':'cell phone', 'microwave':'microwave', 'oven':'oven',
    'toaster':'toaster', 'sink':'sink', 'refrigerator':'refrigerator', 'book':'book', 'clock':'clock', 'vase':'vase',
    'scissors':'scissors', 'teddy bear':'teddy bear', 'hair drier':'hair drier', 'toothbrush':'toothbrush'
}

PASCAL_CALASS_NAMES_clip = ['aeroplane',    # 0
                       'bicycle',      # 1
                       'boat',          # 2
                       'bottle',         # 3
                       'car',            # 4
                       'cat',           # 5
                       'chair',         # 6
                       'diningtable',  # 7
                       'dog',          # 8
                       'horse',    # 9
                       'person',  # 10
                       'pottedplant', # 11
                       'sheep',  # 12
                       'train',  # 13
                       'tvmonitor',  # 14
                       'bird',  # 15
                       'bus',  # 16
                       'cow',  # 17
                       'motorbike', # 18 
                       'sofa'] # 19


new_to_original_map_clip = {
    'aeroplane': 'aeroplane',
    'bicycle': 'bicycle',
    'boat': 'boat',
    'bottle': 'bottle',
    'car': 'car',
    'cat': 'cat',
    'chair': 'chair',
    'diningtable': 'diningtable',
    'dog': 'dog',
    'horse': 'horse',
    'person': 'person',
    'pottedplant': 'pottedplant',
    'sheep': 'sheep',
    'train': 'train',
    'tvmonitor': 'tvmonitor',
    'bird': 'bird',
    'bus': 'bus',
    'cow': 'cow',
    'motorbike': 'motorbike',
    'sofa': 'sofa'
}

def map_indices_to_new_classes_clip(original_classes, original_indices, all_classes):
    """
    Map the indices of the original classes to the indices of the new classes.

    Parameters:
        original_classes (list): List of names of the original classes.
        original_indices (list): List of indices corresponding to the original classes.
        new_classes (list): List of names of the new classes.
        new_to_original_map (dict): A mapping dictionary from original classes to new classes.

    Returns:
        list: A list of new class indices after mapping.
    """
    new_indices = []

    for original_index in original_indices:
        original_class = original_classes[original_index]

        # coco/voc
        if all_classes =='ALL_CLASSES':
            new_class = new_to_original_map_clip_coco[original_class]  
            new_index = COCO_CLASS_NAMES_clip.index(new_class)  
        #voc
        else:
            new_class = new_to_original_map_clip[original_class]  
            new_index = PASCAL_CALASS_NAMES_clip.index(new_class)  

        new_indices.append(new_index) 
    return new_indices




def map_indices_to_new_classes(original_classes, original_indices, all_classes):

    new_indices = []

    for original_index in original_indices:
        original_class = original_classes[original_index]
        new_class = new_to_original_map[original_class] 
        new_index = PASCAL_CALASS_NAMES.index(new_class) 
        new_indices.append(new_index) 
    return new_indices

def load_obj(name):

    base_dir = os.path.dirname(__file__)  

    file_path = os.path.join(base_dir, name + ".pkl")

    with open(file_path, "rb") as f:
        embed = pickle.load(f, encoding="latin-1")
    # vec
    # with open(file_path, "rb") as f:
    #     embed = torch.from_numpy(pickle.load(f, encoding="latin-1"))
    
    embed.requires_grad = False
    return embed   # [20, 512] 


if __name__ == "__main__":


    original_indices = [0, 17, 15, 1, 8, 5, 11, 3, 18, 16, 13, 2, 9, 19, 4, 12, 7, 10, 14, 6]

    new_indices = map_indices_to_new_classes_clip(VOC_SPLIT['ALL_CLASSES_SPLIT1'] , original_indices)
    embedings = load_obj('clip_pascal')
    embedings = embedings[new_indices]
    pdb.set_trace()
    # with open('inference_support_dict.pkl', 'wb') as f:
    #     pickle.dump(embedings, f)  
