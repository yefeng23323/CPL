import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as trans

from mmfewshot.detection.datasets.coco import COCO_SPLIT
from mmfewshot.detection.datasets.voc import VOC_SPLIT
from torchvision.transforms import InterpolationMode
import pickle
import pdb

class PCB:
    def __init__(self, class_names, model="RN101", templates="a photo of {}"):
        super().__init__()
        self.device = torch.cuda.current_device()
        # CLIP configs
        # import clip
        import clip
        self.class_names = class_names
        self.clip, _ = clip.load(model, device=self.device) 
        self.prompts = clip.tokenize([ 
            templates.format(cls_name)
            for cls_name in self.class_names
        ]).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(self.prompts)
            pdb.set_trace()
            # self.text_features = text_features
            self.text_features = F.normalize(text_features, dim=-1, p=2) #   [20, 512]

    @torch.no_grad()
    def __call__(self, img_path, boxes):
        images = self.load_image_by_box(img_path, boxes)
        image_features = self.clip.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
        logit_scale = self.clip.logit_scale.exp()
        # pdb.set_trace()
        logits_per_image = logit_scale * image_features @ self.text_features.t() 
        return logits_per_image.softmax(dim=-1)


if __name__ == "__main__":
    # pcb = PCB(VOC_SPLIT['ALL_CLASSES_SPLIT1'], model='ViT-B/32')
    pcb = PCB(COCO_SPLIT['ALL_CLASSES'], model='ViT-B/32')
    pdb.set_trace()
    with open('clip_coco.pkl', 'wb') as f:
        pickle.dump(pcb.text_features, f)  
