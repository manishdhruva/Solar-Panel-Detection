# importing dependencies
import torch
assert torch.__version__.startswith("1.8") 
import torchvision
import cv2
import zipfile
import os
import numpy as np
import shutil
import json
import math
import random
from tqdm import tqdm

# importing detectron2 modules
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.projects import point_rend

class config:

  def __init__(self, img_height, img_width, base_lr, max_iter, img_per_batch):
    self.img_height = img_height
    self.img_width = img_width
    self.base_lr = base_lr
    self.max_iter = max_iter
    self.img_per_batch = img_per_batch

# function to split dataset into train and test set
  def traintestsplit(self, data_annot_path):
    dataset_path = './Dataset-' + data_annot_path.split('/')[-1] + '/'
    os.mkdir(dataset_path)
    train_path = dataset_path + 'train'
    os.mkdir(train_path)
    test_path = dataset_path + 'test'
    os.mkdir(test_path)
     
    total_files = len(os.listdir(data_annot_path))
    train_len = math.floor(total_files*0.80) if ((math.floor(total_files*0.80)%2)==0) else math.ceil(total_files*0.80)
    count=1

    for file in sorted(os.listdir(data_annot_path)):
      if(count<=train_len):
        shutil.move(os.path.join(data_annot_path, file), train_path)
      else:
        shutil.move(os.path.join(data_annot_path, file), test_path)
      count+=1

    return dataset_path


# function to create a dictionary of annotations and their objects
  def get_data_dicts(self, directory, classes):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        imgPath = img_anns["imagePath"].split('/')[-1]
        filename = os.path.join(directory, imgPath)
        
        record["file_name"] = filename
        record["height"] = self.img_height
        record["width"] = self.img_width
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']] # x coord
            py = [a[1] for a in anno['points']] # y-coord
            poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                #"category_id": classes.index(anno['label']), #original
                "category_id": classes.index(anno['label'] if anno['label']=='panel' else 'panel'), #edited
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

# function to register custom dataset to detectron2 module
  def register_detectron(self, dataset_path):
    classes = ['panel']
    for d in ["train", "test"]:
      DatasetCatalog.register("category_" + d, lambda d=d: self.get_data_dicts(dataset_path+d, classes))
      MetadataCatalog.get("category_" + d).set(thing_classes=classes)
      
    microcontroller_metadata = MetadataCatalog.get("category_train")
    return classes, microcontroller_metadata

# function to build detectron2 model 
  def model_building(self):
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file("detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml")
    cfg.DATASETS.TRAIN = ("category_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl"
    cfg.SOLVER.IMS_PER_BATCH = self.img_per_batch
    cfg.SOLVER.BASE_LR = self.base_lr
    cfg.SOLVER.MAX_ITER = self.max_iter

    return cfg

# function to train detectron2 model
  def model_training(self, cfg, save_path):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train() 
    torch.save(trainer.model, save_path+'model_final.pth')

