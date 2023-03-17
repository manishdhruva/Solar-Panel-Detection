# importing dependencies
import torch
#assert torch.__version__.startswith("1.8") 
import torchvision
import cv2
from tqdm import tqdm
import os
import numpy as np
import json
import random
import math
import shutil

# importing detectron2 modules
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.projects import point_rend

# importing the configuration class containing all menthods
from config import *

data_annot_path = './Temp' # path for images and their respective annotations
save_weights_path = './model_weights_final.pth' # path for saving the model weights

# creating object for config class and defining parameters(image_height, image_width, learning rate, iterations, images per batch)
ob1 = config(512, 640, 0.00025, 1000, 2) 

# spliting input dataset into train and test folders
dataset_path = ob1.traintestsplit(data_annot_path)

# registering consolidated dataset to detectron2 module
classes, microcontroller_metadata = ob1.register_detectron(dataset_path)

# building pointrend detectron model for the custom dataset
cfg = ob1.model_building()

# training custom model and saving weights
ob1.model_training(cfg, save_weights_path)
