from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# In[ ]:

image_trains = glob.glob("../train/*.jpg")
image_test = glob.glob("../test/*.jpg")

dim = (640,480)
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

def crop_image(img, xmin, ymin, xmax, ymax):
    crop = img[ymin:ymax, xmin:xmax]
    return crop

for img_path in image_trains:
    im = plt.imread(img_path)
    im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    # cv2.imshow("im",im)
    # cv2.waitKey(0)
    outputs = predictor(im)

    classes = outputs["instances"].pred_classes
    boxes = outputs["instances"].pred_boxes

    nx = classes.cpu().numpy()
    index = -1
    try:
        index = np.where(nx==16)[0][0]
        print(index)
    except:
        print("no dog detected, remaining the image")
    if index == -1:
        cv2.imwrite(f"./train/{img_path}", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    else:
        bb = []
        count = 0
        for i in boxes.__iter__():
            if count != index:
                count += 1
                continue
            bb = i.cpu().numpy()
            break
        # crop the bounding box of the dog, if dog detected
        int_array = bb. astype(int)
        cropped = crop_image(im, int_array[0],int_array[1],int_array[2],int_array[3])
        cv2.imwrite(f"./train/{img_path}", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))


for img_path in image_test:
    im = plt.imread(img_path)
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    # cv2.imshow("im",im)
    # cv2.waitKey(0)
    outputs = predictor(im)

    classes = outputs["instances"].pred_classes
    boxes = outputs["instances"].pred_boxes

    nx = classes.cpu().numpy()
    index = -1
    try:
        index = np.where(nx==16)[0][0]
        print(index)
    except:
        print("no dog detected, remaining the image")
    if index == -1:
        cv2.imwrite(f"./test/{img_path}", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    else:
        bb = []
        count = 0
        for i in boxes.__iter__():
            if count != index:
                count += 1
                continue
            bb = i.cpu().numpy()
            break
        # crop the bounding box of the dog, if dog detected
        int_array = bb. astype(int)
        cropped = crop_image(im, int_array[0],int_array[1],int_array[2],int_array[3])
        cv2.imwrite(f"./test/{img_path}", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))


