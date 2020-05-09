from __future__ import division
import torchvision
import cv2
import torch
import numpy
import torch.nn as nn




print('CUDA available: {}'.format(torch.cuda.is_available()))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


##CLASS NAMES IN COCO MODEL
COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

threshold = 0.5


def downloadModel():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()
    return model


def getpredictions(image):
    
    frame = cv2.imread(image)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
    img = transform(frame) 
    newModel = downloadModel()
    pred = newModel([img])
    
    for i in list(pred[0]['labels'].numpy()):
            pred_class = [COCO_NAMES[i]]
            print(pred_class)

    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
    pred_score = list(pred[0]['scores'].detach().numpy())
    
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    
    return pred_boxes

