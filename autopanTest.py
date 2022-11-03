import torch
import cv2
import numpy as np
import os
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import glob, os
import matplotlib.pyplot as plt
import pandas as pd
import csv

pathToSaveFrames = #EnterPathHere

pathToVideo = #EnterPathHere

pathToOutputVideo = #EnterPathHere

def getint(name):
    try:
        basename = name.partition('.')
        alpha, num = basename[0].split('_')
        return int(num)
    except ValueError as e:
        print(name)

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_increasing(L) or non_decreasing(L)

#initialize model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]

#get first frame from video
vidcap = cv2.VideoCapture(pathToVideo)
success,image = vidcap.read()
count = 0

#values for panning the camera. currentX: the x value that we are trying to center 
currentX = -1
frameWidth = 0

movingAvg = []
movingAvgListSize = 100

allXVals = []
timeVals = []

N = 10

lag = 90

while success:

  img_copy = cv2.imwrite(path + "frame_%d.jpg" % count, image) 

  penta = np.array([[245,345],[1055,345],[1220,435],[85,435]], np.int32)
  
  mask_value = 255

  stencil  = np.zeros(image.shape[:-1]).astype(np.uint8)
  cv2.fillPoly(stencil, [penta], mask_value)

  sel = stencil != mask_value
  image[sel] = 0

  results = model(image)

  for box in results.xyxy:

      if len(box) > 0:
          
          xmid = 0
          
          for i in range(len(box)):

              xmid = xmid + ((box[i][0] + box[i][2])/2)

          if currentX < 0:
              xmid = int(xmid / len(box))
              currentX = xmid
          else:
              xAvg = int(xmid / len(box))
              

              xmid = xAvg

          if len(movingAvg) > movingAvgListSize:
            movingAvg.insert(0, xmid)
            movingAvg.pop()
          else:
            movingAvg.append(xmid)

          xmid = int(sum(movingAvg) / len(movingAvg))
          print(xmid)

          currentX = xmid

          allXVals.append(xmid)
          timeVals.append(count)

          if count >= lag:
            img_copy = cv2.imread(path + "frame_%d.jpg" % int(count - lag))
            (height, width, D) = img_copy.shape
            frameWidth = width
            y = int(height/2)
            deltaY = int(height / 6)
            deltaX = int(width / 6)
            
            img_copy = img_copy[y - deltaY : y + deltaY, xmid - deltaX : xmid + deltaX]
            cv2.imwrite(path + "frame_%d.jpg" % int(count - lag), img_copy)

      else:
          if count >= lag:
            img_copy = cv2.imread(path + "frame_%d.jpg" % int(count - lag))
            (height, width, D) = img_copy.shape
            y = int(height/2)
            deltaY = int(height / 6)
            deltaX = int(width / 6)
            img_copy = img_copy[y - deltaY : y + deltaY, currentX - deltaX : currentX + deltaX]
            cv2.imwrite(path + "frame_%d.jpg" % int(count - lag), img_copy)
      
  success,image = vidcap.read()
  count += 1

  
dir_list = [i for i in os.listdir(path) if not i.startswith(('.','~','#'))]
dir_list.sort(key=getint)
height,width,layers=cv2.imread(path+dir_list[0]).shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter(pathToOutputVideo, fourcc, 30, (width, height))


for i in range(len(dir_list)):
  img = cv2.imread(path+dir_list[i])
  video.write(img)

