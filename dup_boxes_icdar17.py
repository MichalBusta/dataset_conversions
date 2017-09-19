'''
Created on Sep 12, 2016

@author: Michal.Busta at gmail.com
'''

import glob
import unicodecsv
import codecs
import os

import cv2
import math
import copy

import numpy as np
from csv import reader

inputDir = '/home/busta/data/icdar2017rctw_train/train'
#inputDir = '/mnt/textspotter/evaluation-sets/icdar2013-Train'
#inputDir = '/home/busta/data/icdar2013-Train'
#inputDir = '/home/busta/data/icdar2013-Test'


def read_icdar2015_txt_gt(gt_file, separator = ' '):
    
    
  f = codecs.open(gt_file, "rb", "utf-8-sig")
  lines = f.readlines()
  
  
  gt_rectangles = []
  for line in lines:
    if line[0] == '#':
      continue
    splitLine = line.split(separator);
    if len(splitLine) < 8:
      continue
    xline = u'{0}'.format(line.strip())
    xline = xline.encode('utf-8') 
    
    for splitLine in unicodecsv.reader([xline], skipinitialspace=True, quotechar='"', delimiter=separator, encoding='utf-8'):
      break
    
    x1 = int(float(splitLine[0].strip()))
    x2 = int(float(splitLine[2].strip()))
    x3 = int(float(splitLine[4].strip()))
    x4 = int(float(splitLine[6].strip()))
   
    y1 = int(float(splitLine[1].strip()))
    y2 = int(float(splitLine[3].strip()))
    y3 = int(float(splitLine[5].strip()))
    y4 =  int(float(splitLine[7].strip()))
          
    gt_rectangles.append( [x1, y1, x2, y2, x3, y3, x4, y4, splitLine[9].strip() ] )
          
  if len(gt_rectangles) == 0:
    raise ValueError()
  return gt_rectangles    

def is_chinese(name):
    
  for ch in name:
    if ord(ch) > 0x4e00 and ord(ch) < 0x9fff:
      return True
  return False

if __name__ == '__main__':
  id2 = inputDir
  images = glob.glob('{0}/*.jpg'.format(id2))
  images2 = glob.glob('{0}/*.JPG'.format(id2))
  images.extend(images2)
  images2 = glob.glob('{0}/*.png'.format(id2))
  images.extend(images2) 
  imageNo = 0 
  
  boxes_all = []
  
  list_file = open('/home/busta/data/test_icdar2017.txt', 'w')   
  
  for image_name in sorted(images):
      
    imageNo += 1
    #if imageNo < 100:
    #    continue
    print(image_name)
    im = cv2.imread(image_name)
    
    #dw = 1 / math.sqrt(im.shape[1] * im.shape[1] + im.shape[0]* im.shape[0]) 
    dw = 1./im.shape[1]
    dh = 1./im.shape[0]
    #dh = dw
    
    baseName = os.path.basename(image_name)
    
    baseName = baseName[:-4]
    #baseName = baseName.replace("_", "")
    lineGt = '{0}/gt_{1}.txt'.format(inputDir, baseName)
    
    if os.path.exists(lineGt):
        
      annfileName = image_name.replace("images", "labels")
      annfileName = annfileName.replace("jpg", "txt")
      annfile = open(annfileName, 'w')
      
      
      gt_rect = read_icdar2015_txt_gt(lineGt, separator = ',')
      for rect in gt_rect:
        cv2.line(im, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0) )
        cv2.line(im, (rect[2], rect[3]), (rect[4], rect[5]), (0, 255, 0) )
        width = math.hypot(rect[2] - rect[0], rect[3] - rect[1])
        height = math.hypot(rect[2] - rect[4], rect[3] - rect[5])
        x = (rect[0] + rect[2] + rect[4] + rect[6]) / 4.0 
        y = (rect[1] + rect[3] + rect[5] + rect[7]) / 4.0
        
        angle = math.atan2((rect[3] - rect[1]), (rect[2] - rect[0]))
        
        maxx = max(rect[0], max(rect[2], max(rect[4],rect[6])))
        minx = min(rect[0], min(rect[2], min(rect[4],rect[6])))
        
        maxy = max(rect[1], max(rect[3], max (rect[5], rect[7])))
        miny = min(rect[1], min(rect[3], min (rect[5], rect[7])))
        
        norm = math.sqrt(im.shape[1] * im.shape[1] + im.shape[0] * im.shape[0])
        norm = 1.0 / norm
        
        conv =  [x, y, width, height, angle]
        conv[0] *= dw
        conv[1] *= dh
        conv[2] *= norm
        conv[3] *= norm
        
        cls_id = 0
        if is_chinese(rect[8]):
          cls_id = 1
        else:
          print(rect[8])
            
        
        annfile.write(str(cls_id) + " " + " ".join([str(a) for a in conv]) + " " + rect[8].encode('utf-8') +  '\n')
        
        conv2 = copy.copy(conv)
        conv2[0] = - conv2[2] / 2
        conv2[1] = - conv2[3] / 2 
        conv2[2] = conv2[2] / 2
        conv2[3] = conv2[3] / 2 
        
        boxes_all.append(conv2)
       
      if im.shape[1] > 1042:
        im = cv2.resize(im, (im.shape[1] / 2, im.shape[0] / 2))          
      cv2.imshow("ts", im)
      key = cv2.waitKey(0)
      if key == ord('g'):    
        list_file.write('{0}\n'.format(image_name))
      annfile.close()
        
        
        #cv2.imshow("ts", im)
        #cv2.waitKey(0)
  
  #np.savez('/home/busta/data/icdar2015-Ch4-Train/voc_boxes.npz', box_list = boxes_all)
  np.savez('/home/busta/data/icdar2013-Train/voc_boxes.npz', box_list = boxes_all)         
    
    
    