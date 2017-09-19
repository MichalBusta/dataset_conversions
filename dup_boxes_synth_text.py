'''
Created on Mar 4, 2016

@author: busta
'''

import sys
#from cryptography.hazmat.primitives.asymmetric import dh
sys.path.append("/usr/local/python")

import os

import cv2
import numpy as np
from PIL import Image

import scipy.io
import math
import copy

dataDir = '/home/busta/data/COCO'
dataType='train2014'

classes = ["illegible", "legible"]

write_ann = True

def convert(size, box):
    dw = 1./size[1]
    dh = 1./size[0]
    x = box[0] + box[2]/2.0
    y = box[2] + box[3]/2.0
    w = box[2]
    h = box[3]
    x = x*dw
    w = w*dw
    y = y*dw
    h = h*dw
    return (x,y,w,h)

def convert2(size, box):
    dw = 1./size[1]
    dh = 1./size[1]
    x = box[0] + box[2]/2.0
    y = box[2] + box[3]/2.0
    w = box[2]
    h = box[3]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    x = - w / 2
    y = - h / 2
    return (x,y,w / 2,h / 2)

if __name__ == '__main__':
    
    
    imnames = np.load('/mnt/textspotter/tmp/SynthText/imnames.np.npy')[0]
    
    wordBB = np.load('/mnt/textspotter/tmp/SynthText/wordBB.np.npy')[0]
    txt = np.load('/mnt/textspotter/tmp/SynthText/gt_txt.npz')['txt'][0]

    boxes_all = [] 
    
    list_file = open('/mnt/textspotter/tmp/SynthText/train2.txt', 'w')   
    
    for i in range(imnames.shape[0]):
        img = imnames[i]
        bbs = wordBB[i]
        trans = txt[i]  
        texts = []
        for t in trans:
            line_split  = t.split("\n")
            for spl in line_split:
                spl = spl.strip()
                s2 = spl.split(" ")
                for s in s2:
                    texts.append(s.strip())
        
        if img[0] != u'129/photos_4_48.jpg':
            continue     
        imageName = u'/mnt/textspotter/tmp/SynthText/{0}'.format(img[0])
       
        ii = cv2.imread(imageName, cv2.IMREAD_COLOR)
        annfileName = imageName.replace(" ", "")
        annfileName = annfileName.replace("jpg", "txt")
        annDir = os.path.dirname(annfileName)
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        
        if i % 1000 == 0:
            print(i)
            print(imageName)
        
        #annfileName = imageName[:-3] + "txt"
        
        if len(bbs.shape) < 3:
            bbs = np.reshape(bbs, (bbs.shape[0], bbs.shape[1], 1))
            '''
            bbox0 = [bb[0, 0], bb[1, 0], bb[0, 1], bb[1, 1], bb[0, 2], bb[1, 2], bb[0, 3], bb[1, 3]]
            
            cv2.line(ii, (int(bbox0[0]), int(bbox0[1])), (int(bbox0[2]), int(bbox0[3])), (0, 255, 0))
            cv2.line(ii, (int(bbox0[2]), int(bbox0[3])), (int(bbox0[4]), int(bbox0[5])), (0, 255, 255))
            cv2.imshow("ts", ii)
            cv2.waitKey()
            
            continue
            '''
        
        if write_ann:
            repeat = True
            while repeat:
                try:
                    annfile = open(annfileName, 'w')
                    repeat = False
                except:
                    pass
            
        
        tc = 0
        for j in range(bbs.shape[2]):
            bb = bbs[:, :, j]
            tt = texts[tc]
            while tt == '':
                tc += 1
                tt = texts[tc]        
             
            bbox0 = [bb[0, 0], bb[1, 0], bb[0, 1], bb[1, 1], bb[0, 2], bb[1, 2], bb[0, 3], bb[1, 3]]
            
            x = (bb[0, 0] + bb[0, 1] + bb[0, 2] + bb[0, 3]) / 4.0
            y = (bb[1, 0] + bb[1, 1] + bb[1, 2] + bb[1, 3]) / 4.0
            
            dw1 = bbox0[2] - bbox0[0]
            dw2 = bbox0[3] - bbox0[1]
            
            w = math.sqrt(dw1 * dw1 + dw2 * dw2)
            
            dh1 = bbox0[4] - bbox0[2]
            dh2 = bbox0[5] - bbox0[3]
            h = math.sqrt(dh1 * dh1 + dh2 * dh2)
            
            '''
            minx = min(bb[0, 0], bb[0, 1])
            minx2 = min(bb[0, 2], bb[0, 3])
            minx = min(minx, minx2)
            
            maxx = max(bb[0, 0], bb[0, 1])
            maxx2 = max(bb[0, 2], bb[0, 3])
            maxx = max(maxx, maxx2)
            
            w = maxx - minx
            
            miny = min(bb[1, 0], bb[1, 1])
            miny2 = min(bb[1, 2], bb[1, 3])
            miny = min(miny, miny2) 
            
            maxy = max(bb[1, 0], bb[1, 1])
            maxy2 = max(bb[1, 2], bb[1, 3])
            maxy = max(maxy, maxy2)
                      
            h = maxy - miny
            '''
            
            angle = math.atan2((bb[1, 1] - bb[1, 0]), (bb[0, 1] - bb[0, 0]))
            #print( 180 * angle / 3.14 )
            
            bbox = [x, y, w, h, angle, tt]
                 
            cls_id = 0
            
            
                
            norm = math.sqrt(ii.shape[1] * ii.shape[1] + ii.shape[0] * ii.shape[0])
            norm = 1.0 / norm
            
            dw = 1./ii.shape[1]
            dh = 1./ii.shape[0]
            conv =  copy.copy(bbox)
            conv[0] *= dw
            conv[1] *= dh
            conv[2] *= norm
            conv[3] *= norm
            
            
            #conv = convert(ii.shape, bbox)
            conv2 = copy.copy(conv)
            conv2[0] = - conv2[2] / 2
            conv2[1] = - conv2[3] / 2 
            
            
            if False: #conv[2] <= 0 or conv[3] <= 0:
                print(tt)
                cv2.line(ii, (int(bbox0[0]), int(bbox0[1])), (int(bbox0[2]), int(bbox0[3])), (0, 255, 0))
                cv2.line(ii, (int(bbox0[2]), int(bbox0[3])), (int(bbox0[4]), int(bbox0[5])), (0, 255, 255))
                cv2.circle(ii, (int(x), int(y)), 5, (0, 255, 0))
                cv2.imshow("ts", ii)
                cv2.waitKey()
            else:
                hasText = True
                boxes_all.append(conv2)
                if write_ann:
                    stw = str(cls_id) + " " + " ".join([str(a) for a in conv]) + '\n'
                    annfile.write(stw.encode('utf8'))
                #annfile.write('{0},{1},{2},{3},"{4}"\n'.format(int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]), text ))
            tc += 1
            
        if write_ann:
            annfile.close()
        list_file.write('{0}\n'.format(imageName))
        list_file.flush()
            
        #cv2.imshow("ts", ii)
        #cv2.waitKey()
    list_file.close()
    np.savez('/mnt/textspotter/tmp/SynthText/voc_boxes.npz', box_list = boxes_all) 
    
    
    