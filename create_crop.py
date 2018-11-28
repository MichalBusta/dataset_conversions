from PIL import Image
import glob
import os
from os import path
import numpy as np
import scipy.misc
import sys
#sys.path.insert(0, '/mnt/textspotter/software/opencv/ReleaseP3/lib/python3')
import cv2
import math
import shutil
from collections import OrderedDict
import random

codec = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~£ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž'

f = open('/home/busta/git/DeepSemanticText/codec.txt', 'r')
codec = f.readlines()[0]
f.close()

clist = OrderedDict()
for c in codec:
  clist[c] = 1


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def main(dirname, Fname):
  chyba = list()
  os.makedirs((dirname+'/'+Fname), exist_ok=True)
  out_txt = open((dirname+'/'+Fname+'/train_list.txt'), 'w')
  for file in glob.glob(path.join(dirname,"*.jpg")):
    name = path.splitext(path.basename(file))[0]
    try:
      with Image.open(file,'r') as Obrazek:
        img = np.asarray(Obrazek)
        
        print('Na rade ' + name, end='')
        with open(dirname+'/gt_'+name+'.txt', "r", encoding='utf8') as InText:
          content = InText.readlines()
          content[0] = content[0].replace('\ufeff', '')
          content = [x.strip() for x in content]
          for ind, x in enumerate(content):
            x=x.split(",")
            if x[-1] == '###':
              continue
            
            gt_txt = x[-1].strip('"')
            gt_txt = gt_txt.strip()
            if len(gt_txt) == 0:
              continue
            
            if gt_txt[0] == '#':
              continue
            
            for c in x[-1]:
              if not c in clist:
                clist[c] = 1
            
            
            CL = [int(c) for c in x[:8]]
            # CL = list(map(int,x[:8]))
            x=[CL[0],CL[2],CL[4],CL[6]]
            y=[CL[1],CL[3],CL[5],CL[7]]
            
            min_x = min(x)
            max_x = max(x)
            min_y = min(y)
            if min_y > 10:
              min_y -= random.randint(0, 10)
            max_y = max(y)
            if max_y + 10 < img.shape[0]:
              max_y += random.randint(0, 10)
            
            height = max_y - min_y
            
            min_x -= random.randint(0, int(height / 3))
            max_x += random.randint(0, int(height / 3))
            
            if ( max_x - min_x) < ( max_y - min_y): 
              print(' ---susp image')
              continue
            #out_txt.write('{0}, {1}\n'.format(Fname+'/'+name+'_'+str(ind)+'_'+'B.jpg', gt_txt))
            #scipy.misc.imsave(dirname+'/'+Fname+'/'+name+'_'+str(ind)+'_'+'B.jpg', Image.open(file,'r').crop((min_x, min_y, max_x, max_y)))
            
            CL[0] -= random.randint(0, int(height / 3))
            CL[2] +=  random.randint(0, int(height / 3))
            CL[1] -= random.randint(0, 10)
            CL[3] -= random.randint(0, 10)
            CL[5] += random.randint(0, 10)
            
            height =math.floor(distance((CL[2],CL[3]),(CL[4],CL[5])))
            width= math.floor(distance((CL[4],CL[5]),(CL[6],CL[7])))
            aff = cv2.getAffineTransform(np.array([[CL[0],CL[1]],[CL[2],CL[3]],[CL[4],CL[5]]], dtype="float32"), np.array([[0,0],[width,0],[width,height]], dtype="float32"))
            imgcrop = cv2.warpAffine(img, aff, (int(width), int(height)), borderMode=cv2.BORDER_REPLICATE)
            scipy.misc.imsave(dirname+'/'+Fname+'/'+ name+'_'+str(ind)+'_'+'A.jpg', imgcrop)
            
            out_txt.write('{0}, {1}\n'.format(Fname+'/'+name+'_'+str(ind)+'_'+'A.jpg', gt_txt))
            out_txt.flush()
            
          print(' -> done ')
    except:
        shutil.rmtree(dirname+'/'+Fname+'/'+name, ignore_errors=True, onerror=None)
        print(' -> chyba!')
        chyba.append(name)
  print('Done, chyby v: '+' '.join([str(c) for c in chyba]))
  
  out_codec = open((dirname+'/'+Fname+'/codec_list.txt'), 'w')
  for key in clist:
    out_codec.write(key)
    
  out_codec.close()
  out_txt.close()
  return

import unicodedata as ud

def is_in_alphabet(uchr, alphabet):      
  return alphabet in ud.name(uchr)

def main_bangla(dirname, Fname):
  chyba = list()
  os.makedirs((dirname+'/'+Fname), exist_ok=True)
  out_txt = open((dirname+'/'+Fname+'/train_list_bangla.txt'), 'w')
  for file in sorted(glob.glob(path.join(dirname,"*.jpg")), reverse=True):
    name = path.splitext(path.basename(file))[0]
    try:
      with Image.open(file,'r') as Obrazek:
        img = np.asarray(Obrazek)
        
        print('Na rade ' + name, end='')
        with open(dirname+'/gt_'+name+'.txt', "r", encoding='utf8') as InText:
        #with open('/home/busta/data/SynthText/icdar_ch8/Train/gt_img_6431.txt') as InText:
          content = InText.readlines()
          content = [x.strip() for x in content]
          for ind, x in enumerate(content):
            x=x.split(",")
            if x[-1] == '###':
              continue
            
            gt_txt = x[-1].strip('"')
            gt_txt = gt_txt.strip()
            if len(gt_txt) == 0:
              continue
            
            
            for c in x[-1]:
              if not c in clist:
                clist[c] = 1
            
            
            CL = [int(c) for c in x[:8]]
            # CL = list(map(int,x[:8]))
            x=[CL[0],CL[2],CL[4],CL[6]]
            y=[CL[1],CL[3],CL[5],CL[7]]
            
            min_x = min(x)
            max_x = max(x)
            min_y = min(y)
            if min_y > 10:
              min_y -= random.randint(0, 10)
            max_y = max(y)
            if max_y + 10 < img.shape[0]:
              max_y += random.randint(0, 10)
            
            height = max_y - min_y
            
            min_x -= random.randint(0, int(height / 3))
            max_x += random.randint(0, int(height / 3))
            
            if ( max_x - min_x) < ( max_y - min_y): 
              print(' ---susp image')
              continue
            out_txt.write('{0}, {1}\n'.format(Fname+'/'+name+'_'+str(ind)+'_'+'B.jpg', gt_txt))

            scipy.misc.imsave(dirname+'/'+Fname+'/'+name+'_'+str(ind)+'_'+'B.jpg', Image.open(file,'r').crop((min_x, min_y, max_x, max_y)))
            
            CL[0] -= random.randint(0, int(height / 3))
            CL[2] +=  random.randint(0, int(height / 3))
            CL[1] -= random.randint(0, 10)
            CL[3] -= random.randint(0, 10)
            CL[5] += random.randint(0, 10)
            
            height =math.floor(distance((CL[2],CL[3]),(CL[4],CL[5])))
            width= math.floor(distance((CL[4],CL[5]),(CL[6],CL[7])))
            aff = cv2.getAffineTransform(np.array([[CL[0],CL[1]],[CL[2],CL[3]],[CL[4],CL[5]]], dtype="float32"), np.array([[0,0],[width,0],[width,height]], dtype="float32"))
            imgcrop = cv2.warpAffine(img, aff, (int(width), int(height)), borderMode=cv2.BORDER_REPLICATE)
            scipy.misc.imsave(dirname+'/'+Fname+'/'+ name+'_'+str(ind)+'_'+'A.jpg', imgcrop)
            
            out_txt.write('{0}, {1}\n'.format(Fname+'/'+name+'_'+str(ind)+'_'+'A.jpg', gt_txt))
            out_txt.flush()
          print(' -> done ')
    except:
      import traceback
      traceback.print_exc()  
      shutil.rmtree(dirname+'/'+Fname+'/'+name, ignore_errors=True, onerror=None)
      print(' -> chyba!')
      chyba.append(name)
  print('Done, chyby v: '+' '.join([str(c) for c in chyba]))
  
  out_codec = open((dirname+'/'+Fname+'/codec_list.txt'), 'w')
  for key in clist:
    out_codec.write(key)
    
  out_codec.close()
  out_txt.close()
  return

def main2(dirname, Fname):
  chyba = list()
  cnt = 0
  for file in glob.glob(path.join(dirname,"*.jpg")):
    name = path.splitext(path.basename(file))[0]
    try:
      cnt += 1
      
      print('Na rade ' + name + " " + str(cnt + 1))
      with open(dirname+'/gt_'+name+'.txt', "r", encoding='utf8') as InText:
        content = InText.readlines()
        content = [x.strip() for x in content]
        for ind, x in enumerate(content):
          x=x.split(",")
          if x[-1] == '###':
            continue
          
          gt_txt = x[-1].strip('"')
          
          
          for c in x[-1]:
            if not c in clist:
              clist[c] = 1
    except:
        shutil.rmtree(dirname+'/'+Fname+'/'+name, ignore_errors=True, onerror=None)
        print(' -> chyba!')
        chyba.append(name)
  print('Done, chyby v: '+' '.join([str(c) for c in chyba]))
  
  out_codec = open((dirname+'/codec_list.txt'), 'w')
  for key in clist:
    out_codec.write(key)
    
  print(len(clist))
  out_codec.close()
  return


def main3(dirname):
  cnt = 0
  f = open('/mnt/textspotter/tmp/90kDICT32px/train_mlt_synth.txt', 'r')
  #f = open('/home/busta/data/90kDICT32px/train_icdar_ch8.txt')
  while True:
    line = f.readline()
    if not line: 
      break
    line = line.strip()
    if len(line) == 0:
      continue
    
    
    spl = line.split(" ")
    if len(spl) == 1:
      spl = line.split(",")
    
    gt_txt = ''
    if len(spl) > 1:
      gt_txt = ""
      delim = ""
      for k in range(1, len(spl)):
        gt_txt += delim + spl[k]
        delim =" "
      if len(gt_txt) > 0 and gt_txt[0] == '"':
          gt_txt = gt_txt[1:-1]
      if len(gt_txt) == 0:
        print('!')
    
    gt_txt = gt_txt.replace('`', "'") 
    gt_txt = gt_txt.replace('´', "'") 
    for c in gt_txt:
      if not c in clist:
        print("New codec value: {0}".format(c))
        clist[c] = 1
    
  out_codec = open((dirname+'/codec_list.txt'), 'w')
  for key in clist:
    if key not in codec:
      out_codec.write(key)
    
  print(len(clist))
  out_codec.close()
  return

if __name__ == '__main__':
  
  main('/home/busta/data/icdar2017rctw_train/train','icdar_rctw')
  #main3('/home/busta/data/SynthText')
    
    