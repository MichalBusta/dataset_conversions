'''
Created on May 9, 2017

@author: busta
'''

import xml.sax
import os
import cv2
import math

from shutil import copyfile


import xml.dom.minidom


import numpy as np

class GTHandler(xml.sax.ContentHandler):
  def __init__(self):
    self.currentData = ""
    self.miny = 99999999
    self.maxy = 0
    self.insax = 0
    self.insaxv = 0
    self.minx = 99999999
    self.maxx = 0
    
    
  
  # Call when an element starts
  def startElement(self, tag, attributes):
    self.CurrentData = tag
    if tag == 'axis':
      if attributes.get('type') == u'h':
        self.insax = 1
      elif attributes.get('type') == u'v':
        self.insaxv = 1
      
      
  
  # Call when an elements ends
  def endElement(self, tag):
    if tag == 'axis':
      self.insax = 0
      self.insaxv = 0
  
  # Call when a character is read
  def characters(self, content):
    
    if self.CurrentData == "value":
      self.currentData += content
    elif self.insaxv and self.CurrentData == "x":
      x = int(float(content))
      if x < 0:
        return
      self.minx = min(self.minx, x)
      self.maxx = max(self.maxx, x)
      return
    elif not self.insax:
      return
    elif self.CurrentData == "y":
      y = int(float(content))
      if y < 0:
        return
      #print(y)
      self.miny = min(self.miny, y)
      self.maxy = max(self.maxy, y)
      
      

def to_coords(elem):
  return [float(elem.getAttribute('col')), float(elem.getAttribute('row'))]       

def walkdir(dirname, out_dir = '/tmp/nakladni4'):
  
  out_f = open('{0}/gt.txt'.format(dirname), 'w')
  out_f2 = open('{0}/gt_image.txt'.format(dirname), 'w')
  listf = open('/tmp/train.txt', 'w')
  
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  
  for cur, _dirs, files in os.walk(dirname):
    pref = ''
    head, tail = os.path.split(cur)
    while head != '/':
        pref += '---'
        head, _tail = os.path.split(head)
    print(pref+tail)
    if tail != 'lp':
      continue
    
    if not os.path.exists(cur + "_c"):
      os.mkdir(cur + "_c")
    
    for f in files:
      print(pref+'---'+f)
      image = cur + "/" + f
      gt_file = cur + '/xml/' + f 
      gt_file = gt_file.replace(".jpg", '.xml')
      gt_file = gt_file.replace(".png", '.xml')
      if os.path.exists(gt_file):
        
        # create an XMLReader
        parser = xml.sax.make_parser()
        # turn off namepsaces
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        
        # override the default ContextHandler
        Handler = GTHandler()
        parser.setContentHandler(Handler)
        
        parser.parse(gt_file)
        
        gt_txt = Handler.currentData
        
        img = cv2.imread(image)
        
        #miny = Handler.miny
        #maxy = Handler.maxy
        #minx = Handler.minx
        #maxx = Handler.maxx
        #img = img[(miny-4):(maxy+4), max(0, (minx-3)):(maxx+3)]
        
        if img.shape[0] > img.shape[1]:
          print('bad image!')
        
        
        image2 = cur + "_c/" + f
        
        
        cv2.imwrite(image2 , img)
        gt_txt = gt_txt.strip()
        gt_txt = gt_txt.replace("_", " ")
        out_f.write(u'{0} {1}\n'.format(image2, gt_txt).encode('utf-8'))
        full_image = image.replace("text", "images")
        full_image = image.replace("/lp", "")
        base_name = os.path.basename(full_image)
        if base_name.find("_") < 4:
          base_name = base_name[3:]
        elif not base_name.rfind("-lp") == -1:
          base_name = base_name[0:base_name.rfind("-lp")]
          base_name += ".jpg"
        dir_full = os.path.dirname(full_image)
        out_f2.write(u'{0}/{1} {2}\n'.format(dir_full, base_name, gt_txt).encode('utf-8'))
        
        imgo = cv2.imread('{0}/{1}'.format(dir_full, base_name))
        img_gt = '{0}/mgt/{1}'.format(dir_full, base_name)
        img_gt = img_gt.replace(".jpg", ".xml")
        img_gt = img_gt.replace(".png", ".xml")
        
        if os.path.exists(img_gt):
          parser2 = xml.sax.make_parser()
          # turn off namepsaces
          parser2.setFeature(xml.sax.handler.feature_namespaces, 0)
          
          # override the default ContextHandler
          dom = xml.dom.minidom.parse(img_gt)
          
          copyfile('{0}/{1}'.format(dir_full, base_name), '{0}/{1}'.format(out_dir, base_name))
          
          annfileName = '{0}/{1}'.format(out_dir, base_name)
          annfileName = annfileName.replace("jpg", "txt")
          annfile = open(annfileName, 'w')
          
          positions = dom.getElementsByTagName("position")
          
          norm = math.sqrt(imgo.shape[1] * imgo.shape[1] + imgo.shape[0] * imgo.shape[0])
          norm = 1.0 / norm
          dw = 1./imgo.shape[1]
          dh = 1./imgo.shape[0]
          
          for pos in positions:
            topleft = pos.getElementsByTagName("topleft")[0]
            topright = pos.getElementsByTagName("topright")[0]
            bottomright = pos.getElementsByTagName("bottomright")[0]
            bottomleft = pos.getElementsByTagName("bottomleft")[0]
            
            tl = to_coords(topleft)
            tr = to_coords(topright)
            br = to_coords(bottomright)
            bl = to_coords(bottomleft)
            
            width = math.sqrt((tr[0] - tl[0]) * (tr[0] - tl[0]) + (tr[1] - tl[1]) * (tr[1] - tl[1]))
            height = math.sqrt((tr[0] - br[0]) * (tr[0] - br[0]) + (tr[1] - br[1]) * (tr[1] - br[1]))
            
            cx = (tl[0] + tr[0] + br[0] + bl[0]) / 4.0
            cy = (tl[1] + tr[1] + br[1] + bl[1]) / 4.0
            
            angle = math.atan2(tr[1] - tl[1], tr[0] - tl[0])
          
            conv =  [cx, cy, width, height, angle]
            conv[0] *= dw
            conv[1] *= dh
            conv[2] *= norm
            conv[3] *= norm
          
            cls_id = -10
          
            annfile.write(str(cls_id) + " " + " ".join([str(a) for a in conv]) + " " + gt_txt.encode('utf-8') +  '\n')
          
          annfile.close()
          listf.write('{0}/{1}\n'.format(os.path.basename(out_dir), base_name)) 
          
          
        #cv2.imshow('ts', imgo)
        #cv2.waitKey(0)
            
  listf.close()
  out_f.close()          

if __name__ == '__main__':
  
  walkdir('/home/busta/data/eydea/20170606-nakladni/sggnss')
  
  
  
