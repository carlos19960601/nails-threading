# coding=utf-8

import sys
import time
from multiprocessing import Pool as ThreadPool

import numpy
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageQt
from PySide6 import QtCore, QtGui, QtWidgets
from scipy import ndimage

from PointCloud import PointCloud


class Viewer(QtWidgets.QMainWindow):
  def __init__(self, parameters):
    super(Viewer, self).__init__()

    self.parameters = parameters
    self.multiWidget = QtWidgets.QWidget()

    self.imageLabel = QtWidgets.QLabel(self.multiWidget)
    self.imageLabel.setBackgroundRole(QtGui.QPalette.ColorRole.Base)
    self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, 
                                  QtWidgets.QSizePolicy.Policy.Minimum)
    self.imageLabel.setScaledContents(False)
    self.imageLabel.setStyleSheet("border:0px")
    self.imageLabel.setContentsMargins(0,0,0,0)

    self.imageLabel2 = QtWidgets.QLabel(self.multiWidget)
    self.imageLabel2.setBackgroundRole(QtGui.QPalette.ColorRole.Base)
    self.imageLabel2.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, 
                                  QtWidgets.QSizePolicy.Policy.Minimum)
    self.imageLabel2.setScaledContents(False)
    self.imageLabel2.setStyleSheet("border:0px")
    self.imageLabel2.setContentsMargins(0,0,0,0)

    self.imageLabel3 = QtWidgets.QLabel(self.multiWidget)
    self.imageLabel3.setBackgroundRole(QtGui.QPalette.ColorRole.Base)
    self.imageLabel3.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, 
                                  QtWidgets.QSizePolicy.Policy.Minimum)
    self.imageLabel3.setScaledContents(False)
    self.imageLabel3.setStyleSheet("border:0px")
    self.imageLabel3.setContentsMargins(0,0,0,0)

    self.bl = QtWidgets.QVBoxLayout(self.multiWidget)
    self.bl.addWidget(self.imageLabel)
    self.bl.addWidget(self.imageLabel2)
    self.bl.addWidget(self.imageLabel3)

    self.scrollArea = QtWidgets.QScrollArea()
    self.scrollArea.setBackgroundRole(QtGui.QPalette.ColorRole.Dark)
    self.scrollArea.setWidget(self.multiWidget)
    self.setCentralWidget(self.scrollArea)
    self.scrollArea.setLayout(self.bl)

    self.mode ="ProcessImage"
    self.setWindowTitle("NailedIt - "+self.mode)
    self.resize(parameters["proc_width"]+50, parameters["proc_height"]*2+50)

    self.avg_improvement = -2*parameters["proc_width"]

    self.segmentCount = {}
    self.string_path = []
    self.string_length = 0
    self.iterationCounter = 0
    self.currentWidth = 1
    self.threadCol = [self.parameters["threadColor"][0]/255.0, self.parameters["threadColor"][1]/255.0]

    self.targetImage = self.parameters["TargetImage"]
    self.np_targetArray = PIL_to_array(self.targetImage)

    self.parameters["CurrentImage"] = numpy.array(Image.new("L", self.targetImage.size, parameters["backgroundColor"]), dtype="float32")
    self.residual = image_diff(self.parameters["CurrentImage"], self.np_targetArray)

    self.threadpool = ThreadPool()
    self.lastTime = time.time()

    self.timer = QtCore.QTimer(self)
    self.timer.timeout
    self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.workImage)
    self.timer.setSingleShot(True)
    self.timer.start(2000)

  def workImage(self):
    targetImage = self.parameters["TargetImage"]
    if not "DetailImage" in self.parameters:
      self.setWindowTitle("NailedIt - Detail Image")
      # 从这 https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_gradient_magnitude.html
      # 看这个函数处理图片的效果
      gradmag = ndimage.gaussian_gradient_magnitude(self.np_targetArray, 3)
      gradmag = gradmag / gradmag.max()
      self.parameters["DetailImage"] = gradmag
      
      self.showImage(targetImage)
      self.showImage(gradmag, slot=1)
      self.timer.start(1000)

    elif not "EdgesImage" in self.parameters:
      if "edgesImagePath" in self.parameters:
        img = Image.open(self.parameters["edgesImagePath"])
      else:
        self.setWindowTitle("NailedIt - Edges Image")
        gradmag = ndimage.gaussian_gradient_magnitude(self.np_targetArray, 1.5)
        gradmag = gradmag / gradmag.max()
        self.parameters["EdgesImage"] = gradmag
  
      self.showImage(self.parameters["EdgesImage"], slot=1)
      self.timer.start(1000)
    
    else:
      npt = ndimage.gaussian_filter(self.np_targetArray, self.parameters["blurAmount"])
      self.blurredTarget = npt
      self.showImage(self.blurredTarget, slot=1)

      self.disconnect(self.timer, QtCore.SIGNAL("timeout()"), self.workImage)
      self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
      self.timer.start(10)
      self.mode = "ProcessPoints"
      self.setWindowTitle("NailedIt - "+self.mode)

  def workPoints(self):
    currentImage = self.parameters["CurrentImage"]
    currentImage[:] = self.parameters["backgroundColor"]
    img = array_to_PIL_rgb(currentImage)
    
    minDist = self.parameters["nailDistMin"]
    maxDist = self.parameters["nailDistMax"]
    img_w = self.parameters["proc_width"]
    img_h = self.parameters["proc_height"]

    if "PointCloud" in self.parameters:
      pnts = self.parameters["PointCloud"]
    else:
      pc = PointCloud(img_w, img_h)
      self.parameters["PointCloud"] = pc

      img = array_to_PIL_rgb(self.parameters["EdgesImage"])
      scat_start = len(pc.p) - 1
      # 随机选取线上的点，并将ignore设置成True
      pc.scatterOnMask(self.parameters["EdgesImage"], (img_w*img_h)/(minDist**2), minDist, threshold=self.parameters["edgeThreshold"])
      for pid in range(scat_start, len(pc.p)):
        pc.p[pid].ignore = True

      # grid
      numx = int(img_w/maxDist)
      numy = int(img_h/maxDist)
      if numy % 2 == 0:
        numx += 1
        numy += 1

      gridstart = len(pc.p)
      pc.addGrid(numx, numy)
      for i in range(0, numy, 2):
        lineStart = gridstart + i*numx-i//2
        pc.p[lineStart].heat = 1.0
        pc.p[lineStart + (numx-1)].heat = 1.0
      for i in range(numx):
        pc.p[gridstart+i].heat = 1.0
        pc.p[gridstart + i + (numy-1)*numx-((numy-1)//2)].heat = 1.0

      #random points
      pc.addRandom(int(numx * numy * 0.3))

      draw_points(img, pc, 3)
      self.showImage(img, slot=1)
      self.timer.start(5000)
      return
  
    if self.iterationCounter < 50:
      pnts.relax(img, 10, self.parameters["DetailImage"], minDist, maxDist)
      draw_points(img, pnts)
      self.showImage(img)
      self.iterationCounter += 1
      self.timer.start(10)

    elif self.iterationCounter == 50:     
      draw = ImageDraw.Draw(img, "RGB")

      problems = [0]*len(pnts.p)
      for me,p in enumerate(pnts.p) :
        cps = pnts.closestPoints(p.x, p.y, minDist, me)
        if not p.ignore:
          if not p.heat: # not the edge
            if len(cps):
              problems[me] = len(cps)
        else: # a point from the edge mask can only be deleted if the rim of the grid is close
          problems[me] = sum([1 if pnts.p[c].heat else 0 for c in cps])

      numOffenders = 0
      for me, p in enumerate(pnts.p):
        bad = problems[me] > 0
        if bad:
          numOffenders += 1
        draw.rectangle((p.x-1, p.y-1, p.x+1, p.y+1), (255, 0, 0) if bad else (255,255,0))

      if numOffenders:
        # remove the offending point with the most neighbours
        problems = [ (prob, id) for id,prob in enumerate(problems)]
        problems.sort(reverse=True)
        del pnts.p[problems[0][1]]
        pnts.relax(None, 10, self.parameters["DetailImage"], minDist, maxDist)
      else:
        # last ditch check including all points (even the ones on edges)
        debug = 0
        for me,p in enumerate(pnts.p) :
          cps = pnts.closestPoints(p.x, p.y, minDist, me)
          cps = [cp for cp in cps if p.dist(pnts.p[cp]) < minDist]
          if len(cps):
            numOffenders +=1
            if p.ignore:
              debug += 1
        if numOffenders > 0:
          raise UserWarning
        self.iterationCounter += 1

      self.showImage(img)
      self.timer.start(10)
      
    elif self.iterationCounter == 51:
      start = self.parameters["start_at"]
      self.parameters["currentPoint"] = pnts.closestPoint(float(start[0])*img_w, float(start[1]*img_h))[0]
      self.string_path.append(self.parameters["currentPoint"])
      pnts.heat(0)
      for p in pnts.p:
        p.neighbors = None
        p.ignore = False

      self.disconnect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
      self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.March)
      self.iterationCounter = 0
      currentImage[:] = self.parameters["backgroundColor"]
      self.mode = "Threading"
      self.setWindowTitle("NailedIt - " + self.mode)
      self.timer.start(10)

    else:
      self.timer.start(10)  

  def March(self):
    beauty_image = self.parameters["BeautyImage"] if "BeautyImage" in self.parameters else Image.new("RGB", self.targetImage.size, self.parameters["backgroundColor"])
    beauty_image2 = self.parameters["BeautyImage2"] if "BeautyImage2" in self.parameters else Image.new("RGB", self.targetImage.size, self.parameters["backgroundColor"])
    currentImage = self.parameters["CurrentImage"]

    pnts = self.parameters["PointCloud"]
    current_point_idx = self.parameters["currentPoint"]
    last_point_idx = self.parameters["lastPoint"]
    maxConnects = self.parameters["maxConnectsPerNail"]
    blurAmount = self.parameters["blurAmount"]
    searchRadius = self.parameters["searchRadius"]

    # test #######################################################
    if self.iterationCounter > 0 and not self.iterationCounter%300:
      self.parameters["blurAmount"] *= 0.9
      blurAmount = self.parameters["blurAmount"]
      npt = ndimage.gaussian_filter(self.np_targetArray, self.parameters["blurAmount"])
      self.blurredTarget = npt  # numpy.clip(npt, 0, self.currentDensity)/self.currentDensity
    #############################################################

    # find next best point

    # get most reasonable neightbors
    neighbours_idx = pnts.p[current_point_idx].neighbors
    if neighbours_idx is None:
      neighbours_idx = pnts.findNeighbours(current_point_idx, self.parameters["proc_width"]*searchRadius)
      remove_nail_collisions(pnts, current_point_idx, neighbours_idx, self.parameters["nailDiameter"]/2)
      pnts.p[current_point_idx].neighbors = neighbours_idx

    remove_saturated_segments(current_point_idx, neighbours_idx, self.segmentCount, self.parameters["maxSegmentConnect"])
    # check how good the neighbours are
    col = self.threadCol

    params = [(currentImage, pnts.p[current_point_idx], pnts.p[neighbour], self.np_targetArray, neighbour, col, self.residual, self.blurredTarget, self.currentWidth, blurAmount) for neighbour in neighbours_idx if neighbour != last_point_idx]
    candidates = self.threadpool.map(check_quality, params)

    # fish out the best match
    candidates.sort()

    jumped = False
    if not candidates or candidates[0][0] >= 0:
      nid = self.find_next_island(currentImage, pnts, current_point_idx, 0)
      candidates.append(check_quality( (currentImage, pnts.p[current_point_idx], pnts.p[nid], self.np_targetArray, nid, col, self.residual, self.blurredTarget, self.currentWidth, blurAmount) ) )
      jumped = True
    
    bestMatch = candidates[0]
    improvement = bestMatch[0]#self.residual - candidates[0][2]
    residual = bestMatch[2]

    self.residual = residual # (has to be recalculated if changing target data below)
    self.avg_improvement = self.avg_improvement*.9 + improvement * .1

    self.string_length += bestMatch[3] * self.parameters["ppi"]
    self.string_path.append(bestMatch[1])

    currentImage = draw_thread(currentImage, pnts.p[current_point_idx], pnts.p[bestMatch[1]], col, width=self.currentWidth)
    self.parameters["CurrentImage"] = currentImage
    self.parameters["lastPoint"] = current_point_idx
    self.parameters["currentPoint"] = bestMatch[1]
    pnts.p[bestMatch[1]].numConnects += 1

    seg = (min(current_point_idx, bestMatch[1]), max(current_point_idx, bestMatch[1]))
    if seg in self.segmentCount:
      self.segmentCount[seg] += 1
    else:
      self.segmentCount[seg] = 1

    pnts.cool(0.1)
    pnts.p[bestMatch[1]].heat = 1.0

    # Update the UI to reflect that we just did

    # pretty render
    beauty_image = draw_thread_rgb(beauty_image, pnts.p[current_point_idx], pnts.p[bestMatch[1]], (1.0,.2,.1, 1.) if not jumped else (0.,1.,0., 1.), width=self.currentWidth)
    #beauty_image2 = draw_thread_rgb(beauty_image2, pnts.p[current_point_idx], pnts.p[bestMatch[1]], (col[0], col[0], col[0], col[1]), width=self.currentWidth)#(col[0],col[0],col[0]), width=1)
    beauty_image2 = draw_thread_rgb(beauty_image2, pnts.p[current_point_idx], pnts.p[bestMatch[1]], (col[0], col[0], col[0], .5), width=1)
    beauty_image = Image.blend(beauty_image, beauty_image2, 0.1)
    draw_points(beauty_image, pnts, highlighed=[c[1] for c in candidates])
    self.parameters["BeautyImage"] = beauty_image
    self.parameters["BeautyImage2"] = beauty_image2
    if "img_invert" in self.parameters and self.parameters["img_invert"] > 0:
      self.showImage(ImageOps.invert(beauty_image))
    else:
      self.showImage(beauty_image)
    if self.save_image and self.iterationCounter%4==0:
      beauty_image.save(self.outPath.format(self.imgCounter))
      self.imgCounter += 1

    # render a difference image
    if self.iterationCounter % 10 == 0:
      redlut   = tuple(((127-i)*2) if i <= 127 else 0 for i in range(256))
      greenlut = tuple(0 if i <= 127 else ((i-127)*2) for i in range(256))
      bluelut  = tuple([0]*256)

      #difImage = ImageChops.subtract(self.targetImage, currentImage.getchannel("R"), 2, 127)
      #difImage = Image.merge("RGB", (difImage, difImage, difImage))
      #df = self.blurredTarget - ndimage.filters.gaussian_filter(currentImage, self.currentBlur)
      sb = sel_blur(currentImage,self.np_targetArray)
      df = sb - self.np_targetArray

      numpy.multiply(df, 0.5, out=df)
      numpy.add(df, 0.5, out=df)
      if self.parameters["img_invert"]:
          df = 1.0 - df
      difImage = array_to_PIL_rgb(df)
      difImage = difImage.point((redlut + greenlut + bluelut))

      self.showImage(difImage, slot=1)
      self.showImage(sb, slot=2)

      now = time.time()
      self.lastTime = now

    self.iterationCounter += 1

    if self.iterationCounter >= self.parameters["maxIterations"]:
      self.close()
    
    if abs(self.avg_improvement) <= 0.001:
      print ("no more improvement")
      self.timer.start(10)
    else:
      self.timer.start(10)

  def showImage(self, image, slot = 0):
    if isinstance(image, numpy.ndarray):
      image = array_to_PIL_rgb(image)

    if slot == 0:
      self.qim = ImageQt.ImageQt(image)
      self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.qim))
      self.imageLabel.adjustSize()
    elif slot == 1:
      self.qim2 = ImageQt.ImageQt(image)
      self.imageLabel2.setPixmap(QtGui.QPixmap.fromImage(self.qim2))
      self.imageLabel2.adjustSize()


def draw_points(pil_image, pnts, size=1,highlighted=None):
  w = int(size-1)/2
  draw = ImageDraw.Draw(pil_image, mode="RGBA")
  if w < 1:
    for p in pnts.p:
      if p.ignore:
        col = (0, 100, 255, 255)
      else:
        col = (255, int(255*(1.0-p.heat)), 0, 255)
      
      draw.point((p.x, p.y), col)
  else:
    for p in pnts.p:
      if p.ignore:
        col = (0, 100, 255, 255)
      else:
        col = (255, int(255*(1.0-p.heat)), 0, 255)
      
      draw.rectangle([p.x-w, p.y-w, p.x+w, p.y+w], fill=(col[0], col[1], col[2], 120), outline=col)

def PIL_to_array(pil_image):
  if pil_image.mode == "RGB":
    ret = numpy.array(pil_image.getchannel("R"), dtype="float32")
    numpy.multiply(ret, 1.0/255, out=ret)
  elif pil_image.mode == "L":
    ret = numpy.array(pil_image, dtype="float32")
    numpy.multiply(ret, 1.0/255, out=ret)
  elif pil_image.mode == "F":
    ret = numpy.array(pil_image, dtype="float32")
  else:
    raise UserWarning("unexpected Image")
  
  return ret    

def array_to_PIL_rgb(imgArray):
  ar = imgArray*255
  ar = numpy.clip(ar, 0, 255, out=ar)
  img = Image.fromarray(ar.astype("uint8"))
  img = Image.merge("RGB", (img, img, img))
  return img

def remove_saturated_segments(fromIdx, neighbours, segCounts, maxCount):
  modified=False
  rem = set()
  for n in neighbours:
    seg = (min(fromIdx,n),max(fromIdx,n))
    if seg in segCounts and segCounts[seg] >= maxCount:
      rem.add(n)
      modified = True

  for r in rem:
    neighbours.remove(r)

  return modified

def remove_nail_collisions(pc, pt_id1, neighbors, maxDist):
  occl = set()
  for n in neighbors:
      occl.update(detect_points_on_line(pc, pt_id1, n, neighbors, maxDist) )

  for o in occl:
      neighbors.remove(o)

def detect_points_on_line(pc, ind1, ind2, neighbors, maxDist):
  """ returns all points of neighbors that are closer then maxDist from the line """

  occluded = set()

  a = list(neighbors)
  #a.remove(ind2)
  A = numpy.array([pc.p[i].asTupple() for i in a])
  B = numpy.repeat((pc.p[ind1].asTupple(),), len(a), axis=0)
  C = numpy.repeat((pc.p[ind2].asTupple(),), len(a), axis=0)

  lenBC = pc.p[ind1].dist(pc.p[ind2])
  #print "len", lenBC, ind1, ind2
  # project A onto BC (all the points onto the line
  CB = (C - B)
  D = CB / lenBC #/ numpy.sqrt((CB**2).sum(-1))[..., numpy.newaxis]   # normaized vector BC
  V = A - B
  t = (V*D).sum(-1)[...,numpy.newaxis] # dot product element wise
  P = B + D * t
  AP = (A - P)
  distSqr = (AP**2).sum(-1)[..., numpy.newaxis]
  onRay = distSqr <= maxDist * maxDist
  onLine = [(t[i][0], a[i]) for i in range(len(a)) if onRay[i][0] and t[i][0] >= 0.0]
  onLine.sort()
  if len(onLine) > 0:
    for i in onLine[1:]:
      occluded.add(i[1])

  return occluded

def check_quality(params):
  img = params[0]
  p1 = params[1]
  p2 = params[2]
  trg = params[3]
  ind = params[4]
  col = params[5]
  prevResidual = params[6]
  blurredTarget = params[7]
  width = params[8]
  blur = params[9]

  length = p1.dist(p2)
  b_len = max(int(abs(p1.x-p2.x)+1), int(abs(p1.y-p2.y))+1) # bresenham num pixels drawn

  cur_diff = 0
  new_img, np = draw_thread_qual(img, pnt1=p1, pnt2=p2, color=col, width=width)
  #new_img, np = draw_thread(img, pnt1=p1, pnt2=p2, color=col, width=width), 1

  #cur_diff = image_diff(new_img, trg)    # what is the difference to the target
  cur_diff = image_diff(sel_blur(new_img, trg), trg)

  #if blur > 0.1:
  #    blurredImg = ndimage.filters.gaussian_filter(new_img, blur)
  #    #cur_diff = image_diff(blurredImg, blurredTarget)    # what is the difference to the target
  #    cur_diff += image_diff(blurredImg, blurredTarget)
  #else:
  #    cur_diff = image_diff(new_img, trg)    # what is the difference to the target

  #quality = (cur_diff - prevResidual)/(b_len**2)    # how much better did this line make the result
  #quality = (cur_diff - prevResidual)
  #quality = (cur_diff - prevResidual) / (b_len)
  #quality = (cur_diff - prevResidual) / length
  quality = (cur_diff - prevResidual) / np
  quality += abs(quality) * 1.5 * p2.heat # attenuate by previously visited
  return (quality, ind, cur_diff, length)

def image_diff(imageArray, targetArray):
  error = numpy.subtract(imageArray, targetArray)

  better = numpy.clip(error, -2000000000, 0)
  worse  = numpy.multiply(numpy.clip(error, 0, 2000000000, out=error), 4, out=error)
  #worse = numpy.multiply(numpy.clip(error, 0, 2000000000, out=error), 2, out=error)
  error = numpy.add(better, worse, out=error)

  #error = numpy.multiply(error, error, out=error) # error**2
  #error = numpy.sqrt(error, out=error)
  error = numpy.abs(error, out=error)

  return numpy.sum(error)

def draw_thread(imageArray, pnt1, pnt2, color, width):
  return draw_thread_qual(imageArray, pnt1, pnt2, color, width, False)[0]

def draw_thread_qual(imageArray, pnt1, pnt2, color, width, calc_num=True):
  width=int(width)
  img = Image.new("F", (imageArray.shape[1], imageArray.shape[0]))#array_to_PIL_f(imageArray)
  draw = ImageDraw.Draw(img, mode="F")
  draw.line([pnt1.x, pnt1.y, pnt2.x, pnt2.y], width=width, fill=color[1])
  if width > 2:
    w=width/2.0
    draw.ellipse((pnt1.x-w, pnt1.y-w, pnt1.x+w, pnt1.y+w), fill=color[1])
    draw.ellipse((pnt2.x-w, pnt2.y-w, pnt2.x+w, pnt2.y+w), fill=color[1])

  col = PIL_to_array(img)
  numpix = len(numpy.nonzero(col.flatten())[0]) if calc_num else 0
  msk = 1 - col
  numpy.multiply(col, color[0], out=col)

  msk = numpy.multiply(imageArray, msk, out=msk)
  ret = numpy.add(msk, col, out=msk)
  return ret, numpix

def sel_blur(img_np, mask_np):
  num_mipmaps = 5

  img = array_to_PIL_f(img_np)
  mipmaps = [img_np.copy()]
  for m in range(num_mipmaps):
      #f = 2**(m+1)
      f = (2 + m)
      mm = img.resize((img.width//f, img.height//f), resample=Image.BICUBIC ).resize( img.size, resample=Image.BICUBIC )
      mipmaps.append(PIL_to_array(mm) )

  for i,m in enumerate(mipmaps):
      dst = 1.0 / (num_mipmaps)
      mm_end   = 1.0 - dst * (i-1)
      mm_mid   = 1.0 - dst *  i
      mm_start = 1.0 - dst * (i+1)
      #msk = numpy.logical_and( trg > mm_start, trg <= mm_end ).astype("float32")
      weightup = (mask_np - mm_start) / (mm_mid - mm_start)
      weightdown = 1.0 - (mask_np - mm_mid) / (mm_end - mm_mid)
      weightup = numpy.clip(weightup, 0, 1, out=weightup)
      weightdown = numpy.clip(weightdown, 0, 1, out=weightdown)
      numpy.multiply(m, weightup, out=m)
      numpy.multiply(m, weightdown, out=m)

      #print i,":",m.shape, mm_start,"-",mm_mid,"-",mm_end
      #array_to_PIL_rgb(m).show()

  return sum(mipmaps)

def draw_thread_rgb(image, pnt1, pnt2, color, width):
  width=int(width)
  if width > 2 or color[3] < 1.0:
    img = Image.new("L", image.size)
    draw = ImageDraw.Draw(img)
    draw.line([pnt1.x, pnt1.y, pnt2.x, pnt2.y], width=width, fill=int(color[3]*255))
    if width > 2:
      w=width/2.0
      draw.ellipse((pnt1.x-w, pnt1.y-w, pnt1.x+w, pnt1.y+w), fill=int(color[3]*255))
      draw.ellipse((pnt2.x-w, pnt2.y-w, pnt2.x+w, pnt2.y+w), fill=int(color[3]*255))
    return Image.composite(Image.new("RGB", image.size, (int(color[0]*255), int(color[1]*255), int(color[2]*255))), image, img)
  else:
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    draw.line([pnt1.x, pnt1.y, pnt2.x, pnt2.y], width=width, fill=(int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255)))
    return img


def array_to_PIL_f(imgArray):
  ar = imgArray
  #ar = numpy.clip(ar, 0, 255, out=ar)
  #img = Image.fromarray(ar.astype("uint8"))
  img = Image.fromarray(imgArray)
  #img = Image.merge("RGB", (img,img,img))
  return img

def Enhance(image, params):
  width = params["proc_width"]
  height = params["proc_height"]
  contrast = params["img_contrast"]
  brightness = params["img_brightness"]
  invert = params["img_invert"]

  img = image.resize((width, height))

  if invert > 0:
    img = ImageOps.invert(img)

  if contrast != 1.0:
    enh = ImageEnhance.Contrast(img)
    img = enh.enhance(contrast)

  if brightness != 1.0:
    bt = ImageEnhance.Brightness(img)
    img = bt.enhance(brightness)

  # 图片灰阶
  return img.convert("L")

if __name__ == '__main__':
  app = QtWidgets.QApplication(sys.argv)

  mpp = 0.3/600 # meters per pixel
  params = {
    "searchRadius": 0.25,
    "proc_height": 600,
    "inputImagePath": "einstein3.png",
    "img_invert": 0,
    "img_contrast": 1.0,
    "img_brightness": 1.0,
    "blurAmount" : 6.0, 
    "backgroundColor":0,
    "nailDistMin": 6 / 1000.0 / mpp,      # minimum nail distance
    "nailDistMax": 16.0 / 1000.0 / mpp,     # maximum nail distance
    "nailDiameter": 1.5 / 1000.0 / mpp, 
    "edgeThreshold": 0.3,
    "start_at": (0.5,0), 
    "maxConnectsPerNail": 8,  
    "currentPoint" : 0,
    "lastPoint": -1,
    "threadColor":(255, 160),
    "maxSegmentConnect": 1, 
  }

  # 加载图片
  img = Image.open(params["inputImagePath"]).convert("RGB")
  params["proc_width"] = int(params["proc_height"]*float(img.width)/img.height)
  print("input image {}x{}, proc dim:{}x{}".format(img.width, img.height, params["proc_width"], params["proc_height"]))
  img = Enhance(img, params)

  params["TargetImage"] = img

  imageViewer = Viewer(params)
  imageViewer.show()

  sys.exit(app.exec())
