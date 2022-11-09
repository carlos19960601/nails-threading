import math
import random

import numpy
from PIL import ImageDraw
from scipy.spatial import Delaunay


def remap(val, from_min, from_max, to_min, to_max):
  return (((val - from_min) * (to_max - to_min)) / (from_max - from_min)) + to_min


class Point2(object):
  def __init__(self, x=0.0, y=0.0, heat=0.0,ignore=False, numConnects=0):
    self.x = float(x)
    self.y = float(y)
    self.heat = heat
    self.ignore = ignore
    self.numConnects = numConnects

  def __add__(self, other):
    return Point2(self.x+other.x, self.y+other.y, self.heat, self.ignore, self.numConnects)

  def __sub__(self, other):
    return Point2(self.x-other.x, self.y-other.y, self.heat, self.ignore, self.numConnects)

  def __mul__(self, other):
    if isinstance(other, Point2):
      return Point2(self.x * other.x, self.y * other.x, self.heat, self.ignore, self.numConnects)
    else:
      return Point2(self.x*other, self.y*other, self.heat, self.ignore, self.numConnects)

  def __div__(self, other):
    return Point2(self.x/other, self.y/other, self.heat, self.ignore, self.numConnects)

  def dist2(self, other):
    return (self.x - other.x)**2 + (self.y-other.y)**2

  def dist(self, other):
    return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)
  
  def clamped(self, minx, maxx, miny, maxy):
    return Point2( max(minx, min(maxx, self.x)), max(miny, min(maxy, self.y)), self.heat, self.ignore, self.numConnects)

  def asTupple(self):
    return (self.x, self.y)

class PointCloud(object):
  def __init__(self, dimx, dimy):
    self.p = []
    self.width = dimx
    self.height = dimy
    self.kd = None

  def heat(self, temp):
    for pnt in self.p:
      pnt.heat = temp

  def addGrid(self, w, h, offset=0.5):
    pt = [Point2(float(x) / (w-1) + ((offset/(w-1)) if y % 2 else 0), float(y)/(h-1)) for y in range(int(h)) for x in range(int(w if (y%2==0) else (w-1)))]
    self.p += [Point2(p.x*(self.width-1), p.y*(self.height-1)) for p in pt]

  def addRandom(self, num):
    random.seed(1234)
    self.p += [Point2(random.uniform(0, float(self.width-1)), random.uniform(0, float(self.height-1))) for n in range(num)]

  def relax(self, image, iterations, detail_img, minDist, maxDist): 
    npp = numpy.array([[pnt.x, pnt.y] for pnt in self.p])
    tri = Delaunay(npp)

    msk = [pt.heat for pt in self.p]
    # mask the outside border
    for t_ind, ns in enumerate(tri.neighbors):
      for n_ind, n in enumerate(ns):
        if n == -1:
          for i in [0,1,2]:
            if i != n_ind:
              msk[tri.simplices[t_ind][i]] = 1.0

    # draw mesh
    if image:
      drawn = set()
      draw = ImageDraw.Draw(image)
      for t in tri.simplices:
        pp = (tri.points[t[0]], tri.points[t[1]], tri.points[t[2]])
        for i,j in [(0,1),(1,2),(2,0)]:
          pair = (min(t[i], t[j]), max(t[i], t[j]))
          if not pair in drawn:
            draw.line([pp[i][0], pp[i][1], pp[j][0],pp[j][1]], (180,150,0))
            drawn.add(pair)

    # try to average edge length
    numEdges = 0
    targetLength = 0.0
    edgedone = set()
    for t in tri.simplices:
      for i,j in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
        pair = (min(i, j), max(i, j))
        if pair not in edgedone:
          edgedone.add(pair)
          targetLength += self.p[i].dist(self.p[j])
          numEdges += 1
    targetLength /= numEdges

    ease = 0.25 # only move it this much of the desired distance
    edgedone = set()
    for t in tri.simplices:
      for i,j in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]:
        pair = (min(i, j), max(i, j))
        if not pair in edgedone:
          l = self.p[i].dist(self.p[j])
          f = (targetLength/l)*ease + (1.0-ease)
          # scale edge around midpoint
          mp = (self.p[i] + self.p[j]) * 0.5

          # scale lenght by detail image
          if not isinstance(detail_img, type(None)):
            #det = detail_img.getpixel((mp.x, mp.y))
            det = detail_img[int(mp.y)][int(mp.x)]
            det = remap(det, 0., 1., maxDist/l, minDist/l)
            f *= 1.0 - det / iterations

          if msk[i] <= 0.0 and not self.p[i].ignore:
            self.p[i] = (self.p[i] - mp) * f + mp
            self.p[i] = self.p[i].clamped(0.0, self.width-1, 0.0, self.height-1)
          if msk[j] <= 0.0 and not self.p[j].ignore:
            self.p[j] = (self.p[j] - mp) * f + mp
            self.p[j] = self.p[j].clamped(0.0, self.width-1, 0.0, self.height-1)
          
          edgedone.add(pair)


  def scatterOnMask(self, maskImg, numPoints, minDist, threshold = 0.2):
    random.seed(4826)
    num = 0
    fail = 0

    numpy.random.seed(64726)
    f = maskImg.flatten()
    interesting = numpy.where(f >= threshold)[0]
    numpy.random.shuffle(interesting)
    for i in interesting:
      pt = Point2(float(i % maskImg.shape[1]), float(i / maskImg.shape[1]))
      if len(self.p) == 0 or self.closestPoint(pt.x, pt.y)[1] >= minDist:
        self.p.append(pt)
        num+=1
        if num >= numPoints:
          break
      else:
        fail+=1
        if fail >= numPoints*20:
          break

  def closestPoint(self, x,y,thatsNot=-1):
    to = Point2(x, y)
    dst = [(pnt.dist2(to), i) for i, pnt in enumerate(self.p) if i != thatsNot]
    dst.sort()
    return dst[0][1], math.sqrt(dst[0][0])


  def closestPoints(self, x, y, radius, thatsNot=-1):
    radius = radius*radius
    to = Point2(x, y)
    dst = [(pnt.dist2(to), i) for i,pnt in enumerate(self.p) if  i != thatsNot]
    ret = [d[1] for d in dst if d[0] <= radius]
    return ret

  def findNeighbours(self, pntInd, max_radius):
    #grid
    # for now just return everything but the given point
    #ret = range(len(pnts))
    #del ret[pnt]
    r = max_radius**2
    ret = [i for i in range(len(self.p)) if (i != pntInd and not self.p[i].ignore and self.p[pntInd].dist2(self.p[i]) < r)]

    random.seed(73674)
    random.shuffle(ret)
    return ret
