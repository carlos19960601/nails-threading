import sys
import time

from PIL import Image, ImageEnhance, ImageOps, ImageQt
from PySide6 import QtCore, QtGui, QtWidgets


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

    self.targetImage = self.parameters["TargetImage"]
    
    self.lastTime = time.time()

    self.timer = QtCore.QTimer(self)
    self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.workImage)
    self.timer.setSingleShot(True)
    self.timer.start(2000)

  def workImage(self):
    targetImage = self.parameters["TargetImage"]
    if not "DetailImage" in self.parameters:
      self.setWindowTitle("NailedIt - Detail Image")
      self.showImage(targetImage)
      self.timer.start(1000)
  
  def showImage(self, image, slot = 0):
    if slot == 0:
      self.qim = ImageQt.ImageQt(image)
      self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.qim))
      self.imageLabel.adjustSize()

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
  params = {
    "proc_height": 600,
    "inputImagePath": "einstein3.png",
    "img_invert": 0,
    "img_contrast": 1.0,
    "img_brightness": 1.0,
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
