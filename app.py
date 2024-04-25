from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QMessageBox, QColorDialog, QVBoxLayout, QPushButton, QWidget, QHBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QDir, Qt
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import sys
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_path = ""
        self.roi = None
        self.bg_color = (255, 255, 255) 
        self.pozaRoi = None
        self.create_ui()

    def create_ui(self):
       
        self.setWindowTitle('Procesare imagini')

        self.select_image_button = QtWidgets.QPushButton("Incarca imaginea", self)
        self.select_image_button.setGeometry(50, 50, 200, 30)
        self.select_image_button.clicked.connect(self.select_image)

        self.roi_button = QtWidgets.QPushButton("Selecteaza ROI", self)
        self.roi_button.setGeometry(50, 100, 200, 30)
        self.roi_button.setEnabled(False)
        self.roi_button.clicked.connect(self.select_roi)

        self.bg_color_button = QtWidgets.QPushButton("Culoare fundal", self)
        self.bg_color_button.setGeometry(50, 150, 200, 30)
        self.bg_color_button.setEnabled(False)
        self.bg_color_button.clicked.connect(self.select_bg_color)

        self.generate_button = QtWidgets.QPushButton("Schimba culoarea", self)
        self.generate_button.setGeometry(50,200,200,30)
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_image)

        self.roi_label = QtWidgets.QLabel(self)
        self.roi_label.setGeometry(10, 300, 200, 200)
        self.roi_label.setAlignment(QtCore.Qt.AlignCenter)

        self.thresh_label = QtWidgets.QLabel(self)
        self.thresh_label.setGeometry(220, 300, 200, 200)
        self.thresh_label.setAlignment(QtCore.Qt.AlignCenter)

        self.dilate_label = QtWidgets.QLabel(self)
        self.dilate_label.setGeometry(460, 300, 200, 200)
        self.dilate_label.setAlignment(QtCore.Qt.AlignCenter)

        self.erodare_label = QtWidgets.QLabel(self)
        self.erodare_label.setGeometry(690, 300, 200, 200)
        self.erodare_label.setAlignment(QtCore.Qt.AlignCenter)

        self.horizontalSliderThreshold = QSlider(self)
        self.horizontalSliderThreshold.setOrientation(Qt.Horizontal)
        self.horizontalSliderThreshold.setObjectName("horizontalSliderThreshold")
        self.horizontalSliderThreshold.setMaximum(255)
        self.horizontalSliderThreshold.setGeometry(100, 550, 700, 35)
        self.horizontalSliderThreshold.valueChanged.connect(self.on_horizontalSliderThreshold_valueChanged)

        text_roi = QLabel('Imagine ROI', self)
        text_roi.setGeometry(50, 250, 200, 30)

        text_thresh = QLabel('Threshold', self)
        text_thresh.setGeometry(270, 250, 200, 30)

        text_dilate = QLabel('Dilatare', self)
        text_dilate.setGeometry(550, 250, 200, 30)

        text_erode = QLabel('Erodare', self)
        text_erode.setGeometry(760, 250, 200, 30)

        self.pushButtonSave = QPushButton("Salveaza", self)
        self.pushButtonSave.setObjectName("pushButtonSave")
        self.pushButtonSave.setGeometry(730,100,150,30)
        self.pushButtonSave.clicked.connect(self.on_pushButtonSave_clicked)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setGeometry(300, -100, 400, 400)

    def select_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selectează imaginea", "", "Images (*.png *.xpm *.jpg *.bmp)", options=options)
        if file_name:
            self.image_path = file_name

            self.image_label.setPixmap(QtGui.QPixmap(self.image_path).scaled(400, 400, QtCore.Qt.KeepAspectRatio))
            self.image_label.setAlignment(QtCore.Qt.AlignCenter)
            self.roi_button.setEnabled(True)

    def select_roi(self):
        image = cv2.imread(self.image_path)
        roi = cv2.selectROI(image)

        self.roi = roi

        self.roi_button.setEnabled(False)
        self.bg_color_button.setEnabled(True)

        roi_image = image[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2]]
        roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        height, width, channel = roi_image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(roi_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.roi_label.setPixmap(QPixmap.fromImage(q_image))
        self.roi_label.setScaledContents(True)

        self.pozaRoi = roi_image.copy()


    def select_bg_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            rgb = color.getRgb()
            self.bg_color = (rgb[2], rgb[1], rgb[0])

            self.generate_button.setEnabled(True)
            self.applyBackgroundColor(self.pozaRoi, self.erodedDilatedMat)


    def generate_image(self):
        image = cv2.imread(self.image_path)
        mask = np.full(image.shape[:2], 2, np.uint8)

        x, y, w, h = self.roi

        mask[y:y + h, x:x + w] = 3
        mask[:y, :] = 0
        mask[y + h:, :] = 0
        mask[:, :x] = 0
        mask[:, x + w:] = 0

        model = np.zeros((1, 13 * 5), np.float64) 
        cv2.grabCut(image, mask, None, model, None, 5, cv2.GC_INIT_WITH_MASK)

        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * fg_mask[:, :, np.newaxis]
        image[np.where((image == [0, 0, 0]).all(axis=2))] = self.bg_color

        cv2.imwrite("rezultat.jpg", image)

        self.image_label.setPixmap(QtGui.QPixmap("rezultat.jpg").scaled(400, 400, QtCore.Qt.KeepAspectRatio))
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)


    def on_horizontalSliderThreshold_valueChanged(self, value):
        if self.pozaRoi is not None and self.pozaRoi.size != 0:
            grayMat = cv2.cvtColor(self.pozaRoi, cv2.COLOR_RGB2GRAY)
            _, thresholdedMat = cv2.threshold(grayMat, value, 255.0, cv2.THRESH_BINARY_INV)

            thresholdedImage = QImage(thresholdedMat.data, thresholdedMat.shape[1], thresholdedMat.shape[0],
                                    thresholdedMat.strides[0], QImage.Format_Grayscale8)
            self.thresh_label.setPixmap(QPixmap.fromImage(thresholdedImage))
            self.thresh_label.setScaledContents(True)


            matr_dilate = np.ones((3, 3), np.uint8)
            dilatedMat = cv2.dilate(thresholdedMat, matr_dilate, iterations=10)

            dilatedImage = QImage(dilatedMat.data, dilatedMat.shape[1], dilatedMat.shape[0],
                                dilatedMat.strides[0], QImage.Format_Grayscale8)
            self.dilate_label.setPixmap(QPixmap.fromImage(dilatedImage))
            self.dilate_label.setScaledContents(True)


            matr_erode = np.ones((3, 3), np.uint8)
            erodedDilatedMat = cv2.erode(dilatedMat, matr_erode, iterations=5)

            erodedDilatedImage = QImage(erodedDilatedMat.data, erodedDilatedMat.shape[1], erodedDilatedMat.shape[0],
                                        erodedDilatedMat.strides[0], QImage.Format_Grayscale8)
            self.erodare_label.setPixmap(QPixmap.fromImage(erodedDilatedImage))
            self.erodare_label.setScaledContents(True)
            self.erodedDilatedMat = erodedDilatedMat
            

    def applyBackgroundColor(self, originalImage, erodedDilatedMat):
        newBackgroundMat = np.zeros_like(self.pozaRoi)

        for y in range(self.pozaRoi.shape[0]):
            for x in range(self.pozaRoi.shape[1]):
                if erodedDilatedMat[y, x] == 255:
                    newBackgroundMat[y, x] = self.pozaRoi[y, x]
                else:
                    newBackgroundMat[y, x] = self.bg_color
                    

        newBackgroundImage = QImage(newBackgroundMat.data, newBackgroundMat.shape[1], newBackgroundMat.shape[0],
                                    newBackgroundMat.strides[0], QImage.Format_RGB888)
        self.roi_label.setPixmap(QPixmap.fromImage(newBackgroundImage))
        self.roi_label.setScaledContents(True)

    def on_pushButtonSave_clicked(self):
        newBackgroundImage = self.image_label.pixmap()

        if newBackgroundImage.isNull():
            print("Nicio imagine de salvat.")
            return

        filePath, _ = QFileDialog.getSaveFileName(self, "Salveaza imaginea", QDir.homePath(), "Images (*.png *.jpg *.jpeg *.bmp)")

        if filePath:
            if newBackgroundImage.save(filePath):
                print("Imaginea a fost salvata la: ", filePath)
            else:
                print("Eroare la salvarea imaginii.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())