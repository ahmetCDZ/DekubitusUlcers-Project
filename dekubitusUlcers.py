import cv2
import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui

evre = ''
def detected(photoUrl, scar):
    img = cv2.imread(photoUrl)
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)

    labels = ["evre1", "evre2", "evre3", "evre4"]
    colors = ["0,0,0", "139,69,19", "179,255,255", "22,100,100"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors, (18, 1))

    model = cv2.dnn.readNetFromDarknet("/Users/ahmetzincir/Desktop/PyCharm Projects/yolov4.cfg",
                                       "/Users/ahmetzincir/Desktop/PyCharm Projects/yolov4_final.weights")
    layers = model.getLayerNames()
    output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]
    model.setInput(img_blob)
    detection_layers = model.forward(output_layer)

    ids_list = []
    boxes_list = []
    confidence_list = []

    for detection_layer in detection_layers:
        for object_detection in detection_layer:

            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            if confidence > 0.60:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width / 2))
                start_y = int(box_center_y - (box_height / 2))

                ids_list.append(predicted_id)
                confidence_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5, 0.4)
    for max_id in max_ids:
        max_clases_id = max_id
        box = boxes_list[max_clases_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_clases_id]
        label = labels[predicted_id]
        confidence = confidence_list[max_clases_id]

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]

        label = "{}:{: .2f}%".format(label, confidence * 100)
        scar = label[:5]

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 1)
        cv2.putText(img, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    cv2.imwrite("/Users/ahmetzincir/Desktop/PyCharm Projects/1.JPG", img)
    return "/Users/ahmetzincir/Desktop/PyCharm Projects/1.JPG", scar

class DekubitusUlcer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.icerik = QtWidgets.QLabel(self)
        self.inform = QtWidgets.QLabel(self)
        self.icerik2 = QtWidgets.QLabel(self)
        self.butonSearch = QtWidgets.QPushButton("Lütfen resim seçmek için tıklayınız...")
        self.butonClose = QtWidgets.QPushButton("Kapat")
        self.icerik.setScaledContents(True)
        self.butonSearch.clicked.connect(self.openPhoto)
        self.butonClose.clicked.connect(self.closePhoto)
        self.icerik2.setPixmap(QtGui.QPixmap("/Users/ahmetzincir/Desktop/PyCharm Projects/resim4.JPG"))

        self.menu_bar = QtWidgets.QMenuBar()
        self.setMenuBar(self.menu_bar)
        self.file_menu = self.menu_bar.addMenu("Dosya")
        self.file_menu.addAction("Aç")
        self.file_menu.addAction("Kaydet")

        layoutV = QtWidgets.QVBoxLayout()
        layoutH = QtWidgets.QHBoxLayout()
        layoutV.addWidget(self.icerik2)

        self.icerik2.setScaledContents(True)
        self.icerik.setScaledContents(True)

        layoutH.addStretch()
        layoutV.addStretch()
        layoutV.addWidget(self.inform)
        layoutV.addWidget(self.icerik)
        layoutV.addWidget(self.butonSearch)
        layoutV.addWidget(self.butonClose)
        layoutV.addStretch()
        layoutH.addLayout(layoutV)
        layoutH.addStretch()

        self.icerik2.setFixedSize(700, 150)
        self.setLayout(layoutH)
        self.setWindowTitle("Hasta Bası Yarası Uygulaması")
        self.setFixedSize(800, 800)
        self.show()
        self.butonClose.close()

    def openPhoto(self):
        resimUrl = QtWidgets.QFileDialog.getOpenFileName(self, "Lütfen resim seçmek için tıklayınız...")
        newUrl = detected(resimUrl[0], evre)
        gelen = newUrl[1]
        if gelen == 'evre1':
            self.icerik2.setPixmap(QtGui.QPixmap("/Users/ahmetzincir/Desktop/PyCharm Projects/Evre1.JPG"))
        elif gelen == 'evre2':
            self.icerik2.setPixmap(QtGui.QPixmap("/Users/ahmetzincir/Desktop/PyCharm Projects/Evre2.JPG"))
        elif gelen == 'evre3':
            self.icerik2.setPixmap(QtGui.QPixmap("/Users/ahmetzincir/Desktop/PyCharm Projects/Evre3.JPG"))
        elif gelen == 'evre4':
            self.icerik2.setPixmap(QtGui.QPixmap("/Users/ahmetzincir/Desktop/PyCharm Projects/Evre3.JPG"))
        else:
            self.icerik2.setPixmap(QtGui.QPixmap("/Users/ahmetzincir/Desktop/PyCharm Projects/Evre0.JPG"))

        self.icerik.setPixmap(QtGui.QPixmap(newUrl[0]))
        self.butonClose.show()

    def closePhoto(self):
        self.icerik.setPixmap(QtGui.QPixmap(""))
        self.butonClose.close()

    def setMenuBar(self, menu_bar):
        pass

uygulama = QtWidgets.QApplication(sys.argv)
dekubitUlcer = DekubitusUlcer()
sys.exit(uygulama.exec_())
