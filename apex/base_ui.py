import sys
import cv2
from PySide6.QtWidgets import QMainWindow,QApplication,QFileDialog
from PySide6.QtGui import QPixmap,QImage
from main_window import Ui_MainWindow
from PySide6.QtCore import QTimer
import torch

def convert2QImage(img):
    height,width,channel = img.shape
    return QImage(img,width,height,width * channel,QImage.Format_RGB888)

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # model ,参数代表的意思分别为，yolov5根目录，custom表示自定义模型，相对路径，本地资源
        self.model = torch.hub.load("../","custom",path="./weight/best.pt", source="local")
        self.timer = QTimer()
        self.video = None
        self.timer.setInterval(1)
        # 执行一下，将其绑定
        self.bind_slots()

    def image_pred(self,file_path):
        results = self.model(file_path)
        image = results.render()[0]
        return convert2QImage(image)

    def video_pred(self):
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.input.setPixmap(QPixmap.fromImage(convert2QImage(frame)))
            results = self.model(frame)
            image = results.render()[0]
            self.output.setPixmap(QPixmap.fromImage(convert2QImage(image)))

    def open_image(self):
        print("点击了监测图片")
        self.timer.stop()
        file_path = QFileDialog.getOpenFileName(self,dir="./data/train",filter="*.jpg;*.png;*.jpeg")
        if file_path[0]:
            file_path = file_path[0]
            qimage = self.image_pred(file_path)
            self.input.setPixmap(QPixmap(file_path))
            self.output.setPixmap(QPixmap.fromImage(qimage))

    def open_video(self):
        print("点击了视频检测")
        file_path = QFileDialog.getOpenFileName(self, dir="./data", filter="*.mp4")
        if file_path[0]:
            file_path = file_path[0]
            self.video = cv2.VideoCapture(file_path)
            self.timer.start()



    def bind_slots(self):
        self.det_image.clicked.connect(self.open_image)
        self.det_video.clicked.connect(self.open_video)
        self.timer.timeout.connect(self.video_pred)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()