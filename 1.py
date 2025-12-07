import sys
import numpy as np
import rasterio
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QVBoxLayout, QWidget, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier

win_width, win_height = 800,600



class main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dự đoán sạt lở")
        self.resize(win_width, win_height)


        self.label_status = QLabel("Load các file để bắt đầu...")
        self.btn_load_terrain = QPushButton(" Chọn file DEM (.tif)")
        self.btn_load_rain = QPushButton(" Chọn file Mưa (.tif)")
        self.btn_load_NVDI = QPushButton(" Chọn file NDVI (.tif)")
        self.btn_run = QPushButton(" Dự đoán sạt lở")
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))


       
        

        layout = QVBoxLayout()

        layout.addWidget(self.label_status)
        layout.addStretch(1) 
        self.buttons = [self.btn_load_terrain, self.btn_load_rain, self.btn_load_NVDI]
        for btn in self.buttons:
            layout.addWidget(btn)
        layout.addWidget( self.btn_run)
        layout.addWidget(self.canvas)
        
        for btn in (self.btn_load_terrain,self.btn_load_rain, self.btn_run):
            btn.setMinimumWidth(self.width()// 2)
            btn.setMinimumHeight(self.height()// 7)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addSpacing(8)


        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


        self.btn_load_terrain.clicked.connect(lambda: self.load("DEM"))
        self.btn_load_rain.clicked.connect(lambda: self.load("Mưa"))
        self.btn_load_NVDI.clicked.connect(lambda: self.load("NDVI"))



        self.btn_run.clicked.connect(self.run_prediction)


        self.dem = None
        self.rain = None
        self.NDVI = None
        self.dia_chat = None

    def resizeEvent(self, event):

        new_font_size = max(15, self.height() // 25)
        for btn in self.buttons:
            font = btn.font()
            font.setPointSize(new_font_size)
            btn.setFont(font)
        font = self.btn_run.font()
        font.setPointSize(new_font_size)
        self.btn_run.setFont(font)
        super().resizeEvent(event)


    def load(self, type: str):
        path, _ = QFileDialog.getOpenFileName(self, f"Chọn {type} (.tif)", "", "GeoTIFF Files (*.tif)")
        if not path:
            return

        with rasterio.open(path) as src:
            data = src.read(1, masked=True)

        if type == "DEM":
            self.dem = data
        elif type == "Mưa":
            self.rain = data
        elif type == "NDVI":
            self.NDVI = data
        elif type == "Địa chất":
            self.dia_chat = data

        self.label_status.setText(f" Đã tải {type}: {path.split('/')[-1]}")




    def run_prediction(self):
        if self.dem is None or self.rain is None or self.NDVI is None or self.dia_chat is None:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng tải cả DEM và Mưa trước.")
            return

        self.label_status.setText(" Đang tính toán...")

       
        if self.rain.shape != self.dem.shape:
            self.rain = np.resize(self.rain, self.dem.shape)
        if self.NDVI.shape != self.dem.shape:
            self.NDVI = np.resize(self.NDVI, self.dem.shape)
        if self.dia_chat.shape != self.dem.shape:
            self.dia_chat = np.resize(self.dia_chat, self.dem.shape)


        x, y = np.gradient(self.dem.filled(0))
        slope_deg = np.degrees(np.arctan(np.sqrt(x*x + y*y)))


        label = ((slope_deg > 30) & (self.rain > np.percentile(self.rain, 80)  ) & self.NDVI < 0.3  ).astype(np.uint8)

        mask = (
            (~self.dem.mask) &
            (~self.rain.mask) &
            (~self.NDVI.mask)
        )


        X = np.column_stack((
            self.dem[mask].ravel(),
            slope_deg[mask].ravel(),
            self.rain[mask].ravel(),
            self.NDVI[mask].ravel(),
            self.dia_chat[mask].ravel()
        ))

        y = label[mask].ravel()


        model = RandomForestClassifier(n_estimators=50, max_depth=4, n_jobs=-1)
        model.fit(X, y)


        proba = np.zeros_like(self.dem, dtype=np.float32)
        proba[mask] = model.predict_proba(X)[:, 1]


        self.kq_win = kq(proba)
        self.kq_win.show()

        




class kq(QMainWindow):
    def __init__(self, proba):
        super().__init__()
        self.setWindowTitle("Kết quả dự đoán sạt lở đất")
        self.resize(700, 500)

        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Hiển thị bản đồ ngay khi khởi tạo
        ax = self.canvas.figure.subplots()
        im = ax.imshow(proba, cmap="Reds")
        ax.set_title("Bản đồ nguy cơ sạt lở đất")
        self.canvas.figure.colorbar(im, ax=ax, label="Xác suất sạt lở")
        self.canvas.draw()


app = QApplication(sys.argv)
window = main()
window.show()
sys.exit(app.exec())
