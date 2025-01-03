import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton,
    QWidget, QFileDialog, QComboBox, QHBoxLayout, QMessageBox, QSlider
)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Annotation Tool")
        self.data_files = []
        self.data = None
        self.current_file_index = 0
        self.current_frame_index = 0
        self.labels = []  # One label per file
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)

        self.init_ui()

    def init_ui(self):
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # PyQtGraph view for rendering
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setXRange(0, 1280)
        self.plot_widget.setYRange(0, 720)
        self.plot_widget.invertY(True)
        layout.addWidget(self.plot_widget)

        # Frame slider
        self.frame_slider = QSlider()
        self.frame_slider.setOrientation(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.update_frame)
        layout.addWidget(self.frame_slider)

        # Label selection
        label_layout = QHBoxLayout()
        layout.addLayout(label_layout)

        self.label_1 = QComboBox()
        self.label_1.addItems(["None", "追逐", "对峙", "压制"])
        label_layout.addWidget(QLabel("Label 1:"))
        label_layout.addWidget(self.label_1)

        self.label_2 = QComboBox()
        self.label_2.addItems(["None", "追逐", "对峙", "压制"])
        label_layout.addWidget(QLabel("Label 2:"))
        label_layout.addWidget(self.label_2)

        # Navigation and playback buttons
        nav_layout = QHBoxLayout()
        layout.addLayout(nav_layout)

        self.prev_button = QPushButton("Previous File")
        self.prev_button.clicked.connect(self.prev_file)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next File")
        self.next_button.clicked.connect(self.next_file)
        nav_layout.addWidget(self.next_button)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        nav_layout.addWidget(self.play_button)

        # Save button
        self.save_button = QPushButton("Save Labels")
        self.save_button.clicked.connect(self.save_labels)
        layout.addWidget(self.save_button)

        # Load data button
        self.load_button = QPushButton("Load Data Folder")
        self.load_button.clicked.connect(self.load_data_folder)
        layout.addWidget(self.load_button)

    def load_data_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder_path:
            self.data_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')
            ]
            if not self.data_files:
                QMessageBox.warning(self, "Warning", "No .npy files found in the selected folder.")
                return

            self.current_file_index = 0
            self.labels = [["None", "None"] for _ in range(len(self.data_files))]  # One label per file
            self.load_current_file()

    def load_current_file(self):
        if not self.data_files:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return

        file_path = self.data_files[self.current_file_index]
        self.data = np.load(file_path)
        self.current_frame_index = 0
        self.frame_slider.setMaximum(self.data.shape[0] - 1)
        self.render_current_frame()
        labels = self.labels[self.current_file_index]
        self.label_1.setCurrentText(labels[0])
        self.label_2.setCurrentText(labels[1])

    def render_current_frame(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return

        self.plot_widget.clear()

        frame_data = self.data[self.current_frame_index]
        x1, y1, r1, x2, y2, r2 = frame_data

        # Draw circles and connecting line
        scatter = pg.ScatterPlotItem()
        scatter.setData(
            x=[x1, x2],
            y=[y1, y2],
            size=[2 * r1, 2 * r2],
            brush=[pg.mkBrush('b'), pg.mkBrush('r')]
        )
        self.plot_widget.addItem(scatter)

        # Update label selections
        # labels = self.labels[self.current_file_index]
        # self.label_1.setCurrentText(labels[0])
        # self.label_2.setCurrentText(labels[1])

    def update_frame(self, value):
        self.current_frame_index = value
        self.render_current_frame()

    def save_labels(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Labels", "", "Text Files (*.txt)", options=options)
        if file_path:
            with open(file_path, 'w') as f:
                for file_idx, labels in enumerate(self.labels):
                    f.write(f"File: {os.path.basename(self.data_files[file_idx])}\n")
                    f.write(f"  Labels: {labels[0]}, {labels[1]}\n")
            QMessageBox.information(self, "Success", "Labels saved successfully.")

    def prev_file(self):
        if self.current_file_index > 0:
            self.save_current_labels()
            self.current_file_index -= 1
            self.load_current_file()

    def next_file(self):
        if self.current_file_index < len(self.data_files) - 1:
            self.save_current_labels()
            self.current_file_index += 1
            self.load_current_file()

    def save_current_labels(self):
        self.labels[self.current_file_index] = [
            self.label_1.currentText(),
            self.label_2.currentText()
        ]

    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(1000 // 30)  # Set interval for 30 FPS playback
            self.play_button.setText("Pause")

    def play_next_frame(self):
        if self.current_frame_index < self.data.shape[0] - 1:
            self.frame_slider.setValue(self.current_frame_index + 1)
        else:
            self.timer.stop()
            self.frame_slider.setValue(0)
            self.play_button.setText("Play")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tool = AnnotationTool()
    tool.show()
    sys.exit(app.exec_())
