# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:21:59 2026

@author: AS
"""
## Aim: realtime raw and envlope MEG

   
import sys
import time
import collections
import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import pyqtgraph as pg

# ==========================================
# 1. EMG Signal Processing Utility
# ==========================================
class RealtimeEMGFilter:
    def __init__(self, fs=1000.0):
        self.fs = fs
       
        # 4th-order Bandpass filter (20 - 450 Hz)
        low, high = 20.0 / (0.5 * fs), 450.0 / (0.5 * fs)
        self.b_bp, self.a_bp = butter(4, [low, high], btype='band')
        self.zi_bp = lfilter_zi(self.b_bp, self.a_bp)
       
        # 60 Hz Notch filter (Powerline noise)
        w0 = 60.0 / (0.5 * fs)
        self.b_notch, self.a_notch = iirnotch(w0, Q=30.0)
        self.zi_notch = lfilter_zi(self.b_notch, self.a_notch)

    def process(self, chunk):
        """Processes a chunk of raw voltage samples in real time."""
        if len(chunk) == 0:
            return np.array([]), np.array([])
       
        # Apply Bandpass
        filtered, self.zi_bp = lfilter(self.b_bp, self.a_bp, chunk, zi=self.zi_bp)
       
        # Apply Notch
        filtered, self.zi_notch = lfilter(self.b_notch, self.a_notch, filtered, zi=self.zi_notch)
       
        # Full-Wave Rectification & Linear Envelope
        rectified = np.abs(filtered)
        envelope = np.convolve(rectified, np.ones(50)/50, mode='same')
       
        return filtered, envelope


# ==========================================
# 2. OpenScope Data Acquisition Thread
# ==========================================
class OpenScopeWorker(QThread):
    data_received = pyqtSignal(np.ndarray)

    def __init__(self, sample_rate=1000):
        super().__init__()
        self.running = False
        self.fs = sample_rate

    def run(self):
        self.running = True
       
        # Replace this simulation block with your OpenScope API/Serial reads:
        # e.g., raw_bytes = serial_port.read(...)
        t_step = 1.0 / self.fs
        t = 0.0
       
        while self.running:
            # Generate simulated EMG burst data (chunk size = 20 samples)
            chunk_size = 20
            time_chunk = np.linspace(t, t + (chunk_size * t_step), chunk_size, endpoint=False)
            t += chunk_size * t_step
           
            # Synthetic EMG signal: noise + periodic muscle contraction spikes
            noise = np.random.normal(0, 0.05, chunk_size)
            burst = 0.5 * np.sin(2 * np.pi * 100 * time_chunk) if (int(t) % 3 == 0) else 0.0
            raw_chunk = burst + noise
           
            self.data_received.emit(raw_chunk)
            time.sleep(0.02)  # Simulate ~50 FPS acquisition rate

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# 3. Main Real-time GUI Window
# ==========================================
class EMGMonitorGUI(QMainWindow):
    def __init__(self, buffer_size=2000, fs=1000):
        super().__init__()
       
        self.fs = fs
        self.buffer_size = buffer_size
        self.filter = RealtimeEMGFilter(fs=fs)
       
        # Circular buffers for raw & envelope visualization
        self.raw_buffer = collections.deque(maxlen=buffer_size)
        self.env_buffer = collections.deque(maxlen=buffer_size)
       
        # Initialize buffer with zeros
        self.raw_buffer.extend([0.0] * buffer_size)
        self.env_buffer.extend([0.0] * buffer_size)

        self.init_ui()
       
        # Worker Thread Setup
        self.worker = OpenScopeWorker(sample_rate=fs)
        self.worker.data_received.connect(self.on_data_received)

    def init_ui(self):
        self.setWindowTitle("OpenScope EMG Real-Time Monitor")
        self.resize(1000, 600)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Control Panel Header
        control_layout = QHBoxLayout()
       
        self.btn_start = QPushButton("Start Acquisition")
        self.btn_start.clicked.connect(self.start_acquisition)
        control_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_acquisition)
        control_layout.addWidget(self.btn_stop)
       
        self.lbl_status = QLabel("Status: Idle")
        control_layout.addWidget(self.lbl_status)
       
        layout.addLayout(control_layout)

        # Plot Widgets (PyQtGraph)
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        # Plot 1: Filtered Raw EMG
        self.plot_raw = self.plot_widget.addPlot(title="Filtered Signal (20-450 Hz + Notch)")
        self.plot_raw.setYRange(-1.5, 1.5)
        self.plot_raw.showGrid(x=True, y=True)
        self.curve_raw = self.plot_raw.plot(pen=pg.mkPen('c', width=1.5))

        self.plot_widget.nextRow()

        # Plot 2: Linear Envelope (Muscle Activation Strength)
        self.plot_env = self.plot_widget.addPlot(title="Linear Envelope (Activation Magnitude)")
        self.plot_env.setYRange(0, 1.5)
        self.plot_env.showGrid(x=True, y=True)
        self.curve_env = self.plot_env.plot(pen=pg.mkPen('r', width=2))

        # Refresh Timer for Plot Rendering (~30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(33)

    def start_acquisition(self):
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Status: Streaming...")

    def stop_acquisition(self):
        self.worker.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Status: Stopped")

    def on_data_received(self, chunk):
        filtered_chunk, env_chunk = self.filter.process(chunk)
        self.raw_buffer.extend(filtered_chunk)
        self.env_buffer.extend(env_chunk)

    def update_plots(self):
        self.curve_raw.setData(np.array(self.raw_buffer))
        self.curve_env.setData(np.array(self.env_buffer))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EMGMonitorGUI()
    gui.show()
    sys.exit(app.exec()) 