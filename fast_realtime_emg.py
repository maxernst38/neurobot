# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 01:00:49 2026

@author: AS
"""

## V5
## Aim 1:making a GUI for real time openscope for  monitoring EMG
## Aim 2:making raw and envlope emg
## Aim 3:using PyQt6 for the user interface (more advanced compared to tkinter)
## Aim 3:using PyQtGraph for fast, real-time data visualization in scientific, mathematical, and engineering applications
   
   
import sys
import collections
import time
import numpy as np
import pandas as pd
import serial

from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox
)
import pyqtgraph as pg


# ==============================================================================
# 1. Real-Time EMG Signal Processing (Filtering & Envelope Calculation)
# ==============================================================================
class RealTimeEMGProcessor:
    def __init__(self, fs=1000.0):
        self.fs = fs

        # 4th Order Bandpass (20 - 450 Hz)
        low, high = 20.0 / (0.5 * fs), 450.0 / (0.5 * fs)
        self.b_bp, self.a_bp = butter(4, [low, high], btype="band")
        self.zi_bp = lfilter_zi(self.b_bp, self.a_bp)

        # 60 Hz Notch Filter (Powerline Noise Removal)
        w0 = 60.0 / (0.5 * fs)
        self.b_notch, self.a_notch = iirnotch(w0, Q=30.0)
        self.zi_notch = lfilter_zi(self.b_notch, self.a_notch)

        # Lowpass Filter for Envelope (5 Hz)
        low_env = 5.0 / (0.5 * fs)
        self.b_env, self.a_env = butter(2, low_env, btype="low")
        self.zi_env = lfilter_zi(self.b_env, self.a_env)

    def process_sample(self, raw_voltage):
        """Processes a single voltage sample through bandpass, notch, and envelope filters."""
        # Bandpass
        filtered, self.zi_bp = lfilter(
            self.b_bp, self.a_bp, [raw_voltage], zi=self.zi_bp
        )
        # Notch
        filtered, self.zi_notch = lfilter(
            self.b_notch, self.a_notch, filtered, zi=self.zi_notch
        )

        # Full-Wave Rectification
        rectified = np.abs(filtered[0])

        # Envelope
        envelope, self.zi_env = lfilter(
            self.b_env, self.a_env, [rectified], zi=self.zi_env
        )

        return filtered[0], envelope[0]


# ==============================================================================
# 2. Non-Blocking Serial Reader Worker Thread
# ==============================================================================
class SerialWorker(QThread):
    # Emits: time_s, adc_val, voltage_val
    data_received = pyqtSignal(float, float, float)
    connection_error = pyqtSignal(str)

    def __init__(self, port="COM3", baud=115200):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = False

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # Give microcontroller time to reset
            self.running = True

            while self.running:
                if ser.in_waiting:
                    line = (
                        ser.readline().decode("utf-8", errors="ignore").strip()
                    )
                    try:
                        t_str, emg_str = line.split(",")
                        t = float(t_str)
                        adc = int(emg_str)
                        # Converting ADC to Voltage (matches base code)
                        voltage = adc * 5.0 / 1023.0

                        self.data_received.emit(t, adc, voltage)
                    except ValueError:
                        # Skip corrupted or partial lines
                        pass

            ser.close()
        except Exception as e:
            self.connection_error.emit(str(e))

    def stop(self):
        self.running = False
        self.wait()


# ==============================================================================
# 3. Main PyQt6 Real-Time Monitoring GUI
# ==============================================================================
class EMGMonitorGUI(QMainWindow):
    def __init__(self, window_samples=2000):
        super().__init__()

        self.window_samples = window_samples
        self.processor = RealTimeEMGProcessor(fs=1000.0)

        # Circular buffers for live plotting window
        self.time_buf = collections.deque(maxlen=window_samples)
        self.raw_buf = collections.deque(maxlen=window_samples)
        self.env_buf = collections.deque(maxlen=window_samples)

        # Storage for saving complete sessions to Pandas DataFrame / CSV
        self.recorded_data = []

        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("OpenScope / MyoWare EMG Real-Time Monitor")
        self.resize(1100, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Control Panel Controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Port:"))
        self.txt_port = QLineEdit("COM3")
        self.txt_port.setFixedWidth(80)
        controls_layout.addWidget(self.txt_port)

        controls_layout.addWidget(QLabel("Baud:"))
        self.txt_baud = QLineEdit("115200")
        self.txt_baud.setFixedWidth(80)
        controls_layout.addWidget(self.txt_baud)

        self.btn_start = QPushButton("Start Streaming")
        self.btn_start.clicked.connect(self.start_streaming)
        controls_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_streaming)
        controls_layout.addWidget(self.btn_stop)

        self.btn_save = QPushButton("Save CSV")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_csv)
        controls_layout.addWidget(self.btn_save)

        self.lbl_status = QLabel("Status: Idle")
        controls_layout.addWidget(self.lbl_status)
        controls_layout.addStretch()

        main_layout.addLayout(controls_layout)

        # PyQtGraph Plot Layout
        pg.setConfigOptions(antialias=True)
        self.plot_layout = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.plot_layout)

        # Plot 1: Raw Voltage Signal
        self.p_raw = self.plot_layout.addPlot(
            title="Raw EMG Signal (Voltage V)"
        )
        self.p_raw.setLabel("left", "Voltage", "V")
        self.p_raw.showGrid(x=True, y=True)
        self.curve_raw = self.p_raw.plot(pen=pg.mkPen("c", width=1.5))

        self.plot_layout.nextRow()

        # Plot 2: Linear Envelope Signal
        self.p_env = self.plot_layout.addPlot(
            title="EMG Linear Envelope (Activation Strength)"
        )
        self.p_env.setLabel("left", "Amplitude", "V")
        self.p_env.setLabel("bottom", "Time", "s")
        self.p_env.showGrid(x=True, y=True)
        self.curve_env = self.p_env.plot(pen=pg.mkPen("r", width=2))

        # Synchronize X-axes between both plots
        self.p_env.setXLink(self.p_raw)

        # UI Plot Update Timer (~30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)

    def start_streaming(self):
        port = self.txt_port.text().strip()
        try:
            baud = int(self.txt_baud.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid Baud Rate!")
            return

        # Clear existing buffers
        self.time_buf.clear()
        self.raw_buf.clear()
        self.env_buf.clear()
        self.recorded_data.clear()

        # Initialize and Start Worker
        self.worker = SerialWorker(port=port, baud=baud)
        self.worker.data_received.connect(self.on_data_received)
        self.worker.connection_error.connect(self.on_connection_error)
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.lbl_status.setText(f"Status: Streaming from {port}...")

        self.timer.start(33)

    def stop_streaming(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(len(self.recorded_data) > 0)
        self.lbl_status.setText("Status: Stopped")

    def on_data_received(self, t, adc, voltage):
        # Calculate envelope signal sample
        filtered_v, env_v = self.processor.process_sample(voltage)

        # Update circular display buffers
        self.time_buf.append(t)
        self.raw_buf.append(voltage)
        self.env_buf.append(env_v)

        # Log complete session data
        self.recorded_data.append([t, adc, voltage, filtered_v, env_v])

    def update_plots(self):
        if len(self.time_buf) > 0:
            t_data = np.array(self.time_buf)
            self.curve_raw.setData(t_data, np.array(self.raw_buf))
            self.curve_env.setData(t_data, np.array(self.env_buf))

    def on_connection_error(self, err_msg):
        self.stop_streaming()
        QMessageBox.critical(self, "Serial Error", f"Failed to connect:\n{err_msg}")

    def save_csv(self):
        if not self.recorded_data:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save EMG Data", "EMG_Recorded_Data.csv", "CSV Files (*.csv)"
        )
        if file_path:
            df = pd.DataFrame(
                self.recorded_data,
                columns=["Time_s", "ADC", "Voltage", "Filtered_V", "Envelope_V"],
            )
            df.to_csv(file_path, index=False)
            self.lbl_status.setText(f"Saved {len(df)} samples to {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EMGMonitorGUI()
    gui.show()
    sys.exit(app.exec())