# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:25:03 2026

@author: admin
"""

import serial
import pandas as pd
import time
import pandas as pd
import matplotlib.pyplot as plt

PORT = "COM3"
BAUD = 115200
DURATION = 5  # seconds

ser = serial.Serial(PORT, BAUD, timeout=1)

# Give Arduino time to reset
time.sleep(2)

data = []

print("Contract your muscle >>>> Recording EMG...")

start_time = time.time()

while time.time() - start_time < DURATION:

   # line = ser.readline().decode("utf-8").strip()
    line = ser.readline().decode("utf-8", errors="ignore").strip()

    try:
        t, emg = line.split(",")

        data.append([float(t), int(emg)])

    except ValueError:
        # Ignore incomplete or malformed lines
        pass

ser.close()

df = pd.DataFrame(data, columns=["Time_s", "ADC"])

df.to_csv("EMG_5sec.csv", index=False)

print(f"Saved {len(df)} samples to EMG_5sec.csv")

#___________==========__________ Plotting for 5 Seconds

df = pd.read_csv("EMG_5sec.csv")

plt.figure(figsize=(12,4))

plt.plot(df["Time_s"], df["ADC"], linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("ADC Value")
plt.title("MyoWare EMG Signal (5 s)")
plt.grid(True)

plt.show()

#___________==========__________ Plotting for 5 Secons


df["Voltage"] = df["ADC"] * 5.0 / 1023

plt.figure(figsize=(12,4))

plt.plot(df["Time_s"], df["Voltage"], linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("MyoWare EMG Signal (5 s)")
plt.grid(True)

plt.show()