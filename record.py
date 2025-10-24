import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
import csv

# ----------------- USER SETTINGS -----------------
PORT = "COM10"       # Change to your Arduino port
BAUD = 115200
SAMPLE_RATE = 500    # must match Arduino
WINDOW_SEC = 5       # length of plot window in seconds
SAVE_FILE = "eeg_record.csv"

# Fixed meta values
SEX_ENC = 0
AGE = 25
# -------------------------------------------------

# Design filters
def design_filters(fs):
    nyq = 0.5 * fs
    # Bandpass 0.5–40 Hz
    low = 0.5 / nyq
    high = 40.0 / nyq
    b_bp, a_bp = signal.butter(4, [low, high], btype="band")
    # 50 Hz notch
    notch_freq = 50.0
    q = 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq / nyq, q)
    return b_bp, a_bp, b_notch, a_notch

b_bp, a_bp, b_notch, a_notch = design_filters(SAMPLE_RATE)

# Serial
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # wait for Arduino reset

# Buffers
maxlen = WINDOW_SEC * SAMPLE_RATE
ch1_buf = deque(maxlen=maxlen)
ch2_buf = deque(maxlen=maxlen)
t_buf = deque(maxlen=maxlen)

# --- Plot Style ---
plt.style.use("seaborn-v0_8-darkgrid")
plt.ion()
fig, axs = plt.subplots(2, 1, figsize=(12,6), sharex=True)
line1, = axs[0].plot([], [], color="royalblue", lw=1.2, antialiased=True, label="CH1 (FP1)")
line2, = axs[1].plot([], [], color="darkorange", lw=1.2, antialiased=True, label="CH2 (FP2)")

axs[0].set_ylabel("EEG (µV)")
axs[1].set_ylabel("EEG (µV)")
axs[1].set_xlabel("Time (s)")

axs[0].legend(loc="upper right")
axs[1].legend(loc="upper right")
axs[0].set_title("Real-Time EEG Signals")

# Pre-set limits (for cleaner view, adjust if needed)
axs[0].set_ylim(-200, 200)
axs[1].set_ylim(-200, 200)

plt.tight_layout()

# Save file
csvfile = open(SAVE_FILE, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["time","ch1_raw","ch2_raw","ch1_filt","ch2_filt","sex_enc","age"])

start = time.time()

try:
    while True:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if "," not in line:
            continue
        try:
            v1, v2 = [int(x) for x in line.split(",")]
        except:
            continue

        t = time.time() - start
        ch1_buf.append(v1)
        ch2_buf.append(v2)
        t_buf.append(t)

        if len(ch1_buf) > 30:   # only filter after ~30 samples
            ch1 = np.array(ch1_buf)
            ch2 = np.array(ch2_buf)

            # Apply notch then bandpass
            ch1_notch = signal.filtfilt(b_notch, a_notch, ch1)
            ch1_filt = signal.filtfilt(b_bp, a_bp, ch1_notch)
            ch2_notch = signal.filtfilt(b_notch, a_notch, ch2)
            ch2_filt = signal.filtfilt(b_bp, a_bp, ch2_notch)

            # Update plot (avoid rescaling to reduce flicker)
            t_plot = np.array(t_buf) - t_buf[0]
            line1.set_data(t_plot, ch1_filt)
            line2.set_data(t_plot, ch2_filt)

            axs[0].set_xlim(t_plot.min(), t_plot.max())
            axs[1].set_xlim(t_plot.min(), t_plot.max())

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Save last row including meta features
            writer.writerow([t, v1, v2, ch1_filt[-1], ch2_filt[-1], SEX_ENC, AGE])
            csvfile.flush()
except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()
    csvfile.close()
