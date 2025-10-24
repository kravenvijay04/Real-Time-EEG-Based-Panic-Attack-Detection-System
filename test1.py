import os
import time
import joblib
import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
from scipy.signal import welch, coherence
import tensorflow as tf
import warnings

# -------------------------------
# CLEAN OUTPUT (suppress warnings & TF logs)
# -------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow logs
warnings.filterwarnings("ignore")         # suppress warnings globally

# -------------------------------
# USER SETTINGS
# -------------------------------
PORT = "COM10"       # Arduino Port
BAUD = 115200
FS = 256             # Sampling rate (Hz)
WINDOW_SEC = 4       # Length of analysis window
SAVE_FILE = "realtime_record.csv"

# Model and Scaler
MODEL_PATH = "./models_lstm_cv/lstm_fold2_best.keras"
SCALER_PATH = "scaler.pkl"

# EEG Bands
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "highbeta": (20, 30),
    "gamma": (30, 45)
}

# Expected feature order (must match training)
FEATURE_ORDER = [
    "sex_enc","age",
    "delta.FP1","delta.FP2","theta.FP1","theta.FP2",
    "alpha.FP1","alpha.FP2","beta.FP1","beta.FP2",
    "highbeta.FP1","highbeta.FP2","gamma.FP1","gamma.FP2",
    "COH.delta.FP1.FP2","COH.theta.FP1.FP2","COH.alpha.FP1.FP2",
    "COH.beta.FP1.FP2","COH.highbeta.FP1.FP2","COH.gamma.FP1.FP2"
]

# -------------------------------
# Feature Extraction
# -------------------------------
def bandpower(data, sf, band):
    band = np.array(band)
    freqs, psd = welch(data, sf, nperseg=min(len(data), sf*2))
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    if np.any(idx_band):
        bp = np.trapz(psd[idx_band], dx=freq_res)
    else:
        bp = 0.0
    return bp

def extract_features(ch1, ch2, sf=FS):
    feats = {}
    for band, rng in bands.items():
        feats[f"{band}.FP1"] = bandpower(ch1, sf, rng)
        feats[f"{band}.FP2"] = bandpower(ch2, sf, rng)
        f, Cxy = coherence(ch1, ch2, fs=sf, nperseg=min(len(ch1), sf*2))
        idx = np.logical_and(f >= rng[0], f <= rng[1])
        feats[f"COH.{band}.FP1.FP2"] = np.mean(Cxy[idx]) if np.any(idx) else 0.0

    # Add dummy meta features
    feats["sex_enc"] = 0
    feats["age"] = 25
    return feats

from playsound import playsound

def play_alert():
    try:
        playsound("My Audio.mp3", block=True)
    except Exception:
        pass  # keep silent in presentation if audio fails

# -------------------------------
# Load Model + Scaler
# -------------------------------
scaler = joblib.load(SCALER_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

def predict_from_features(features):
    # Enforce feature order & fill missing
    for col in FEATURE_ORDER:
        if col not in features:
            features[col] = 0
    X = pd.DataFrame([features])[FEATURE_ORDER]

    # Log transform & scale
    X = np.log1p(X)
    X_scaled = scaler.transform(X)
    X_input = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    prob = model.predict(X_input, verbose=0)[0][0]
    pred = int(prob > 0.95)
    if pred == 1:   # PANIC detected
        play_alert()
    return prob, pred

# -------------------------------
# OPTION 1: Test on CSV
# -------------------------------
def test_from_csv(file_path):
    df = pd.read_csv(file_path)
    ch1 = df["ch1_filt"].values[-FS*WINDOW_SEC:]
    ch2 = df["ch2_filt"].values[-FS*WINDOW_SEC:]
    feats = extract_features(ch1, ch2)
    prob, pred = predict_from_features(feats)
    print(f"\nCSV Prediction: {'PANIC' if pred else 'CALM'}")

# -------------------------------
# OPTION 2: Test on Dataset
# -------------------------------
def test_from_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df[df["main.disorder"].isin(["Anxiety disorder","Healthy control"])]
    labels = df["main.disorder"].map({"Healthy control":0,"Anxiety disorder":1}).values

    selected_features = [c for c in FEATURE_ORDER if c in df.columns]
    X = df[selected_features].copy()

    for col in FEATURE_ORDER:
        if col not in X.columns:
            X[col] = 0

    X = X[FEATURE_ORDER]
    X = np.log1p(X)
    X_scaled = scaler.transform(X)
    X_input = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    probs = model.predict(X_input, verbose=0).ravel()
    preds = (probs > 0.5).astype(int)
    acc = np.mean(preds == labels)
    print(f"\nCSV Prediction= {'PANIC' if acc==1 else 'CALM'}")
    if acc==1:
        play_alert()

# -------------------------------
# OPTION 3: Real-Time Recording + Testing
# -------------------------------
def record_and_test():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("📡 Recording started... Press CTRL+C to stop.")

    ch1_buf = deque(maxlen=FS*WINDOW_SEC)
    ch2_buf = deque(maxlen=FS*WINDOW_SEC)

    nyq = 0.5 * FS
    b_bp, a_bp = signal.butter(4, [0.5/nyq, 40/nyq], btype="band")
    b_notch, a_notch = signal.iirnotch(50/nyq, 30)

    try:
        while True:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if "," not in line:
                continue
            try:
                v1, v2 = [int(x) for x in line.split(",")]
            except:
                continue

            ch1_buf.append(v1)
            ch2_buf.append(v2)

            if len(ch1_buf) >= FS*WINDOW_SEC:
                ch1 = signal.filtfilt(b_notch, a_notch, np.array(ch1_buf))
                ch1 = signal.filtfilt(b_bp, a_bp, ch1)
                ch2 = signal.filtfilt(b_notch, a_notch, np.array(ch2_buf))
                ch2 = signal.filtfilt(b_bp, a_bp, ch2)

                feats = extract_features(ch1, ch2)
                prob, pred = predict_from_features(feats)
                print(f"\nCSV Prediction={'PANIC' if pred else 'CALM'}")

    except KeyboardInterrupt:
        print("\n🛑 Recording stopped by user.")
    finally:
        ser.close()

# -------------------------------
# MENU
# -------------------------------
def main():
    print("\nChoose Mode:")
    print("1 - Test on recorded EEG CSV file")
    print("2 - Test on dataset with labels")
    print("3 - Record real-time EEG and test")
    choice = input("Enter choice: ")

    if choice == "1":
        path = input("Enter CSV path (recorded EEG): ")
        test_from_csv(path)
    elif choice == "2":
        path = input("Enter dataset CSV path: ")
        test_from_dataset(path)
    elif choice == "3":
        record_and_test()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
