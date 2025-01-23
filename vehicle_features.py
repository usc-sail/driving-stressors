import os, pandas as pd
from tqdm import tqdm

raw_path = "/media/data/toyota/toyota_no_video/"
folder = "mirise_features_1sec"
os.makedirs(folder, exist_ok=True)

for p in tqdm(os.listdir(raw_path)):
    ppath = os.path.join(raw_path, p)
    for condition in os.listdir(ppath):
        cpath = os.path.join(ppath, condition)

        driving = os.path.join(cpath, "UnityDrivingSignals.csv")
        df = pd.read_csv(driving, engine="python")
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df.resample("50ms").mean().interpolate()

        sr = 20
        win = 60 * sr
        sm = 31

        # driving speed
        speed = df["vsp"]

        clean_speed = []
        for i in range(0, len(speed), sr):
            window = speed.iloc[i - win : i + win]
            clean_speed.append(window.mean())

        vsp = pd.Series(clean_speed)
        vsp = vsp.rolling(window=sm).mean()

        # steering angle (abs)
        steering = abs(df["sta"])

        clean_steering = []
        for i in range(0, len(steering), sr):
            window = steering.iloc[i - win : i + win]
            clean_steering.append(window.mean())

        sta = pd.Series(clean_steering)
        sta = sta.rolling(window=sm).mean()

        # acceleration pedal
        acc = df["acc"]
        acc[acc < 0] = 0
        acc[acc > 1] = 1
        acc = 1 - acc

        clean_acc = []
        for i in range(0, len(acc), sr):
            window = acc.iloc[i - win : i + win]
            clean_acc.append(window.mean())

        acc = pd.Series(clean_acc)
        acc = acc.rolling(window=sm).mean()

        # brake pedal
        brake = df["brk"]
        brake = (brake + 1) / 2
        brake = 1 - brake

        clean_brake = []
        for i in range(0, len(brake), sr):
            window = brake.iloc[i - win : i + win]
            clean_brake.append(window.mean())

        brk = pd.Series(clean_brake)
        brk = brk.rolling(window=sm).mean()

        # save to file
        driving_dct = {"vsp": vsp, "sta": sta, "acc": acc, "brk": brk}
        driving_df = pd.DataFrame(driving_dct)
        driving_df = driving_df.bfill().ffill()
        driving_df.to_csv(f"{folder}/{p}_{condition}_driving.csv")
