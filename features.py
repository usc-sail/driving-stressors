import numpy as np, pandas as pd
import os, neurokit2 as nk
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
raw_path = "/media/data/toyota/raw_data/trina_33/"
datapath = "/media/data/toyota/processed_data/trina_33_wdrive/"
samples_path = "/media/data/toyota/processed_data/trina_33_samples_wdrive/"
os.makedirs(samples_path, exist_ok=True)


def load_session(sub, mode):
    """
    Load a session from the Toyota dataset
    """
    session = pd.read_csv(f"{datapath}/P{sub}_{mode}.csv", low_memory=False)
    session["time"] = pd.to_datetime(session["time"], unit="s")
    return session


def load_driving_signals(sub, mode, session):
    """
    Load the driving signals from the Toyota dataset
    """
    drive_file = os.path.join(raw_path, f"P{sub}", mode, "General.csv")
    if not os.path.exists(drive_file):
        drive_file = os.path.join(raw_path, f"P{sub}", "General.csv")
        if not os.path.exists(drive_file):
            print(f"No driving data found for P{sub} in {mode}.")
            return None

    driving_data = pd.read_csv(drive_file, low_memory=False)
    driving_data = driving_data[
        [
            "time",
            "velocity (m/s)",
            "revolutions per minute",
            "rotation of the steering wheel",
            "throttle",
            "brake",
        ]
    ]
    driving_data["time"] = pd.to_datetime(driving_data["time"], unit="s")
    driving_data = driving_data.set_index("time")

    # resample driving data to match session time
    driving_data = driving_data.resample("0.5ms").ffill().bfill()

    # align to session: start is the "Collection Start" event
    start_time = session[session["event"] == "Collection Start"].index[0]
    end_time = start_time + len(driving_data)
    if end_time > session.index[-1]:
        # Sessions: 11_I, 14_IM, 7_I, 13_I, 27_S
        print("Driving data is longer than session data")
        end_time = session.index[-1]
        driving_data = driving_data.iloc[: end_time - start_time]

    # Create full zero-filled DataFrame
    driving_full = pd.DataFrame(
        data=np.zeros((len(session), len(driving_data.columns))),
        columns=driving_data.columns,
    )
    # Insert driving data at the right position
    driving_full.iloc[start_time:end_time] = driving_data.values

    # Merge with session
    session = session.reset_index(drop=True).join(driving_full.reset_index(drop=True))
    return session


def get_video_baseline(session, air):
    """
    Get the ~10-min baseline of video watching
    """
    video_start = session[session["event"] == "Video Start"].index[0]
    video_end = session[session["event"] == "Video End"].index[0]
    return session.loc[video_start - air : video_end + air]


def get_main_driving(session, air):
    """
    Get the main driving session
    """
    # first align driving data with event "Collection Start"
    exp_start = session[session["event"] == "Experiment Start"].index[0]
    exp_end = session[session["event"] == "Experiment End"].index[0]
    return session.loc[exp_start - air : exp_end + air]


def get_driving_baseline(main_driving, air):
    """
    Get the free driving baseline before events
    """
    # drop row with event "Experiment Start"
    main_driving = main_driving.drop(
        main_driving[main_driving["event"] == "Experiment Start"].index
    )
    # drop row with event "Collection Start"
    main_driving = main_driving.drop(
        main_driving[main_driving["event"] == "Collection Start"].index
    )
    event_start = main_driving[main_driving["event"].notna()].index[0]
    return main_driving.loc[: event_start + air]


def get_event_driving(main_driving, air, recovery=False):
    """
    Get the driving session during events
    """
    max_patience = 5 * 60 * 2000
    # drop row with event "Experiment Start"
    main_driving = main_driving.drop(
        main_driving[main_driving["event"] == "Experiment Start"].index
    )
    # drop row with event "Collection Start"
    main_driving = main_driving.drop(
        main_driving[main_driving["event"] == "Collection Start"].index
    )

    main_start = main_driving[main_driving["event"].notna()].index[0]
    if main_start > main_driving.index[0] + max_patience:
        main_start = main_driving.index[0] + max_patience

    if recovery:
        # end is end of experiment
        return main_driving.loc[main_start - air :]
    else:
        # end is when the "Clear Text" appears
        try:
            main_end = main_driving[main_driving["event"] == "Clear Text"].index[0]
            return main_driving.loc[main_start - air : main_end + air]
        except IndexError:
            print("No Clear Text found, using end of experiment")
            return main_driving.loc[main_start - air :]


def get_recovery_driving(main_driving, air):
    """
    Get the driving session during recovery
    """
    max_patience = 60 * 2000
    try:
        main_start = main_driving[main_driving["event"] == "Clear Text"].index[0]
    except IndexError:
        print("No Clear Text found, using end of experiment")
        main_start = main_driving.index[-1] - max_patience
    return main_driving.loc[main_start - air :]


def get_ecg_features(ecg_rec, ppg_rec, rsp_rec, sr):
    try:
        ppg_signals, _ = nk.ppg_process(ppg_rec, sampling_rate=sr)
        ppg_hr = ppg_signals["PPG_Rate"].mean()
    except:
        ppg_hr = 2048.0

    try:
        ecg_signals, info = nk.ecg_process(ecg_rec, sampling_rate=sr)
        ecg_hr = ecg_signals["ECG_Rate"].mean()
    except:
        ecg_hr = ppg_hr

    hr = (ecg_hr + ppg_hr) / 2
    hr = hr.mean()
    hr = min(hr, 200)  # max 200 bpm
    hr = max(hr, 40)  # min 40 bpm

    try:
        assert ecg_hr != ppg_hr
        rsp_signals, rsp_info = nk.rsp_process(rsp_rec, sampling_rate=sr)
    except:
        return hr, 2048.0, 2048.0, 2048.0, 2048.0

    # calculate the RSP period
    troughs = rsp_info["RSP_Troughs"]
    if len(troughs) < 2:
        rsp_period = 10.0
    else:
        rsp_diff = np.diff(rsp_info["RSP_Troughs"])
        rsp_period = np.mean(rsp_diff) / sr
        # rsp_period = max(rsp_period, 2)  # min 2s
        # rsp_period = min(rsp_period, 10)  # max 10s

    rsp_depth = rsp_signals["RSP_Amplitude"].mean()
    rvt = rsp_signals["RSP_RVT"].mean()
    rsa = nk.hrv_rsa(
        ecg_signals,
        rsp_signals,
        info,
        window=len(ecg_rec) // sr,
        sampling_rate=sr,
        continuous=False,
    )
    rsa = rsa["RSA_P2T_Mean"]

    return hr, rsa, rsp_period, rsp_depth, rvt


def get_eda_features(eda_rec, sr):
    eda, _ = nk.eda_process(eda_rec, sampling_rate=sr)

    scl = eda["EDA_Tonic"]
    scl_mean = np.mean(scl)
    scl_slope = np.polyfit(np.arange(len(scl)), scl, 1)[0]

    scr_indices = np.where(eda["SCR_Peaks"])[0]
    if len(scr_indices) < 2:
        scr_dist = 20.0
    else:
        scr_dist = np.mean(np.diff(scr_indices)) / sr

    scr_amp = eda["SCR_Amplitude"]
    scr_amp = scr_amp[scr_indices].mean()
    scr_rise = eda["SCR_RiseTime"]
    scr_rise = scr_rise[scr_indices].mean()

    # scr_amp = min(scr_amp, 8)  # max 8μS
    # scr_rise = min(scr_rise, 5)  # max 5s
    return (
        scl_mean,
        scl_slope,
        scr_dist,
        scr_amp,
        scr_rise,
    )


def get_skt_features(skt_rec):
    skt_mean = np.mean(skt_rec)
    skt_slope = np.polyfit(np.arange(len(skt_rec)), skt_rec, 1)[0]
    return skt_mean, skt_slope


def get_driving_features(drv_rec):
    # mean velocity
    vel_mean = drv_rec["velocity (m/s)"].mean()
    # std velocity
    vel_std = drv_rec["velocity (m/s)"].std()
    # std of RPM
    rpm_std = drv_rec["revolutions per minute"].std()
    # std of steering
    str_std = drv_rec["rotation of the steering wheel"].std()
    # entropy of steering
    str_entropy = nk.entropy_sample(
        drv_rec["rotation of the steering wheel"], sampling_rate=10
    )[0]
    str_entropy = 0.0 if str_entropy == -0.0 else str_entropy

    throttle = drv_rec["throttle"].values
    # throttle rate
    throttle_rate = (throttle > 0).sum() / len(drv_rec)
    # throttle magnitude
    throttle_magnitude = (
        np.mean(throttle[throttle > 0]) if len(throttle[throttle > 0]) > 0 else 0
    )
    # throttle entropy
    throttle_entropy = nk.entropy_sample(throttle, sampling_rate=10)[0]
    throttle_entropy = 0.0 if throttle_entropy == -0.0 else throttle_entropy

    brake = drv_rec["brake"].values
    # brake rate
    brake_rate = (brake > 0).sum() / len(drv_rec)
    # brake magnitude
    brake_magnitude = np.mean(brake[brake > 0]) if len(brake[brake > 0]) > 0 else 0
    # brake entropy
    brake_entropy = nk.entropy_sample(brake, sampling_rate=10)[0]
    brake_entropy = 0.0 if brake_entropy == -0.0 else brake_entropy

    return {
        "vel_mean": vel_mean,
        "vel_std": vel_std,
        "rpm_std": rpm_std,
        "str_std": str_std,
        "str_entropy": str_entropy,
        "throttle_rate": throttle_rate,
        "throttle_magnitude": throttle_magnitude,
        "throttle_entropy": throttle_entropy,
        "brake_rate": brake_rate,
        "brake_magnitude": brake_magnitude,
        "brake_entropy": brake_entropy,
    }


def feature_extraction(session, name, sr):
    """
    Extract features from a session
    """
    ppg_rec = session["ppg"].values
    ecg_rec = session["ecg"].values
    eda_rec = session["eda"].values
    rsp_rec = session["rsp"].values
    skt_rec = session["skt"].values

    drv_cols = [
        "velocity (m/s)",
        "revolutions per minute",
        "rotation of the steering wheel",
        "throttle",
        "brake",
    ]
    drv_rec = session[drv_cols].values
    print(session[drv_cols])

    if len(ppg_rec) < 30 * sr:
        print("Recording too short")
        return None

    features = []
    for i in tqdm(range(0, len(ppg_rec), sr)):
        ppg_segment = ppg_rec[i - sr * 15 : i + sr * 15]
        ecg_segment = ecg_rec[i - sr * 15 : i + sr * 15]
        eda_segment = eda_rec[i - sr * 15 : i + sr * 15]
        rsp_segment = rsp_rec[i - sr * 15 : i + sr * 15]
        skt_segment = skt_rec[i - sr * 15 : i + sr * 15]
        drv_segment = drv_rec[i - sr * 15 : i + sr * 15]

        if len(ppg_segment) == 30 * sr:
            ecg_feats = get_ecg_features(ecg_segment, ppg_segment, rsp_segment, sr)
            eda_feats = get_eda_features(eda_segment, sr)
            skt_feats = get_skt_features(skt_segment)
            drv_feats = get_driving_features(
                pd.DataFrame(drv_segment, columns=drv_cols)
            )

            hr, rsa, rsp_rate, rsp_depth, rvt = ecg_feats
            scl_mean, scl_slope, scr_peaks, scr_amp, scr_rise = eda_feats
            skt_mean, skt_slope = skt_feats

            prev = features[-1] if len(features) > 0 else [np.nan] * 11
            if np.isnan(hr):
                hr = prev[0]

            if np.isnan(scr_amp):
                scr_amp = prev[4]
                scr_rise = prev[5]

            if np.isnan(rsp_depth):
                rsp_depth = prev[7]
                rvt = prev[8]

            if np.isnan(skt_slope) or np.isnan(scl_slope):
                raise ValueError("SKT slope is NaN")

            segment_features = {
                "hr": hr,
                "rsa": rsa,
                "scl_mean": scl_mean,
                "scl_slope": scl_slope,
                "scr_peaks": scr_peaks,
                "scr_amp": scr_amp,
                "scr_rise": scr_rise,
                "rsp_rate": rsp_rate,
                "rsp_depth": rsp_depth,
                "rvt": rvt,
                "skt_mean": skt_mean,
                "skt_slope": skt_slope,
            }
            print(drv_feats)
            segment_features.update(drv_feats)

            nan_token = 2048.0
            feat_vector = list(segment_features.values())
            feat_vector = [nan_token if np.isnan(f) else f for f in feat_vector]
            features.append(feat_vector)

    if len(features) > 0:
        features = np.stack(features, axis=0)
        print(features.shape)
        np.save(f"{samples_path}/{name}.npy", features)


if __name__ == "__main__":

    air, sr = 15, 250

    for session_title in tqdm(os.listdir(datapath)):
        sub, mode = session_title[1:-4].split("_")
        print(f"Subject {sub}, mode {mode}")

        if (
            os.path.exists(f"{samples_path}/P{sub}_{mode}_event.npy")
            and os.path.exists(f"{samples_path}/P{sub}_{mode}_free.npy")
            and os.path.exists(f"{samples_path}/P{sub}_{mode}_video.npy")
        ) or int(sub) in [3, 8]:
            pass

        session = load_session(int(sub), mode)
        session["rsp"] = session.filter(like="rsp").mean(axis=1)
        session = load_driving_signals(int(sub), mode, session)

        video_baseline = get_video_baseline(session, air * 2000)
        driving_session = get_main_driving(session, air * 2000)
        free_driving = get_driving_baseline(driving_session, air * 2000)
        event_driving = get_event_driving(driving_session, air * 2000)
        recovery_driving = get_recovery_driving(driving_session, air * 2000)

        video_baseline = video_baseline.drop(columns="event").set_index("time")
        free_driving = free_driving.drop(columns="event").set_index("time")
        event_driving = event_driving.drop(columns="event").set_index("time")
        recovery_driving = recovery_driving.drop(columns="event").set_index("time")

        # derive msec from sr
        msec = f"{int(1000/sr)}ms"
        video_baseline = video_baseline.resample(msec).ffill().bfill()
        free_driving = free_driving.resample(msec).ffill().bfill()
        event_driving = event_driving.resample(msec).ffill().bfill()
        recovery_driving = recovery_driving.resample(msec).ffill().bfill()

        feature_extraction(video_baseline, f"P{sub}_{mode}_video", sr)
        feature_extraction(free_driving, f"P{sub}_{mode}_free", sr)
        feature_extraction(event_driving, f"P{sub}_{mode}_event", sr)
        feature_extraction(recovery_driving, f"P{sub}_{mode}_recov", sr)
