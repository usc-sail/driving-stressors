import numpy as np, pandas as pd
import os, neurokit2 as nk
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
datapath = "/media/data/toyota/processed_data/trina_33"
samples_path = "/media/data/toyota/processed_data/trina_33_samples_tokens"
os.makedirs(samples_path, exist_ok=True)


def load_session(sub, mode):
    """
    Load a session from the Toyota dataset
    """
    session = pd.read_csv(f"{datapath}/P{sub}_{mode}.csv", low_memory=False)
    session["time"] = pd.to_datetime(session["time"], unit="s")
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


def get_ppg_features(ppg_rec, sr):
    """
    Get the PPG-derived heart rate
    """
    ppg_r = nk.ppg_findpeaks(ppg_rec, sampling_rate=sr)
    return nk.ppg_rate(ppg_r, sampling_rate=sr)


def get_ecg_features(ecg_rec, sr):
    """
    Get ECG-derived features
    """
    ecg = nk.ecg_findpeaks(ecg_rec, sampling_rate=sr)
    hr = nk.ecg_rate(ecg, sampling_rate=sr)
    hrv_t = nk.hrv_time(ecg, sampling_rate=sr)
    hrv_f = nk.hrv_frequency(ecg, sampling_rate=sr)
    hrv = {**hrv_t, **hrv_f}
    return (
        hr.mean(),
        hrv["HRV_SDNN"].values[0],
        hrv["HRV_SDSD"].values[0],
        hrv["HRV_RMSSD"].values[0],
        hrv["HRV_HF"].values[0],
    )


def get_eda_features(eda_rec, sr):
    """
    Get EDA-derived features
    """
    eda, _ = nk.eda_process(eda_rec, sampling_rate=sr)

    scl = eda["EDA_Tonic"]
    scl_mean = np.mean(scl)
    scl_slope = np.polyfit(np.arange(len(scl)), scl, 1)[0]

    scr_indices = np.where(eda["SCR_Peaks"])[0]
    scr_ratio = len(scr_indices) / len(eda_rec)
    scr_amp = eda["SCR_Amplitude"]
    scr_amp = scr_amp[scr_indices].mean()
    scr_rise = eda["SCR_RiseTime"]
    scr_rise = scr_rise[scr_indices].mean()

    return (
        scl_mean,
        scl_slope,
        scr_ratio,
        scr_amp,
        scr_rise,
    )


def get_rsp_features(rsp_rec, sr):
    """
    Get RSP-derived features
    """
    rsp, _ = nk.rsp_process(rsp_rec, sampling_rate=sr)
    rsp_rate = rsp["RSP_Rate"].mean()
    rsp_depth = rsp["RSP_Amplitude"].mean()
    rvt = rsp["RSP_RVT"].mean()

    rrv = nk.rsp_rrv(rsp["RSP_Rate"], troughs=rsp["RSP_Troughs"], sampling_rate=sr)
    return (
        rsp_rate,
        rsp_depth,
        rvt,
        rrv["RRV_RMSSD"].values[0],
        rrv["RRV_SDSD"].values[0],
    )


def get_skt_features(skt_rec):
    """
    Get SKT-derived features
    """
    skt_mean = np.mean(skt_rec)
    skt_slope = np.polyfit(np.arange(len(skt_rec)), skt_rec, 1)[0]
    return skt_mean, skt_slope


def feature_extraction(session, name, sr):
    """
    Extract features from a session
    """
    ppg_rec = session["ppg"].values
    ecg_rec = session["ecg"].values
    eda_rec = session["eda"].values
    rsp_rec = session["rsp"].values
    skt_rec = session["skt"].values

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

        if len(ppg_segment) == 30 * sr:
            ppg_feats = get_ppg_features(ppg_segment, sr)
            ecg_feats = get_ecg_features(ecg_segment, sr)
            eda_feats = get_eda_features(eda_segment, sr)
            rsp_feats = get_rsp_features(rsp_segment, sr)
            skt_feats = get_skt_features(skt_segment)

            hr, sdnn, sdsd, rmssd, hf = ecg_feats
            scl_mean, scl_slope, scr_peaks, scr_amp, scr_rise = eda_feats
            rsp_rate, rsp_depth, rvt, rrv_rmssd, rrv_sdsd = rsp_feats
            skt_mean, skt_slope = skt_feats
            ppg_hr = ppg_feats.mean()

            segment_features = {
                "ppg_hr": ppg_hr,
                "hr": hr,
                "sdnn": sdnn,
                "sdsd": sdsd,
                "rmssd": rmssd,
                "hf": hf,
                "scl_mean": scl_mean,
                "scl_slope": scl_slope,
                "scr_peaks": scr_peaks,
                "scr_amp": scr_amp,
                "scr_rise": scr_rise,
                "rsp_rate": rsp_rate,
                "rsp_depth": rsp_depth,
                "rvt": rvt,
                "rrv_rmssd": rrv_rmssd,
                "rrv_sdsd": rrv_sdsd,
                "skt_mean": skt_mean,
                "skt_slope": skt_slope,
            }

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

        if os.path.exists(f"{samples_path}/P{sub}_{mode}_event.npy"):
            continue

        session = load_session(int(sub), mode)
        # session["time"] = pd.to_datetime(session["time"], unit="s")
        session["rsp"] = session.filter(like="rsp").mean(axis=1)

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
