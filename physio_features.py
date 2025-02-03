import pickle
import numpy as np, pandas as pd
import os, neurokit2 as nk
from tqdm import tqdm

datapath = "/media/data/toyota/processed_data/trina_33"
samples_path = "/media/data/toyota/processed_data/trina_33_samples_fupd"
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


def get_ecg_samples(ecg_signal, length=30, sr=250):
    if len(ecg_signal) < length * sr:
        print("ECG too short")
        return None
    ecg_whole, _ = nk.ecg_process(ecg_signal, sampling_rate=sr)
    ecg_whole = ecg_whole["ECG_Rate"]
    ecg_segments = [
        ecg_whole[i - sr * length // 2 : i + sr * length // 2]
        for i in range(0, len(ecg_whole), sr)
    ]
    # keep only when segment is 30s
    ecg_segments = [seg for seg in ecg_segments if len(seg) == length * sr]
    return np.stack(ecg_segments, axis=0)


def get_eda_samples(eda_signal, length=30, sr=250):
    if len(eda_signal) < length * sr:
        print("EDA too short")
        return None
    eda_whole = nk.eda_clean(eda_signal, sampling_rate=sr)
    eda_whole = nk.eda_phasic(eda_whole, sampling_rate=sr)["EDA_Phasic"]
    eda_segments = [
        eda_whole[i - sr * length // 2 : i + sr * length // 2]
        for i in range(0, len(eda_whole), sr)
    ]
    # keep only when segment is 30s
    eda_segments = [seg for seg in eda_segments if len(seg) == length * sr]
    return np.stack(eda_segments, axis=0)


def get_rsp_samples(rsp_signal, length=30, sr=250):
    if len(rsp_signal) < length * sr:
        print("RSP too short")
        return None
    rsp_whole = nk.rsp_clean(rsp_signal, sampling_rate=sr)
    rsp_rate = nk.rsp_rate(rsp_whole, sampling_rate=sr)
    rsp_segments = [
        rsp_whole[i - sr * length // 2 : i + sr * length // 2]
        for i in range(0, len(rsp_rate), sr)
    ]
    # keep only when segment is 30s
    rsp_segments = [seg for seg in rsp_segments if len(seg) == length * sr]
    return np.stack(rsp_segments, axis=0)


def get_skt_samples(skt_signal, length=30, sr=250):
    if len(skt_signal) < length * sr:
        print("SKT too short")
        return None
    skt_segments = [
        skt_signal[i - sr * length // 2 : i + sr * length // 2]
        for i in range(0, len(skt_signal), sr)
    ]
    # keep only when segment is 30s
    skt_segments = [seg for seg in skt_segments if len(seg) == length * sr]
    return np.stack(skt_segments, axis=0)


def feature_extraction(session, name, length=30, sr=250):
    """
    Extract features from a session
    """
    features = {
        "ecg": get_ecg_samples(session["ecg"].values, length, sr),
        "eda": get_eda_samples(session["eda"].values, length, sr),
        "rsp": get_rsp_samples(session["rsp"].values, length, sr),
        "skt": get_skt_samples(session["skt"].values, length, sr),
    }
    if features["ecg"] is not None:
        print(
            features["ecg"].shape,
            features["eda"].shape,
            features["rsp"].shape,
            features["skt"].shape,
        )
        with open(f"{samples_path}/{name}.pkl", "wb") as f:
            pickle.dump(features, f)


if __name__ == "__main__":

    length = 30
    air = length // 2
    sr = 100

    for session_title in tqdm(os.listdir(datapath)):
        sub, mode = session_title[1:-4].split("_")
        print(f"Subject {sub}, mode {mode}")

        session = load_session(int(sub), mode)
        session["time"] = pd.to_datetime(session["time"], unit="s")
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

        video_baseline = video_baseline.resample("10ms").ffill().bfill()
        free_driving = free_driving.resample("10ms").ffill().bfill()
        event_driving = event_driving.resample("10ms").ffill().bfill()
        recovery_driving = recovery_driving.resample("10ms").ffill().bfill()

        feature_extraction(video_baseline, f"P{sub}_{mode}_video", length, sr)
        feature_extraction(free_driving, f"P{sub}_{mode}_free", length, sr)
        feature_extraction(event_driving, f"P{sub}_{mode}_event", length, sr)
        feature_extraction(recovery_driving, f"P{sub}_{mode}_recov", length, sr)
