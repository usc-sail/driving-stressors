import os, datetime as dt
import pandas as pd, numpy as np


RAW_PATH = "/media/data/toyota/raw_data/trina_33/"
PROC_PATH = "/media/data/toyota/processed_data/trina_33_final/"
os.makedirs(PROC_PATH, exist_ok=True)


def extract_data(p_name, sample_dt):
    return_dict = {}
    for exp in ["I", "S", "M"]:

        # check if physio file exists
        physio_file = os.path.join(RAW_PATH, p_name, exp, f"{p_name}{exp}.txt")
        if not os.path.exists(physio_file):
            physio_file = os.path.join(RAW_PATH, p_name, f"{p_name}{exp}.txt")
            if not os.path.exists(physio_file):
                print(f"No physio file found for {p_name} in {exp}.")
                continue

        # read physio data
        df = pd.read_csv(physio_file, sep="\t", header=19)
        time = np.linspace(0, len(df.values[:, 0]) * sample_dt, len(df.values[:, 0]))
        signals = {
            "ppg": df.values[:, 1],
            "ecg": df.values[:, 2],
            "eda": df.values[:, 3],
            "skt": df.values[:, 5],
            "rsp-ch": df.values[:, 4],  # chest
            "rsp-ab": df.values[:, 6],  # abdomen
        }
        return_dict[exp] = (time, signals)

    return return_dict


def get_event_time(events, name, physio_start, sample_dt):
    event = events[events["Event"] == name]
    event = (
        int(event["Hour"].values[0]),
        int(event["Minute"].values[0]),
        int(event["Second"].values[0]),
    )
    event = dt.time(*event)
    event_dt = dt.datetime.combine(dt.date(1, 1, 1), event)
    start_dt = dt.datetime.combine(dt.date(1, 1, 1), physio_start)
    event_sample = (event_dt - start_dt).total_seconds()
    event_sample = int(event_sample / sample_dt)
    return event_sample if event_sample > 0 else 0


def extract_events(p_name, sample_dt):
    return_dict = {}
    for exp in ["I", "S", "M"]:
        excel_path = os.path.join(RAW_PATH, "events", f"{exp}.xlsx")
        try:
            events = pd.read_excel(excel_path, sheet_name=p_name)
        except:
            try:
                events = pd.read_excel(excel_path, sheet_name=f"{p_name}I")
            except:
                print(f"No events found for {p_name} in {exp}.")
                continue

        # remove the Unnamed columns
        events = events.loc[:, ~events.columns.str.contains("^Unnamed")]
        # remove rows with NaN values
        events = events.dropna(how="any")

        physio_start = events[events["Event"] == "Data Start"]
        physio_start = (
            int(physio_start["Hour"].values[0]),
            int(physio_start["Minute"].values[0]),
            int(physio_start["Second"].values[0]),
        )
        physio_start = dt.time(*physio_start)

        video_start = get_event_time(events, "Video Start", physio_start, sample_dt)
        video_end = get_event_time(events, "Video End", physio_start, sample_dt)
        exp_start = get_event_time(events, "Experiment Start", physio_start, sample_dt)
        exp_end = get_event_time(events, "Experiment End", physio_start, sample_dt)
        try:
            col_start = get_event_time(
                events, "Collection Start", physio_start, sample_dt
            )
        except:
            col_start = exp_start - 1

        return_dict[exp] = {
            "Video Start": video_start,
            "Video End": video_end,
            "Collection Start": col_start,
            "Experiment Start": exp_start,
            "Experiment End": exp_end,
        }

        # get all event names between experiment start and end
        flag = False
        for _, row in events.iterrows():
            if flag:
                if row["Event"] == "Experiment End":
                    flag = False
                else:
                    event_name = row["Event"]
                    event_time = get_event_time(
                        events, event_name, physio_start, sample_dt
                    )
                    return_dict[exp][event_name] = event_time
            elif row["Event"] == "Experiment Start":
                flag = True

    return return_dict


if __name__ == "__main__":
    sample_dt = 0.0005  # data recorded every 0.5 ms (2000 Hz)
    p_names = [f"P{i}" for i in range(1, 34)]

    for p_name in p_names:
        print(f"\nProcessing participant {p_name}...")
        data = extract_data(p_name, sample_dt)
        timestamps = extract_events(p_name, sample_dt)

        # save csv data for each experiment
        for exp in data.keys():
            if exp not in timestamps:
                continue

            time, signals = data[exp]
            df = pd.DataFrame(
                {
                    "time": time,
                    "ppg": signals["ppg"],
                    "ecg": signals["ecg"],
                    "eda": signals["eda"],
                    "skt": signals["skt"],
                    "rsp-ch": signals["rsp-ch"],
                    "rsp-ab": signals["rsp-ab"],
                    "event": [""] * len(time),
                }
            )
            # add event timestamps
            for event, timestamp in timestamps[exp].items():
                df.loc[timestamp, "event"] = event
            # save to csv
            df.to_csv(os.path.join(PROC_PATH, f"{p_name}_{exp}.csv"), index=False)
            print(f"Saved {p_name}_{exp}.csv")

    print("\nAll participants processed.")
