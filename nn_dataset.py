import os, numpy as np, pickle
from torch.utils.data import Dataset
from tqdm import tqdm


class TRINA33(Dataset):
    def __init__(
        self,
        path: str,
        task: str | None = None,
        subjects: list[str] | None = ["P10", "P14"],
        modalities: list[str] | None = ["ecg", "eda", "rsp", "skt"],
        mean: dict[str, np.ndarray] | None = None,
        std: dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        path : str
            Path of the cached data.
        task : str
            Task to load data for, "I" (1), "S" (2), or "M" (3).
            If None, all tasks are loaded.
        subjects : list of str
            List of subjects to load data for.
            If None, all subjects are loaded.
        mean : np.ndarray
            Mean for normalization.
        std : np.ndarray
            Standard deviation for normalization.
        """
        self.path = path
        self.modalities = modalities
        self.tasks = ["I", "M", "S"] if task is None else task
        self.subjects = (
            [name.split("_") for name in os.listdir(self.path)]
            if not len(subjects)
            else subjects
        )
        self.subjects = list(set(self.subjects))
        self.task_dict = {"I": 1, "S": 2, "M": 3}

        # data for normalization
        self.mean, self.std = mean, std
        self.norm_data = {m: [] for m in self.modalities}

        # load data and labels
        self.data = {m: [] for m in self.modalities}
        self.labels, self.names = [], []

        for subject in tqdm(self.subjects):
            for task in self.tasks:
                video_path = os.path.join(self.path, f"{subject}_{task}_video.pkl")
                free_path = os.path.join(self.path, f"{subject}_{task}_free.pkl")
                event_path = os.path.join(self.path, f"{subject}_{task}_event.pkl")
                # recov_path = os.path.join(self.path, f"{subject}_{task}_recov.pkl")

                if not os.path.exists(video_path):
                    continue

                if self.mean is None:
                    self.make_normalization_data(video_path)

                self.load_from_path(video_path, subject, label=0)
                self.load_from_path(free_path, subject, label=1)
                self.load_from_path(event_path, subject, label=1)
                # self.load_from_path(recov_path, subject, label=0)

        for m in self.modalities:
            self.data[m] = np.vstack(self.data[m])

        # global (mean, std) for normalization
        if self.mean is None:
            self.mean = {m: np.mean(self.norm_data[m]) for m in self.modalities}
            self.std = {m: np.std(self.norm_data[m]) for m in self.modalities}

        # normalize data
        for m in self.modalities:
            self.data[m] = (self.data[m] - self.mean[m]) / self.std[m]

        print("Loaded {} samples.".format(len(self)))

    def load_from_path(self, path: str, sub: str, label: int) -> None:
        with open(path, "rb") as f:
            this_data = pickle.load(f)
            for m in self.modalities:
                if this_data[m] is None:
                    continue
                else:
                    this_data[m] = this_data[m][:, ::10]
                self.data[m].extend(this_data[m])

            self.labels.extend([label] * len(this_data[m]))
            self.names.extend([sub] * len(this_data[m]))

    def make_normalization_data(self, path: str) -> None:
        with open(path, "rb") as f:
            this_data = pickle.load(f)
            for m in self.modalities:
                if this_data[m] is None:
                    continue
                else:
                    this_data[m] = this_data[m][:, ::10]

                # include the middle 50% for normalization
                middle_idx = len(this_data[m]) // 4
                self.norm_data[m].extend(
                    this_data[m][middle_idx : len(this_data[m]) - middle_idx]
                )

    def __len__(self) -> int:
        return len(self.data["ecg"])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, str]:
        sample = {m: self.data[m][idx] for m in self.data.keys()}
        sample = np.stack([sample[m] for m in self.data.keys()], axis=0)
        return sample, self.labels[idx], self.names[idx]


if __name__ == "__main__":
    SRC = "/media/data/toyota/processed_data/trina_33_samples_final/"
    sample = TRINA33(SRC)[10]
    print(sample[0].shape, sample[1], sample[2])
