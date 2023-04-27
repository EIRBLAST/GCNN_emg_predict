import numpy as np
import torch, pickle, copy


class DataPreProcessing(object):
    def __init__(self, pickle_path):
        self.data_dict = self.load(pickle_path)

    def cross_validation_out(self):
        keys = list(self.data_dict.keys())
        for k in keys:
            train = {"data": [], "label": []}
            val = {"data": [], "label": []}
            train_n = {"data": [], "label": []}
            val_n = {"data": [], "label": []}

            if self.data_dict[0].get("R") is not None:
                train["ref"] = []
                val["ref"] = []

            if self.data_dict[0].get("Rn") is not None:
                train_n["ref"] = []
                val_n["ref"] = []

            for z in keys:
                if z != k:
                    train["data"].append(self.data_dict[z]["X"])
                    train["label"].append(self.data_dict[z]["T"])

                    train_n["data"].append(self.data_dict[z]["Xn"])
                    train_n["label"].append(self.data_dict[z]["Tn"])

                    if self.data_dict[z].get("Rn") is not None:
                        train_n["ref"].append(self.data_dict[z]["Rn"])
                    if self.data_dict[z].get("R") is not None:
                        train["ref"].append(self.data_dict[z]["R"])

                else:
                    val["data"] = self.data_dict[z]["X"]
                    val["label"] = self.data_dict[z]["T"]

                    val_n["data"] = self.data_dict[z]["Xn"]
                    val_n["label"] = self.data_dict[z]["Tn"]

                    if self.data_dict[z].get("Rn") is not None:
                        val_n["ref"] = self.data_dict[z]["Rn"]
                    if self.data_dict[z].get("R") is not None:
                        val["ref"] = self.data_dict[z]["R"]

            yield copy.deepcopy(train), copy.deepcopy(val), copy.deepcopy(train_n), copy.deepcopy(val_n)

    def cross_validation_in(self, values, labels, ref):
        N = len(values)
        for k in range(N):
            train_data = np.hstack([values[z] for z in range(N) if z != k])
            val_data = np.array(values[k])

            train_label = np.hstack([labels[z] for z in range(N) if z != k])
            val_label = np.array(labels[k])

            train_ref = np.hstack([ref[z] for z in range(N) if z != k])
            val_ref = np.array(ref[k])

            yield copy.deepcopy(train_data), copy.deepcopy(val_data), copy.deepcopy(train_label), copy.deepcopy(
                val_label
            ), copy.deepcopy(train_ref), copy.deepcopy(val_ref)

    def load(self, path):
        with open(path, "rb") as handle:
            b = pickle.load(handle)
            return b


class EMGData(torch.utils.data.Dataset):
    def __init__(self, data, label, ref=None, downsample=True, rate=4):
        self.data = data
        self.label = label
        self.ref = ref

        if downsample:
            self.data = self.downsample(self.data, rate)
            self.label = self.downsample(self.label, rate)
            if self.ref is not None:
                self.ref = self.downsample(self.ref, rate)

        self.data = copy.deepcopy(self.data.T)
        self.label = copy.deepcopy(self.label.T)
        if self.ref is not None:
            self.ref = copy.deepcopy(self.ref.T)

    def downsample(self, x, rate):
        return x[:, ::rate]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = torch.from_numpy(self.data[idx].T).float().unsqueeze(0)
        label = torch.from_numpy(self.label[idx].T).float().unsqueeze(0)
        if self.ref is not None:
            ref = torch.from_numpy(self.ref[idx].T).float().unsqueeze(0)
            return datum, label, ref
        else:
            return datum, label, None
