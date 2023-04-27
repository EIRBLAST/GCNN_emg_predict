import torch, random, copy
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from termcolor import cprint

from mapping import SynMatrix


def print_model_structure(model):
    print("Model structure:")
    for k, v in dict(model.named_parameters()).items():
        print(k, v.shape, v.requires_grad)
    print("*" * 10)


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(self, model, dataset_train, dataset_val, config, checkpoint_dir=None):
        self.config = config
        self.device = config["device"]
        self.logging = config["logging"]
        self.iters = config["iters"]
        self.epochs = self.iters // (len(dataset_train) // config["batch_size"])
        self.min_iter = config["min_iter"]

        hand_name = config["dataset_name"].split("_")[-1]
        if hand_name not in ["ar10", "ub", "berrett"]:
            hand_name = "ar10"  # default
        cprint(f"hand name: {hand_name}", "yellow")
        self.S_matrix = torch.tensor(getattr(SynMatrix, hand_name), dtype=torch.float32).to(self.device)

        self.model = model

        # checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = "checkpoints"
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        #########################
        # DATASET
        self.loader_train = DataLoader(dataset_train, batch_size=config["batch_size"], num_workers=0, shuffle=False)
        self.loader_val = DataLoader(dataset_val, batch_size=config["batch_size"], num_workers=0, shuffle=False)

        #########################
        # MODEL
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=5, min_lr=1e-8)
        self.model.to(config["device"])
        self.model.train()

    def criterion_syn(self, out, label):
        out_s = torch.matmul(out, self.S_matrix.T)
        label_s = torch.matmul(label, self.S_matrix.T)
        loss = self.criterion(out_s, label_s)
        return loss

    def validation(self):
        self.model.eval()
        tot = 0
        for data, label, _ in self.loader_val:
            with torch.no_grad():
                data = data.to(self.device)
                label = label.to(self.device)
                out = self.model(data.squeeze(1))
                loss = self.criterion_syn(out.squeeze(), label.squeeze())

                tot += loss.item()

        self.model.train()
        return tot / len(self.loader_val)

    def train(self):
        self.best_model = None
        best_model_step = 0
        min_val_loss = 1000

        if self.logging:
            train_log_dict = {"step": [], "train_loss": []}
            val_log_dict = {"step": [], "val_loss": []}

        global_step = 0
        early_stopper = EarlyStopper(patience=10)
        with tqdm(total=self.epochs) as pbar:
            for _ in range(self.epochs):
                epoch_loss = 0
                for data, label, _ in self.loader_train:
                    self.optimizer.zero_grad()

                    data = data.to(self.device)
                    label = label.to(self.device)

                    out = self.model(data.squeeze(1))
                    loss = self.criterion_syn(out.squeeze(), label.squeeze())

                    loss.backward()
                    self.optimizer.step()

                    if self.logging:
                        train_log_dict["step"].append(global_step)
                        train_log_dict["train_loss"].append(loss.item())

                    epoch_loss += loss.item()
                    global_step += 1

                epoch_loss /= len(self.loader_train)

                # validation
                val_loss = self.validation()
                self.scheduler.step(val_loss)

                pbar.set_postfix(**{"train_loss": epoch_loss, "val_loss": val_loss})
                pbar.update()

                if self.logging:
                    val_log_dict["step"].append(global_step)
                    val_log_dict["val_loss"].append(val_loss)

                if val_loss < min_val_loss:
                    self.best_model = copy.deepcopy(self.model)
                    min_val_loss = val_loss
                    best_model_step = global_step

                if global_step > self.min_iter:
                    if early_stopper.early_stop(val_loss):
                        print("*** Early Stopping! ***")
                        break

        if self.logging:
            self.config["logs"] = {"train": train_log_dict, "val": val_log_dict, "best_model_step": best_model_step}
            self.config["best_val_loss"] = min_val_loss

        return self.best_model

    def save_best_model(self, path_suffix=None):
        state = dict(self.config).copy()
        state["model"] = self.best_model.state_dict()

        if path_suffix is None:
            path_output = os.path.join(self.checkpoint_dir, self.config["dataset_name"] + ".pth")
        else:
            path_output = os.path.join(self.checkpoint_dir, path_suffix + ".pth")

        torch.save(state, path_output)
        print("Model saved to {}".format(path_output))
        return state

    def load_pretrained_model(self, path, absolute=False):
        print("Loading pretrained model: ", path)
        if not absolute:
            state = torch.load(os.path.join(self.checkpoint_dir, path))
        else:
            state = torch.load(path)
        self.model.load_state_dict(state["model"])

    def load_pretrained_model_from_rbm(self, path):
        print("Loading pretrained model: ", path)
        state = torch.load(os.path.join(self.checkpoint_dir, path))

        for key, val in state["model"].items():
            layer_idx = int(key.split(".")[1])
            layer_type = key.split(".")[-1]
            if layer_type == "W":
                self.model.layers[layer_idx * 2].weight = torch.nn.Parameter(val)
                print("layer ", layer_idx, " weight loaded")
