import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import pickle, copy


from torch_geometric.loader import DataLoader
from gcnn_models import GCNLayerConfigurableMLP
from labeling import Labeler

class EMGDataGCnn(InMemoryDataset):
    def __init__(self,root, edge_index, transform=None, pre_transform=None):
        self.edge_index = edge_index
        super().__init__(None, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        labeler = Labeler(data_path=DATA_PATH, alpha_type=ALPHA, diff_signal=True, n_rep_nmf=N_REP_NMF)
        data_dict = labeler.run_subj(plot=PLOT)
        data_dict.pop(0)
        data_list = []
        for key in data_dict:
            x = torch.tensor(data_dict[key]['Xn'], dtype=torch.float)
            y = torch.tensor(data_dict[key]['Rn'], dtype=torch.float)
            num_samples = x.shape[1]  # 29250 samples
            for i in range(num_samples):
                x_i = x[:, i]
                y_i = y[:, i]
                data = Data(x=x_i, edge_index=self.edge_index, y=y_i)
                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def len(self):
        return 29250

if __name__ == '__main__':

    # Configuration
    DATA_PATH = "/home/sam/programation/self-supervised-regression-emg/nmf_emg/data"
    ALPHA = 'ar10'
    N_REP_NMF = 10
    PLOT = False
    base_config = {
        "batch_size": 256,
        "in_dim": 8,
        "out_dim": 3,
        "mid_dim": 16,
        "num_layers": 8,
        "lr": 1e-3      ,
        "iters": 10000,
        "logging": True,
        "device": "cpu",
        "seed": False,
        "min_iter": 2000,
        "downsample_rate": 4,
        "dataset_name": "ar10",
        "greedy_pretraining": False,
    }
    # Initialize model, loss, and optimizer
    model = GCNLayerConfigurableMLP(config=base_config)

    # Load data
    dataset = EMGDataGCnn(root='/home/sam/programation/self-supervised-regression-emg/nmf_emg/gcnndataset', edge_index=model.edge_index)


    print("done loading now training model --->")