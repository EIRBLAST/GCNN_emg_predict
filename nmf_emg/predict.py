import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch, os

from nmf_emg.utils import print_model_structure
from nmf_emg.models import LayerConfigurableMLP
from nmf_emg.dataset import EMGData, DataPreProcessing

torch.multiprocessing.set_sharing_strategy("file_system")


def plot_3_out(data, label, ref, output, counter):
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))

    ax[0].plot(data)
    ax[1].plot(label[0, :].T, "--", c=cmap(2))
    ax[1].plot(output[0, :].T, c=cmap(0))
    ax[1].plot(ref[0, :].T, c=cmap(1))
    ax[2].plot(label[1, :].T, "--", c=cmap(2))
    ax[2].plot(output[1, :].T, c=cmap(0))
    ax[2].plot(ref[1, :].T, c=cmap(1))
    ax[3].plot(label[2, :].T, "--", c=cmap(2))
    ax[3].plot(output[2, :].T, c=cmap(0))
    ax[3].plot(ref[2, :].T, c=cmap(1))

    ax[0].set_ylim(-0.1, 1.1)
    ax[1].set_ylim(-0.1, 1.1)
    ax[2].set_ylim(-0.1, 1.1)
    ax[3].set_ylim(-0.1, 1.1)

    ax[0].set_ylabel("sEMG")
    ax[1].set_ylabel("power")
    ax[2].set_ylabel("pinch")
    ax[3].set_ylabel("ulnar")
    ax[3].set_xlabel("timesteps")

    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_xticklabels([])

    ax[1].legend(["label", "output", "ref"], loc="upper right")
    ax[2].legend(["label", "output", "ref"], loc="upper right")
    ax[3].legend(["label", "output", "ref"], loc="upper right")

    ax[0].set_title("CV_" + str(counter))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

    ckpt_path = os.path.join(script_path, "checkpoints", "ar10_BS256_D16_N4_normal_training_from_greedy.pth")
    state = torch.load(ckpt_path)

    # DATASET
    dataset_path = os.path.join("datasets", f"subj_{state['dataset_name']}.pickle")

    # Initialize the model
    model = LayerConfigurableMLP(state)
    model.build_form_config()
    model.load_state_dict(state["model"])
    model.eval()

    print_model_structure(model)

    dataset_cv_out = DataPreProcessing(pickle_path=dataset_path).cross_validation_out()
    for out_counter, (_, _, _, val) in enumerate(dataset_cv_out):
        dataset_val = EMGData(val["data"], val["label"], val["ref"])
        data, label, ref, output = [], [], [], []
        loader = torch.utils.data.DataLoader(dataset_val, batch_size=len(dataset_val), num_workers=16, shuffle=False)

        x, l, r = next(iter(loader))
        out = model(x.squeeze(1))

        data = np.array(x.detach().squeeze().numpy())
        label = np.array(l.detach().squeeze().numpy()).T
        ref = np.array(r.detach().squeeze().numpy()).T
        output = np.array(out.detach().squeeze().numpy()).T

        plot_3_out(data, label, ref, output, out_counter)
