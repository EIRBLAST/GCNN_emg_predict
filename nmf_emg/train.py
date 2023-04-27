import numpy as np
from termcolor import cprint

from nmf_emg.dataset import EMGData, DataPreProcessing
from nmf_emg.models import LayerConfigurableMLP
from nmf_emg.utils import set_seeds, print_model_structure, Trainer


def get_checkpoint_name_from_config(config):
    mid_dim = config["mid_dim"]
    num_layers = config["num_layers"]
    bs = config["batch_size"]
    dataset_name = config["dataset_name"]
    return f"{dataset_name}_BS{bs}_D{mid_dim}_N{num_layers}"


def greedy_layerwise_training(dataset_train, dataset_val, config, num_layers_to_add):
    """Perform greedy layer-wise training."""

    # Initialize the model
    model = LayerConfigurableMLP(config)
    print_model_structure(model)

    config["iters"] = config["iters"] // 4

    # Iterate over the number of layers to add
    for _ in range(num_layers_to_add):
        # Train the model
        trainer = Trainer(model, dataset_train, dataset_val, config)
        model = trainer.train()

        # Add layer to model
        model.add_layer_greedy(config)
        print_model_structure(model)

    suffix = get_checkpoint_name_from_config(config) + "_greedy_pretraining"
    return trainer.save_best_model(path_suffix=suffix)


def normal_training(dataset_train, dataset_val, config, pretrained_model=None):
    """Perform normal training."""

    # Initialize the model
    model = LayerConfigurableMLP(config)
    model.build_form_config()

    print_model_structure(model)

    # Construct trainer
    trainer = Trainer(model, dataset_train, dataset_val, config)

    # Load pretrained model
    if pretrained_model is not None:
        ckpt_name = get_checkpoint_name_from_config(config) + "_greedy_pretraining.pth"
        trainer.load_pretrained_model(ckpt_name)

    # Train the model
    trainer.train()

    # Save the model

    suffix = get_checkpoint_name_from_config(config) + "_normal_training"
    if pretrained_model is not None:
        suffix = suffix + "_from_" + pretrained_model

    state = trainer.save_best_model(path_suffix=suffix)
    return state


def get_train_val_dataset(pickle_path, config):
    """Perform normal training."""

    dataset = DataPreProcessing(pickle_path=pickle_path)
    dataset_cv_out = dataset.cross_validation_out()
    _, _, train_n_dict, val_n_dict = next(dataset_cv_out)

    data_train = np.hstack(train_n_dict["data"])
    data_val = np.array(val_n_dict["data"])
    label_train = np.hstack(train_n_dict["label"])
    label_val = np.array(val_n_dict["label"])
    if train_n_dict.get("ref") is not None:
        ref_train = np.hstack(train_n_dict["ref"])
        ref_val = np.array(val_n_dict["ref"])
    else:
        ref_train = None
        ref_val = None

    dataset_train = EMGData(data_train, label_train, ref=ref_train, rate=config["downsample_rate"])
    dataset_val = EMGData(data_val, label_val, ref_val, rate=config["downsample_rate"])
    return dataset_train, dataset_val


if __name__ == "__main__":
    base_config = {
        "batch_size": 256,
        "in_dim": 8,
        "out_dim": 3,
        "mid_dim": 16,
        "num_layers": 4,
        "lr": 1e-3,
        "iters": 10000,
        "logging": True,
        "device": "cpu",
        "seed": False,
        "min_iter": 2000,
        "downsample_rate": 4,
        "dataset_name": "ar10",
        "greedy_pretraining": True,
    }

    print(base_config)

    if base_config["seed"] != False:
        set_seeds(base_config["seed"])

    dataset_train, dataset_val = get_train_val_dataset(
        f"datasets/subj_{base_config['dataset_name']}.pickle", base_config
    )

    pretrained_model = None
    if base_config["greedy_pretraining"]:
        cprint("**** greedy pre-training ****", "yellow")
        greedy_layerwise_training(
            dataset_train, dataset_val, base_config, num_layers_to_add=base_config["num_layers"] - 2
        )
        pretrained_model = "greedy"

    state = normal_training(dataset_train, dataset_val, base_config, pretrained_model=pretrained_model)
