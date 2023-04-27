import torch
import copy


class LayerConfigurableMLP(torch.nn.Module):
    """
    Layer-wise configurable Multilayer Perceptron.
    """

    def __init__(self, config):
        super().__init__()

        # Retrieve model configuration
        self.in_dim = config.get("in_dim")
        self.out_dim = config.get("out_dim")
        self.mid_dim = config.get("mid_dim")
        self.num_layers = config.get("num_layers")
        self.greedy_pretraining = config.get("greedy_pretraining")

        if self.greedy_pretraining:
            self.build_form_config_greedy()
        else:
            self.build_form_config()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if name == "weight" and len(param.shape) > 1:
                    torch.nn.init.kaiming_normal_(layer.weight)  # https://arxiv.org/abs/1502.01852
                elif len(param.shape) == 1:
                    torch.nn.init.zeros_(layer.bias)

    def build_form_config_greedy(self):
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.in_dim, self.mid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.mid_dim, self.out_dim),
                torch.nn.Sigmoid(),
            ]
        )

    def build_form_config(self):
        self.layers = torch.nn.ModuleList([])
        self.layers.extend([torch.nn.Linear(self.in_dim, self.mid_dim), torch.nn.ReLU()])
        for _ in range(self.num_layers - 2):
            self.layers.extend([torch.nn.Linear(self.mid_dim, self.mid_dim), torch.nn.ReLU()])
        self.layers.extend([torch.nn.Linear(self.mid_dim, self.out_dim), torch.nn.Sigmoid()])

    def add_layer_greedy(self, config):
        """Add a new layer to a model, setting all others to nontrainable."""
        last_idx = -2
        last_layers = copy.deepcopy(self.layers[last_idx:])
        layers_new = copy.deepcopy(self.layers[:last_idx])

        # Iterate over all except last layer (linear + activation)
        for layer in layers_new:
            for param in layer.parameters():
                param.requires_grad = False

        # Append new layer to the final intermediate layer
        layers_new.append(torch.nn.Linear(config.get("mid_dim"), config.get("mid_dim")))
        layers_new.append(torch.nn.ReLU())

        # Append last layers
        layers_new.extend(last_layers)

        # update model
        self.layers = copy.deepcopy(layers_new)

    def unfreeze_layers(self):
        print("Unfreezing all layers")
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
