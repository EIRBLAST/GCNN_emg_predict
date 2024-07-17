import torch
import copy
import numpy as np
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
    
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        #         
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
            
        if x.dim() == 1:
            x = x.view(-1, 1)
        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class GCNLayerConfigurableMLP(torch.nn.Module):
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
        self.name = "GCNLayerConfigurableMLP"
        self.short_name = "gcnn"
        self.edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],  # source nodes
            [1, 2, 3, 4, 5, 6, 7, 0]   # target nodes
        ], dtype=torch.long)


        if self.greedy_pretraining:
            self.build_form_config_greedy()
        else:
            self.build_form_config()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
        return torch.as_tensor(x)

    def init_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if name == "weight" and len(param.shape) > 1:
                    torch.nn.init.kaiming_normal_(layer.weight)  # https://arxiv.org/abs/1502.01852
                elif len(param.shape) == 1:
                    torch.nn.init.zeros_(layer.bias)

    def build_form_config_greedy(self):
        # No idea what is the diference with build_form_config 
        self.build_form_config()

    def build_form_config(self):
        self.layers = torch.nn.ModuleList([])
        # Add the first GCN layer
        self.layers.append(GCNConv(self.in_dim, self.mid_dim))
        # Add hidden GCN layers
        for _ in range(1, self.num_layers - 1):
            self.layers.append(GCNConv(self.mid_dim, self.mid_dim))
        # Add the output GCN layer
        self.layers.append(GCNConv(self.mid_dim, self.out_dim))

    def add_layer_greedy(self, config):
        """Add a new layer to the model, setting all others to non-trainable."""
        last_idx = -1
        # Make a copy of the current layers
        last_layer = copy.deepcopy(self.layers[last_idx:])
        layers_new = copy.deepcopy(self.layers[:last_idx])

        # Freeze parameters of existing layers
        for layer in layers_new:
            for param in layer.parameters():
                param.requires_grad = False

        # Add a new GCN layer
        layers_new.append(GCNConv(config.get("mid_dim"), config.get("mid_dim")))

        # Add the output layer
        layers_new.extend(last_layer)

        # Update the layers
        self.layers = torch.nn.ModuleList(layers_new)

    def unfreeze_layers(self):
        print("Unfreezing all layers")
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
