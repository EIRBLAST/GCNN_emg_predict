import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gcnndataset import EMGDataGCnn
from gcnn_models import GCNLayerConfigurableMLP
from labeling import Labeler

# Configuration
base_config = {
    "batch_size": 1,
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

# Set device
device = torch.device(base_config["device"])

# Initialize model, loss, and optimizer
model = GCNLayerConfigurableMLP(config=base_config).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=base_config["lr"])

dataset = EMGDataGCnn(root='/home/sam/programation/self-supervised-regression-emg/nmf_emg/gcnndataset', edge_index=model.edge_index)
dataloader = DataLoader(dataset,batch_size=base_config['batch_size'] ,shuffle=True)


print("done loading now training model --->")
# Training loop     
num_epochs = 2
for epoch in range(num_epochs):
    print("Epoch---> ", epoch)
    model.train()   
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss---> {epoch_loss / len(dataloader)}')

# Save checkpoint at the last epoch
checkpoint_path = os.path.join("/home/sam/programation/self-supervised-regression-emg/checkpoints/", f'checkpoint_{model.short_name}.pth')
state = dict(base_config).copy()
state["model"] = model.state_dict()
torch.save(state, checkpoint_path)
print(f'Model checkpoint saved at {checkpoint_path}')

print("Training done!") 