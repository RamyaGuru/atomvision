'''
Load checkpoint files and apply to test set.

Steps:
1. Run the trained UNet model to perform the localization
2. Run the trained GCN model to perform classification

Unclear how to plot the graph at the end of the script
'''

#Generate the training/val/test split
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, SubsetRandomSampler
import alignn
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from atomvision.data.stem import Jarvis2dSTEMDataset, build_prepare_graph_batch, atom_mask_to_graph
from atomvision.plotting import plot_edges
from atomvision.models.segmentation_utils import (
    to_tensor_resnet18,
    prepare_atom_localization_batch,
)
import numpy as np
from jarvis.db.figshare import data
import matplotlib.pyplot as plt
from alignn.models.alignn import ALIGNN, ALIGNNConfig

checkpoint_dir = Path(".")

device = torch.device("cpu")

batch_size = 1

# Get the (distorted) 2D STEM Dataset used for testing
my_data = data("dft_2d")[0:6]
j2d = Jarvis2dSTEMDataset(  
    image_data =  my_data, 
	label_mode="radius"
)

test_loader = DataLoader(
    j2d, batch_size=batch_size, shuffle=True, num_workers=0
)

# pre-trained UNet model
preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")
unet = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    encoder_depth=3,
    decoder_channels=(64, 32, 16),
    in_channels=3,
    classes=1,
)
state = torch.load(checkpoint_dir / "checkpoint_100.pt", map_location=torch.device('cpu'))
unet.load_state_dict(state["model"])

# Generate Plots of the Graph Representations

graphs = []
for indx in range(len(j2d)):
    #image = np.reshape(j2d[indx]["image"], (256,256,3))
    print(np.shape(j2d[indx]["label"]))
    print(np.shape(j2d[indx]["image"]))
    g, props = atom_mask_to_graph(j2d[indx]["label"], j2d[indx]["image"])
    plt.figure()
    plt.imshow(j2d[indx]["image"])
    plot_edges(g)
    plt.savefig("stem_edges_{}.pdf".format(indx), bbox_inches = "tight")



#pre-trained GCN model
cfg = ALIGNNConfig(
    name="alignn",
    alignn_layers=0,
    #atom_input_features=2,
    output_features=j2d.n_classes,
)
gcn_model = ALIGNN(cfg)
state = torch.load(checkpoint_dir / "gcn_checkpoint_99.pt", map_location=torch.device('cpu'))
gcn_model.load_state_dict(state["model"])
gcn_model.to(device)
gcn_model.eval()



