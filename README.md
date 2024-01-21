# 🛰️ SatCLIP - A Global, General-Purpose Geographic Location Encoder

![CLIP](/figures/satclip.png)

*Overview of the pretraining and deployment pipeline for SatCLIP.*

## Approach

SatCLIP trains location and image encoders via contrastive learning, by matching images to their corresponding locations. This is analogous to the CLIP approach, which matches images to their corresponding text. Through this process, the location encoder learns characteristics of a location, as represented by satellite imagery. For more details, check out our [paper](https://arxiv.org/abs/2311.17179).

## Overview

Usage of SatCLIP is simple:

```python
from model import *
from location_encoder import *

model = SatCLIP(
    embed_dim=512,
    image_resolution=224, in_channels=13, vision_layers=4, vision_width=768, vision_patch_size=32, # Image encoder
    le_type='sphericalharmonics', pe_type='siren', legendre_polys=10, frequency_num=16, max_radius=360, min_radius=1, harmonics_calculation='analytic'  # Location encoder
)

img_batch = torch.randn(32, 13, 224, 224) # Represents a batch of 32 images
loc_batch = torch.randn(32, 2) # Represents the corresponding 32 locations (lon/lat)

with torch.no_grad():
    logits_per_image, logits_per_coord = model(img_batch, loc_batch)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
```

## Training

You first need to download the *S2-100k* dataset in `/data/s2`. First, download the index file:
```bash
cd data/s2
wget https://satclip.z13.web.core.windows.net/satclip/index.csv
```
Within `/data/s2`, navigate to `/images`, download all images and unpack them:
```bash
cd images
wget https://satclip.z13.web.core.windows.net/satclip/satclip.tar
tar -xf satclip.tar
```

Now, to train **SatCLIP** models, set the paths correctly, adapt training configs in `clip/configs/default.yaml` and train SatCLIP by running:
```bash
python clip/main.py
```

### Use of the S2-100K dataset

The S2-100K dataset is a dataset of 100,000 multi-spectral satellite images sampled from Sentinel-2 via the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/). Copernicus Sentinel data is captured between Jan 1, 2021 and May 17, 2023. The dataset is sampled approximately uniformly over landmass and only includes images without cloud coverage. The dataset is available for research purposes only. If you use the dataset, please cite our paper. More information on the dataset can be found in our [paper](https://arxiv.org/abs/2311.17179).

## Pretrained Models

![CLIP](/figures/globes.gif)

*Visualization of embeddings obtained by different location encoders for locations around the globe.*

We provide six pretrained SatCLIP models, trained with different vision encoders and spatial resolution hyperparameters $L$ (these indicate the number of Legendre polynomials used for spherical harmonics location encoding. Please refer to our paper for more details). The pretrained models can be downloaded as follows:
* SatCLIP-ResNet18-L10: `wget https://satclip.z13.web.core.windows.net/satclip/satclip-resnet18-l10.ckpt` 
* SatCLIP-ResNet18-L40: `wget https://satclip.z13.web.core.windows.net/satclip/satclip-resnet18-l40.ckpt` 
* SatCLIP-ResNet50-L10: `wget https://satclip.z13.web.core.windows.net/satclip/satclip-resnet50-l10.ckpt` 
* SatCLIP-ResNet50-L40: `wget https://satclip.z13.web.core.windows.net/satclip/satclip-resnet50-l40.ckpt` 
* SatCLIP-ViT16-L10: `wget https://satclip.z13.web.core.windows.net/satclip/satclip-vit16-l10.ckpt` 
* SatCLIP-ViT16-L40: `wget https://satclip.z13.web.core.windows.net/satclip/satclip-vit16-l40.ckpt` 

Usage of pretrained models is simple:
```python
from load import get_satclip

device = 'cuda'

c = torch.randn(32, 2) # Represents a batch of 32 locations (lon/lat)

model = get_satclip('path_to_satclip', device=device) #Only loads location encoder by default
model.eval()
with torch.no_grad():
  emb  = model(c.double().to(device)).detach().cpu()
```

You can also access SatCLIP model weights directly via [Hugging Face](https://huggingface.co/microsoft?search_models=satclip).

## Examples

Examples on how to obtain and use pretrained SatCLIP embeddings can be found in the `notebooks` folder. We provide notebooks (optimized for use with Google Colab) for the following use cases.

*Setup:*
* [A01 - Simple usage example](notebooks/A01_Simple_SatCLIP_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/A01_Simple_SatCLIP_Usage.ipynb)
* [A02 - Load models from Hugging Face](notebooks/A02_SatCLIP_Hugging_Face_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/A02_SatCLIP_Hugging_Face_Usage.ipynb)

*Example use cases:*
* [B01 - Air temperature prediction with SatCLIP](notebooks/B01_Example_Air_Temperature_Prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/B01_Example_Air_Temperature_Prediction.ipynb)

*Use baseline pretrained location encoders:*
* [C01 - Simple CSP usage](notebooks/C01_Simple_CSP_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/C01_Simple_CSP_Usage.ipynb)
* [C02 - Simple GeoCLIP usage](notebooks/C02_Simple_GeoCLIP_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/C02_Simple_GeoCLIP_Usage.ipynb)
* [C03 - Simple GPS2Vec usage](notebooks/C03_Simple_GPS2Vec_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/C03_Simple_GPS2Vec_Usage.ipynb)

## Citation

```bibtex
@article{klemmer2023satclip,
  title={SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery},
  author={Klemmer, Konstantin and Rolf, Esther and Robinson, Caleb and Mackey, Lester and Ru{\ss}wurm, Marc},
  journal={arXiv preprint arXiv:2311.17179},
  year={2023}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
