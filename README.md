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

You first need to download the *S2-100K* dataset. This can be done directly via [Hugging Face](https://huggingface.co/datasets/torchgeo/s2-100k), using the `huggingface_hub` library:
```python
from huggingface_hub import snapshot_download
snapshot_download("torchgeo/s2-100k", local_dir='.', repo_type='dataset')
```
Alternatively you can clone the repository:
```bash
git clone https://huggingface.co/datasets/torchgeo/s2-100k
```

The images are distributed as 100 sharded tar files under `data/`. Extract them into an `images/` folder so that the dataset directory contains `index.csv` and `images/patch_<N>.tif`, as expected by the training data loader:
```bash
mkdir -p images
for f in data/shard-*.tar; do tar -xf "$f" -C images; done
```

Now, to train **SatCLIP** models, set the paths correctly (point `data.data_dir` in `satclip/configs/default.yaml` to this dataset directory), adapt training configs in `satclip/configs/default.yaml` and train SatCLIP by running:
```bash
cd satclip
python main.py
```

### Use of the S2-100K dataset

The S2-100K dataset is a dataset of 100,000 multi-spectral satellite images (256×256 px, 12 bands, sampled at 10 m resolution) sampled from Sentinel-2 via the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/). Copernicus Sentinel data is captured between Jan 1, 2021 and May 17, 2023. The dataset is sampled approximately uniformly over landmass and only includes images without cloud coverage. If you use the dataset, please cite our paper. More information on the dataset can be found on the [Hugging Face dataset page](https://huggingface.co/datasets/torchgeo/s2-100k) and in our [paper](https://arxiv.org/abs/2311.17179).

### Downstream evaluation datasets

We evaluate SatCLIP location embeddings on a range of downstream tasks. The table below lists the datasets used in our [paper](https://arxiv.org/abs/2311.17179v3), the task they are used for, and where to obtain them.

| Task | Type | Source |
| --- | --- | --- |
| Air temperature / precipitation | Regression | [Hooker et al. (2018)](https://www.nature.com/articles/sdata2018246) (see notebook [B01](notebooks/B01_Example_Air_Temperature_Prediction.ipynb)) |
| Median income | Regression | [Jia & Benson (2020)](https://github.com/000Justin000/gnn-residual-correlation) |
| California housing | Regression | [Pace & Barry (1997)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |
| Elevation | Regression | [SatCLIP downstream data](https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad?usp=sharing) (released by the authors) |
| Population density | Regression | [SatCLIP downstream data](https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad?usp=sharing) (released by the authors) |
| Country classification | Classification | [SatCLIP downstream data](https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad?usp=sharing) (released by the authors) |
| Biome classification | Classification | [SatCLIP downstream data](https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad?usp=sharing) (released by the authors) |
| Ecoregion classification | Classification | [SatCLIP downstream data](https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad?usp=sharing) (released by the authors) |
| Species classification (image localization) | Classification | [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018) ([Van Horn et al., 2018](https://arxiv.org/abs/1707.06642)) |

The five datasets we created for this paper — **Elevation**, **Population density**, **Country**, **Biome** and **Ecoregion** — are available in a single [Google Drive folder](https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad?usp=sharing) (see [issue #6](https://github.com/microsoft/satclip/issues/6)). The biome and ecoregion labels are derived from [Dinerstein et al. (2017)](https://doi.org/10.1093/biosci/bix014).

## Pretrained Models

![CLIP](/figures/globes.gif)

*Visualization of embeddings obtained by different location encoders for locations around the globe.*

We provide six pretrained SatCLIP models, trained with different vision encoders and spatial resolution hyperparameters $L$ (these indicate the number of Legendre polynomials used for spherical harmonics location encoding. Please refer to our paper for more details). The pretrained models are hosted on [Hugging Face](https://huggingface.co/models?other=arxiv:2311.17179):

| Model | Vision encoder | Resolution $L$ | Checkpoint file |
| --- | --- | --- | --- |
| [microsoft/SatCLIP-ResNet18-L10](https://huggingface.co/microsoft/SatCLIP-ResNet18-L10) | ResNet18 | 10 | `satclip-resnet18-l10.ckpt` |
| [microsoft/SatCLIP-ResNet18-L40](https://huggingface.co/microsoft/SatCLIP-ResNet18-L40) | ResNet18 | 40 | `satclip-resnet18-l40.ckpt` |
| [microsoft/SatCLIP-ResNet50-L10](https://huggingface.co/microsoft/SatCLIP-ResNet50-L10) | ResNet50 | 10 | `satclip-resnet50-l10.ckpt` |
| [microsoft/SatCLIP-ResNet50-L40](https://huggingface.co/microsoft/SatCLIP-ResNet50-L40) | ResNet50 | 40 | `satclip-resnet50-l40.ckpt` |
| [microsoft/SatCLIP-ViT16-L10](https://huggingface.co/microsoft/SatCLIP-ViT16-L10) | ViT-B/16 | 10 | `satclip-vit16-l10.ckpt` |
| [microsoft/SatCLIP-ViT16-L40](https://huggingface.co/microsoft/SatCLIP-ViT16-L40) | ViT-B/16 | 40 | `satclip-vit16-l40.ckpt` |

Usage of pretrained models is simple. Simply specify the SatCLIP model you want to access, e.g. `satclip-vit16-l40`:
```python
from huggingface_hub import hf_hub_download
from load import get_satclip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

c = torch.randn(32, 2)  # Represents a batch of 32 locations (lon/lat)

model = get_satclip(
    hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"),
    device=device,
)  # Only loads location encoder by default
model.eval()
with torch.no_grad():
    emb = model(c.double().to(device)).detach().cpu()
```

## Examples

Examples on how to obtain and use pretrained SatCLIP embeddings can be found in the `notebooks` folder. We provide notebooks (optimized for use with Google Colab) for the following use cases.

*Setup:*
* [A01 - Simple usage example](notebooks/A01_Simple_SatCLIP_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/A01_Simple_SatCLIP_Usage.ipynb)
* [A02 - Load models from Hugging Face](notebooks/A02_SatCLIP_Hugging_Face_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/A02_SatCLIP_Hugging_Face_Usage.ipynb)

*Example use cases:*
* [B01 - Air temperature prediction with SatCLIP](notebooks/B01_Example_Air_Temperature_Prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/B01_Example_Air_Temperature_Prediction.ipynb)
* [B02 - Example Image Localization](notebooks/B02_Example_Image_Localization.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/B02_Example_Image_Localization.ipynb)

*Use baseline pretrained location encoders:*
* [C01 - Simple CSP usage](notebooks/C01_Simple_CSP_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/C01_Simple_CSP_Usage.ipynb)
* [C02 - Simple GeoCLIP usage](notebooks/C02_Simple_GeoCLIP_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/C02_Simple_GeoCLIP_Usage.ipynb)
* [C03 - Simple GPS2Vec usage](notebooks/C03_Simple_GPS2Vec_Usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/satclip/blob/main/notebooks/C03_Simple_GPS2Vec_Usage.ipynb)

## Citation

```bibtex
@article{klemmer2025satclip,
    title={SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery},
    volume={39},
    url={https://ojs.aaai.org/index.php/AAAI/article/view/32457}, DOI={10.1609/aaai.v39i4.32457},
    number={4},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Klemmer, Konstantin and Rolf, Esther and Robinson, Caleb and Mackey, Lester and Rußwurm, Marc},
    year={2025},
    month={Apr.},
    pages={4347-4355}
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
