# Model Card for SatCLIP

Here we provide a accompanying information about our model SatCLIP.

## Model Details

### Model Description

SatCLIP is a model for contrastive pretraining of satellite image-location pairs. Training is analogous to the popular [CLIP](https://github.com/openai/CLIP) model. 

- **Developed by:** Konstantin Klemmer, Marc Russwurm, Esther Rolf, Caleb Robinson, Lester Mackey
- **Model type:** Location and image encoder model pretrained using contrastive image-location matching.
- **License:** MIT

### Model Sources 

- **Repository:** [github.com/microsoft/satclip](https://github.com/microsoft/satclip)
- **Paper:** TBA

## Uses

SatCLIP includes an *image* and a *location* encoder. The image encoder processes multi-spectral satellite images of size `[height, width, 13]` into `[d]`-dimensional latent vectors. The location encoder processes location coordinates `[longitude, latitude]` into the same `[d]`-dimensional space. 

SatCLIP is a model trained and tested for use in research projects. It is not intended for use in production environments.

### Downstream Use 

The SatCLIP location encoder learns location characteristics, as captured by the satellite images, and can be deployed for downstream geospatial prediction tasks. Practically, this involves *querying* the location encoder for the `[d]`-dimensional vector embedding of all downstream locations and then using that embedding as predictor during downstream learning. In our paper, we show the useability of the learned location embeddings for predicting e.g. population density or biomes.

### Out-of-Scope Use

Potential use cases of SatCLIP which we did build the model for and did not test for include:
* The SatCLIP image encoder can in theory be used for helping with satellite image localization. If this application interests you, we encourage you to check work focusing on this, e.g. [Cepeda et al. (2023)](https://arxiv.org/abs/2309.16020). 
* Fine-grained geographic problems (i.e. problems constrained to small geographic areas or including many close locations) are out of scope for SatCLIP. SatCLIP location encoders are pretrained for global-scale use.
* Any use outside of research projects is currently out of scope as we don't evaluate SatCLIP in production environments.

## Bias, Risks, and Limitations

The following aspects should be considered before using SatCLIP:
* SatCLIP is trained with freely available Sentinel-2 satellite imagery with a resolution of 10m per pixel. This allows the model to learn larger structures like cities or mountain ranges, but not small scale structures like individual vehicles or people. SatCLIP models are not applicable for fine-grained geospatial problems.
* Location embeddings from SatCLIP only capture location characteristics that represent visually in satellite imagery (at our given resolution). Applications in problems that can not be captured through satellite images are out-of-score for SatCLIP.
* Use cases in the defense or surveillance domain are always out-of-scope regardless of performance of SatCLIP. The use of artificial intelligence for such tasks is premature currently given the lack of testing norms and checks to ensure its fair use.

## How to Get Started with the Model

Information about how to get started with SatCLIP training and deployment in downstream modelling can be found in our GitHub repository at [github.com/microsoft/satclip](https://github.com/microsoft/satclip).

## Training Details

### Training Data

SatCLIP is trained using the *S2-100K* dataset which samples 100,000 multi-spectral satellite image scenes from Sentinel-2 via the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/). Scenes are sampled approximately uniformly over landmass and are only chosen for the dataset if they don't exhibit cloud coverage. More details can be found in our paper.

### Training Procedure 

SatCLIP is trained via contrastive learning, by matching the correct image-location pairs in a batch of images and locations. Each image and each location is processed within an encoder and trasformed into a `[d]`-dimensional embedding. The training objective is to minimize the cosine similarity of image and location embeddings.

#### Training Hyperparameters

The key hyperparameters of SatCLIP are: batch size, learning rate and weight decay. On top of this, the specific location and vision encoder come with their separate hyperparameters. Key hyperparameters for the location encoder include resolution-specific hyperparameters in the positional encoding (e.g. number of Legendre polynomials used for spherical harmonics calculation) and the type, number of layers and capacity of the neural network deployed. For the vision encoder, key hyperparameters depend on the type of vision backbone deployed (e.g. ResNet, Vision Transformer). More details can be found in our paper.

#### Training Speed

Training SatCLIP for 500 epochs using pretrained vision encoders takes aoughly 2 days on a single A100 GPU.

## Evaluation

SatCLIP can be evaluated throughout training and during downstream deployment. During training, we log model loss on a held-out, unseen validation set to monitor the training process for potential overfitting. When SatCLIP embeddings are used in downstream applications, any predictive score can be used for evaluation, e.g. mean squared error (MSE) for regression or accuracy for classification problems.

## Citation

**BibTeX:**
```bibtex
@article{klemmer2023satclip,
  title={SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery},
  author={Klemmer, Konstantin and Rolf, Esther and Robinson, Caleb and Mackey, Lester and Russwurm, Marc},
  journal={TBA},
  year={2023}
}
```

## Model Card Contact

For feedback and comments, contact [kklemmer@microsoft.com](mailto:kklemmer@microsoft.com).