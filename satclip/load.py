# flake8: noqa
import warnings

import numpy as np
import torch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from mlops.trainer import BasePythonModel, MLFlowLightningModule

from satclip.main import SatCLIPLightningModule


def get_satclip(ckpt_path: str | None = None, return_all=False):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["hyper_parameters"].pop("eval_downstream")
        ckpt["hyper_parameters"].pop("air_temp_data_path")
        ckpt["hyper_parameters"].pop("election_data_path")
        lightning_model = SatCLIPLightningModule(**ckpt["hyper_parameters"])

        lightning_model.load_state_dict(ckpt["state_dict"])
        lightning_model.eval()

    else:
        lightning_model = SatCLIPLightningModule()

    geo_model = lightning_model.model

    if return_all:
        return geo_model
    else:
        return geo_model.location


def get_mlflow_satclip(ckpt_path: str | None = None):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["hyper_parameters"].pop("eval_downstream")
        ckpt["hyper_parameters"].pop("air_temp_data_path")
        ckpt["hyper_parameters"].pop("election_data_path")
        lightning_model = SatClipModel(**ckpt["hyper_parameters"])

        lightning_model.backbone.load_state_dict(ckpt["state_dict"])
        lightning_model.eval()

    else:
        lightning_model = SatClipModel()

    return lightning_model


class SatClipWrapper(BasePythonModel):

    # def load_context(self, context):
    #     """
    #     Load the model.
    #     """
    #     # Initialize the model with the stored parameters
    #     # self.model = self.model_class(**self.init_params)
    #     self.model = self.model(ckpt_path=context.artifacts["checkpoint"])

    #     self.model.eval()

    def predict(self, context, model_input) -> np.ndarray:

        # Input ============================================================================
        # Extract patch, height, and width from the input dictionary
        patch = torch.tensor(model_input["patch"])  # Assuming this is a torch.Tensor

        # Make Prediction ==================================================================
        # Make predictions with the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            output = self.model.predict(patch)

        return output.numpy()

    @staticmethod
    def get_signature(**kwargs):

        # Define input schema (considering patch as an object and height, width as integers)
        input_schema = Schema(
            [
                TensorSpec(
                    np.dtype(np.double), (-1, 2), name="patch"
                ),  # Dynamic H and W
            ]
        )

        # Define output schema (assuming the output is a tensor or numerical value)
        output_schema = Schema(
            [TensorSpec(np.dtype(np.double), (-1, 2), name="output")]
        )  # You can adjust this based on your model output

        # Create the signature
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        return signature


class SatClipModel(MLFlowLightningModule):
    def __init__(
        self,
        embed_dim=512,
        image_resolution=256,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        in_channels=4,
        le_type="grid",
        pe_type="siren",
        frequency_num=16,
        max_radius=260,
        min_radius=1,
        legendre_polys=16,
        harmonics_calculation="analytic",
        sh_embedding_dims=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_hidden_layers=2,
        capacity=256,
    ):

        super().__init__(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            le_type=le_type,
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            sh_embedding_dims=sh_embedding_dims,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
        )

        # Utility ==========================================================================
        self.name = "SatClip"
        self.wrapper = SatClipWrapper
        self.signature = self.wrapper.get_signature()

        # Build Model ======================================================================

        self.backbone = SatCLIPLightningModule(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            le_type=le_type,
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            sh_embedding_dims=sh_embedding_dims,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        x = self.backbone.model.location(x.double()).detach().cpu()

        return x
