# flake8: noqa
import warnings

import numpy as np
import torch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from mlops.trainer import BasePythonModel, MLFlowLightningModule

from .main import SatCLIPLightningModule


def get_satclip(ckpt_path, return_all=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt["hyper_parameters"].pop("eval_downstream")
    ckpt["hyper_parameters"].pop("air_temp_data_path")
    ckpt["hyper_parameters"].pop("election_data_path")
    lightning_model = SatCLIPLightningModule(**ckpt["hyper_parameters"])

    lightning_model.load_state_dict(ckpt["state_dict"])
    lightning_model.eval()

    geo_model = lightning_model.model

    if return_all:
        return geo_model
    else:
        return geo_model.location


class SatClipWrapper(BasePythonModel):

    def predict(self, context, model_input) -> np.ndarray:

        # Input ============================================================================
        # Extract patch, height, and width from the input dictionary
        patch = model_input["patch"]  # Input Patch

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
        ckpt_path=None,
    ):

        super().__init__()

        # Utility ==========================================================================
        self.name = "SatClip"
        self.wrapper = SatClipWrapper
        self.signature = self.wrapper.get_signature()

        # Build Model ======================================================================
        self.model = None
        if ckpt_path is not None:
            splits = ckpt_path.split("-")
            emb_model = splits[-2]
            emb_size = splits[-1].split(".")[0]

            self.name = f"{self.name}.{emb_model.upper()}.{emb_size.upper()}"

            self.model = get_satclip(
                ckpt_path=ckpt_path,
                return_all=False,
            )

        self.ckpt_path = ckpt_path

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        x = self.model(x.double()).detach().cpu()

        return x
