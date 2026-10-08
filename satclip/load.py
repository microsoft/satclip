import inspect

from .main import *

def get_satclip(ckpt_path, device, return_all=False):
    ckpt = torch.load(ckpt_path,map_location=device)
    # Checkpoints can carry hyperparameters SatCLIPLightningModule does not accept:
    # released ones have legacy keys (e.g. eval_downstream), and ones trained via
    # LightningCLI have '_instantiator'. Keep only the constructor's arguments.
    accepted = inspect.signature(SatCLIPLightningModule.__init__).parameters
    hparams = {k: v for k, v in ckpt['hyper_parameters'].items() if k in accepted}
    lightning_model = SatCLIPLightningModule(**hparams).to(device)

    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()

    geo_model = lightning_model.model

    if return_all:
        return geo_model
    else:
        return geo_model.location
