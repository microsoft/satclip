from main import *

def get_satclip(ckpt_path, device, return_all=False):
    ckpt = torch.load(ckpt_path,map_location=device)
    ckpt['hyper_parameters'].pop('eval_downstream')
    ckpt['hyper_parameters'].pop('air_temp_data_path')
    ckpt['hyper_parameters'].pop('election_data_path')
    lightning_model = SatCLIPLightningModule(**ckpt['hyper_parameters']).to(device)

    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()

    geo_model = lightning_model.model

    if return_all:
        return geo_model.eval
    else:
        return geo_model.location