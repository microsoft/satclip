"""SatCLIP: a global, general-purpose geographic location encoder.

See https://github.com/microsoft/satclip for details.
"""

from .load import get_satclip
from .location_encoder import LocationEncoder
from .loss import SatCLIPLoss
from .model import SatCLIP

__all__ = ["SatCLIP", "LocationEncoder", "SatCLIPLoss", "get_satclip"]
