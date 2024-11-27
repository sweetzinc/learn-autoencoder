from .vanilla_vae import *
from .linear_vae import *
from .encoders import *
from .decoders import *

vae_models = {
    'VanillaVAE': VanillaVAE,
    'LinearVAE': LinearVAE
}