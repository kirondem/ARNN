import torchvision
from lib import enums, constants, utils, plot_utils, cifar_features
from models.associative_network import AssociativeNetwork
from lib.utils import concat_images, dynamic_lambda
from lib.activation_functions import relu
import logging

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger('matplotlib.font_manager').disabled = True

logging.info(torchvision.__version__)