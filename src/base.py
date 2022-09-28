from __future__ import division
from configparser import ConfigParser
import os


class Base(object):
    parser = ConfigParser()
    PATH  =  os.path.dirname(os.path.abspath(__file__))
    
    parser.read(os.path.join(PATH, 'configuration.ini'))

    lambda_max = float(parser.get('Network_Config', 'lambda_max'))

    def __init__(self):
        pass

    
  


