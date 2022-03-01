from __future__ import division
from configparser import ConfigParser


class Base(object):
    parser = ConfigParser()
    parser.read('configuration.ini')

    lambda_max = float(parser.get('Network_Config', 'lambda_max'))

    def __init__(self):
        pass

    
  


