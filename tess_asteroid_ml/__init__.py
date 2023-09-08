import os
import logging

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

__version__ = '0.1.0'
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .tess_ffi_cutout import AsteroidTESScut
