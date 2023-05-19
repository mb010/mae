from .vision import *
from .rgz import *
from .fits import FITS_DataModule

datasets = {
    "stl10": STL10_DataModule,
    # "imagenette": Imagenette_DataModule,
    "rgz": RGZ_DataModule,
    "fits": FITS_DataModule,
    "fits_rgz": FITS_DataModule,
}
