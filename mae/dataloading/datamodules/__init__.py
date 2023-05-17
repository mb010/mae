from .vision import *
from .rgz import *
from .fits import FIRST_DataModule

datasets = {
    "stl10": STL10_DataModule,
    # "imagenette": Imagenette_DataModule,
    "rgz": RGZ_DataModule,
    "fits": FIRST_DataModule,
}
