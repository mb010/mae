from .vision import *
from .rgz import *

datasets = {
    "stl10": STL10_DataModule,
    # "imagenette": Imagenette_DataModule,
    "rgz": RGZ_DataModule,
}
