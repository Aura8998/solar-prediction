"""ResNet50 model for Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from GUI_CODE.projectGUI.static.DLFile.diffModel.resnet import resnet50


def ResNet50(*args, **kwargs):
    return resnet50.ResNet50(*args, **kwargs)


def decode_predictions(*args, **kwargs):
    return resnet50.decode_predictions(*args, **kwargs)


def preprocess_input(*args, **kwargs):
    return resnet50.preprocess_input(*args, **kwargs)
