"""
@file resnet18.py
@brief ResNet18 implementation in Keras.
"""
from tensorflow.keras import layers, models

# ==================================================================================
#                                  PUBLIC FUNCTIONS
# ==================================================================================

def ResNet18Functional(inputs):
    """
    Build a neural network with ResNet18 architecture.

    Args:
        `input_shape` (tuple): input shape.

    Returns:
        `model`: Keras model object. 
    """
    x = _conv_block(inputs, 16, kernel_size=3, stride=2)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    x = _identity_block(x, 16)
    x = _identity_block(x, 16)
    
    x = _conv_block_with_downsampling(x, 32)
    x = _identity_block(x, 32)
    
    x = _conv_block_with_downsampling(x, 64)
    x = _identity_block(x, 64)
    
    x = _conv_block_with_downsampling(x, 64)
    outputs = _identity_block(x, 64)
    
    return outputs

def ResNet18(input_shape):
    """
    Build a neural network with ResNet18 architecture.

    Args:
        `input_shape` (tuple): input shape.

    Returns:
        `model`: Keras model object. 
    """
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, 16, kernel_size=3, stride=2)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    x = _identity_block(x, 16)
    x = _identity_block(x, 16)
    
    x = _conv_block_with_downsampling(x, 32)
    x = _identity_block(x, 32)
    
    x = _conv_block_with_downsampling(x, 64)
    x = _identity_block(x, 64)
    
    x = _conv_block_with_downsampling(x, 64)
    outputs = _identity_block(x, 64)
    
    model = models.Model(inputs, outputs)
    return model

# ==================================================================================
#                                  PRIVATE FUNCTIONS
# ==================================================================================

def _conv_block(x, filters, kernel_size=3, stride=1):
    """
    Convolutional block with batch normalization and ReLU activation.

    Args:
        `x`: input tensor.
        `filters`: number of filters for the convolution.
        `kernel_size`: size of the convolutional kernel.
        `stride`: stride of the convolution.

    Returns:
        `x`: output tensor.
    """
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def _identity_block(x, filters):
    """
    Identity block (residual connection).

    Args:
        `x`: input tensor.
        `filters`: number of filters for the convolutional layer.

    Returns:
        x: output tensor.
    """
    shortcut = x
    x = _conv_block(x, filters)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def _conv_block_with_downsampling(x, filters):
    """
    Applies a convolutional block with downsampling.

    Args:
        `x` (tensor): input tensor.
        `filters` (int): number of filters for the convolutional layer.

    Returns:
        `x`: output tensor.
    """
    shortcut = layers.Conv2D(filters, 1, strides=2, padding='same', use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)
    x = _conv_block(x, filters, stride=2)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x
