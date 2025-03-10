import numpy as np
from keras import layers, models
from Models.model_VGG16 import choose_vgg_16 # type: ignore
from keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def create_model_CNN(type_vgg16):

    # Add the pretrained VGG16 model (without the top classification layers)
    vgg16_base = choose_vgg_16(type_vgg16=type_vgg16)

    # Sequential layer for data augmentation
    data_augmentation = models.Sequential([
        layers.RandomRotation(0.2/(2*np.pi), name="random_rotation"),  # Apply random rotations
        layers.RandomFlip("horizontal_and_vertical", name="random_flip")  # Apply random flipping
    ], name="sequential")


    # Neural Network model
    model_CNN = models.Sequential([
        
        layers.InputLayer(input_shape=(227, 227, 3), name="input_layer"),
        data_augmentation,
        layers.Normalization(name='normalization'),
        vgg16_base,
        layers.GlobalMaxPooling2D(name="global_max_pooling2d"), # Add Global Max Pooling layer
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(1, activation="sigmoid", name="dense") # Add Dense layer for binary classification

    ], name='CNN_for_Crack_Detection')

    # Compile the model
    model_CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_CNN.summary()
    return model_CNN

