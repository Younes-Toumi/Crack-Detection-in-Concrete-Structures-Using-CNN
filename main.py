import numpy as np
from Models.model_CNN import *
from Classes.Classifier import Classifier
import time


# Loading the target seperatly since it si the same for all
y = np.load('Loaded Data//y.npy')

# # Comparative study: Image preprocessing
file_names_preprocessing = ['original', 'grayscale', 'threshold', 'sobel']
execution_time_array_preprocessing = []

for file_name in file_names_preprocessing:
    
    # Loading the corresponding feature
    X = np.load('Loaded Data//X_'+ file_name + '.npy')

    model_CNN = create_model_CNN(type_vgg16='pretrained')

    # Training and Saving
    classifier = Classifier(model_CNN, X, y, device="GPU")  # Runs on GPU if available
    
    # Measuring Training Time
    start_time = time.time()
    classifier.train_model(amount=-1)
    end_time = time.time()
    classifier.save_training(file_name=file_name)

    execution_time = end_time - start_time
    execution_time_array_preprocessing.append(execution_time)
    print(f"Process for {file_name}: {execution_time:.2f} [s]")


print(f"Execution Time Array: {execution_time_array_preprocessing}")
# Execution Time Array: [188.53752326965332, 181.01556777954102, 172.31493496894836, 170.94777059555054]


# Comparative Study: Number of Images
file_names_amount = [256, 512, 1024, 2048, 4300]
execution_time_array_amount = []

# Loading the corresponding feature
X = np.load('Loaded Data//X_threshold.npy')

for file_name in file_names_amount:
    


    model_CNN = create_model_CNN(type_vgg16='pretrained')

    # Training and Saving
    classifier = Classifier(model_CNN, X, y, device="GPU")  # Runs on GPU if available
    
    # Measuring Training Time
    start_time = time.time()
    classifier.train_model(amount=file_name)
    end_time = time.time()
    classifier.save_training(file_name=str(file_name))

    execution_time = end_time - start_time
    execution_time_array_amount.append(execution_time)
    print(f"Process for {file_name}: {execution_time:.2f} [s]")


print(f"Execution Time Array: {execution_time_array_amount}")
# Execution Time Array: [29.15678334236145, 35.732768535614014, 55.38185429573059, 91.00420618057251, 173.36359858512878]
