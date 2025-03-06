import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
from dataLoader import load_data


def encode_labels(labels: list) -> list:
    """
    Encodes a list of string labels into numerical values based on a predefined mapping.

    Args:
        labels (list): A list of string labels representing different classes.

    Returns:
        list: A list of integers corresponding to the encoded labels.
    
    Label Mapping:
        'apple' : 0,  'banana' : 1,  'beetroot' : 2,  'bell pepper' : 3,  
        'cabbage' : 4,  'capsicum' : 5,  'carrot' : 6,  'cauliflower' : 7,  
        'chilli pepper' : 8,  'corn' : 9,  'cucumber' : 10,  'eggplant' : 11,  
        'garlic' : 12,  'ginger' : 13,  'grapes' : 14,  'jalepeno' : 15,  
        'kiwi' : 16,  'lemon' : 17,  'lettuce' : 18,  'mango' : 19,  
        'onion' : 20,  'orange' : 21,  'paprika' : 22,  
        'pear' : 23,  'peas' : 24,  'pineapple' : 25,  'pomegranate' : 26,  
        'potato' : 27,  'raddish' : 28,  'soy beans' : 29,  'spinach' : 30,  
        'sweetcorn' : 31,  'sweetpotato' : 32,  'tomato' : 33,  
        'turnip' : 34,  'watermelon' : 35 
    """
    
    encode_label = []

    path_dir = r"C:\Jean Eudes Folder\_Projects\Computer_Vision_Project\Fruit&Vegetables\src\Datasets\train"
    lab_dir = os.listdir(path_dir)

    for ind, val in enumerate(lab_dir):
        for element in labels:
            if element == val:
                encode_label.append(ind)
    
    return encode_label

def preprocessing(_type: str = "train", resize_dim: tuple = (224, 224)) -> dict:
    """
    Preprocesses image data by resizing, normalizing, and encoding labels.

    Args:
        _type (str, optional): The type of dataset to preprocess. 
                               Can be "train", "validation", or "test". Defaults to "train".
        resize_dim (tuple, optional): The target dimensions (width, height) for image resizing. 
                                      Defaults to (224, 224).

    Returns:
        dict: A dictionary containing:
            - "feature" (list): A list of preprocessed images as normalized arrays.
            - "label" (list): A list of numerical labels corresponding to each image.

    Processing Steps:
        1. Load dataset images and labels using load_data().
        2. Encode labels into numerical values using encode_labels().
        3. Resize all images to `resize_dim` using OpenCV.
        4. Normalize pixel values by scaling them between 0 and 1.
        5. Return the processed images and encoded labels as a dictionary.
    """
   
    output = {
        "feature" : [],
        "label" : []
    }

    normalized_images = []

    # load dataset
    _dict = load_data(_type)
    images = _dict['feature']
    labels = _dict['label']

    # encode labels
    _Labels = encode_labels(labels)

    for im in images:
        # Resize the image
        img = cv2.resize(im, resize_dim)
        
        # normalize the image
        img_norm = img /255.0
        normalized_images.append(img_norm)
    
    # Convert lists to NumPy arrays
    normalized_images_arr = np.array(normalized_images, dtype=np.float32)
    _Labels_arr = np.array(_Labels, dtype=np.int32)

    # shuffle the data
    indices = np.arange(len(_Labels_arr))
    np.random.shuffle(indices)
    
    normalized_images_arr = normalized_images_arr[indices]
    _Labels_arr = _Labels_arr[indices]

    # save labels and images in the dictionary
    output["feature"] = normalized_images_arr
    output["label"] = _Labels_arr

    return output

if __name__ == "__main__":
    process = preprocessing()

    print(process["feature"][0])

    plt.imshow(process["feature"][0])
    plt.title(process["label"][0])
    plt.show()