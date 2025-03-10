import os
import cv2
import matplotlib.pyplot as plt


def fetch_images(path: str, directory: str) -> list:
    """
    Retrieves a list of image filenames from a specified directory.

    Args:
        path (str): The base directory path.
        directory (str): The name of the subdirectory containing the images.

    Returns:
        list: List of images converted to arrays in the specified directory
    """
    images = []

    _path = os.path.join(path, directory)

    if not os.path.exists(_path):
        print(f"Error: Directory '{_path}' not found.")
        return []
    
    output = os.listdir(_path)

    # read images with openCV
    for element in output:
        path_im = os.path.join(_path, element)

        images_cv2 = cv2.imread(path_im, cv2.IMREAD_UNCHANGED)

        if images_cv2 is None:  # Checks if the image is loaded
            print(f"Warning: Unable to load image {path_im}. Skipping...")
            continue  # Go to next image

        images_cv2 = cv2.cvtColor(images_cv2, cv2.COLOR_BGR2RGB)
        images.append(images_cv2)
    
    return images


def load_data(_type: str = "train") -> dict:
    """
    Loads image filenames and their corresponding labels from a specified dataset type.

    Args:
        _type (str, optional): The type of dataset to load. Can be "train", "validation", or "test".
                               Defaults to "train".

    Returns:
        dict: A dictionary containing:
            - "feature" (list): A list of image filenames.
            - "label" (list): A list of labels corresponding to each image.
    """
    output = {
        "feature" : [],
        "label" : []
    }

    images = []
    labels = []

    path_dir = r"C:\Jean Eudes Folder\_Projects\Computer_Vision_Project\Fruit&Vegetables\src\Datasets"
    all_datasets = os.listdir(path_dir)

    # retrieving train, validation, testing paths
    train_path = os.path.join(path_dir, all_datasets[1])
    val_path = os.path.join(path_dir, all_datasets[2])
    test_path = os.path.join(path_dir, all_datasets[0])

    train_dir = os.listdir(train_path)
    val_dir = os.listdir(val_path)
    test_dir = os.listdir(test_path)

    if _type == "train":
        for element in train_dir:
            _list = fetch_images(train_path, element)
            # adapt labels in line with images
            labels += [element for i in range(len(_list))]
            for im in _list:
                images.append(im)
        
        # save labels and images in the dictionary
        output["feature"] = images
        output["label"] = labels

    elif _type == "validation":
        for element in val_dir:
            _list = fetch_images(val_path, element)
            # adapt labels in line with images
            labels += [element for i in range(len(_list))]
            for im in _list:
                images.append(im)
        
        # save labels and images in the dictionary
        output["feature"] = images
        output["label"] = labels

    elif _type == "test":
        for element in test_dir:
            _list = fetch_images(test_path, element)
            # adapt labels in line with images
            labels += [element for i in range(len(_list))]
            for im in _list:
                images.append(im)

        # save labels and images in the dictionary
        output["feature"] = images
        output["label"] = labels

    return output

def visualize_data(rows: int = 4, cols: int = 9) -> None:
    """
    Visualizes a grid of images from the dataset.

    Args:
        rows (int, optional): Number of rows in the grid. Defaults to 4.
        cols (int, optional): Number of columns in the grid. Defaults to 9.
    """
    data = []
    path = r"C:\Jean Eudes Folder\_Projects\Computer_Vision_Project\Fruit&Vegetables\src\Datasets\train"
    # retrieve labels
    labels = os.listdir(path)
    
    for element in labels:
        # retrieve images by category
        data.append(fetch_images(path, element)[0])
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Fruit & Vegetables Dataset", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i >= len(data):
            break  # Stop if we run out of images
        
        img = data[i]
        label = labels[i]

        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    
    # save the plot
    save_path = r"C:\Jean Eudes Folder\_Projects\Computer_Vision_Project\Fruit&Vegetables\src\Outcomes\Fruit&Vegetables_Dataset.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=600) # Well-framed and high-quality image

    plt.show()



if __name__ == "__main__":
    train_dataset = load_data(_type="train")
    validation_dataset = load_data(_type="validation")
    test_dataset = load_data(_type="test")

    print(f"Training dataset size : {len(train_dataset['feature'])}")
    print(f"Validation dataset size : {len(validation_dataset['feature'])}")
    print(f"Testing dataset size : {len(test_dataset['feature'])}")

    # visualize_data()