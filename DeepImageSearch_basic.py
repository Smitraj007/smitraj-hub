import DeepImageSearch.config as config
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import   
 numpy as np
from torchvision import transforms
import torch
from torch.autograd import Variable
import timm
from PIL import ImageOps
import math
import faiss   


def extract_last_layer_embeddings(image_path, model_name='vgg19', pretrained=True):
    """
    Extracts the embeddings from the last layer of the model for a given image.

    Parameters:
    -----------
    image_path : str
        The path to the image.
    model_name : str, optional (default='vgg19')
        The name of the pre-trained model to use for feature extraction.
    pretrained : bool, optional (default=True)
        Whether to use the pre-trained weights for the chosen model.   


    Returns:
    --------
    numpy.ndarray
        The embeddings from the last layer of the model.
    """

    base_model = timm.create_model(model_name, pretrained=pretrained)
    model = torch.nn.Sequential(*list(base_model.children())[:-1])
    model.eval()

    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = img.convert('RGB')

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = preprocess(img)
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

    # Extract features   
 from the last layer
    embeddings = model(x)
    embeddings = embeddings.data.numpy().flatten()
    return embeddings

def load_images_from_folder(folder_path):
    """
    Loads images from a given folder.

    Parameters:
    -----------
    folder_path : str
        The path to the folder containing the images.

    Returns:
    --------
    list
        A list of image paths.
    """

    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths   


def perform_similarity_search(query_image_path, database_folder_path, model_name='vgg19', pretrained=True):
    """
    Performs similarity search between a query image and a database of images.

    Parameters:
    -----------
    query_image_path : str
        The path to the query image.
    database_folder_path : str
        The path to the folder containing the database images.
    model_name : str, optional (default='vgg19')
        The name of the pre-trained model to use for feature extraction.
    pretrained : bool, optional (default=True)
        Whether to use the pre-trained weights for the chosen model.   


    Returns:
    --------
    list
        A list of paths to the most similar images in the database.
    """

    # Load images from the database folder
    database_image_paths = load_images_from_folder(database_folder_path)

    # Extract embeddings for the query image and database images
    query_embeddings = extract_last_layer_embeddings(query_image_path, model_name, pretrained)
    database_embeddings = [extract_last_layer_embeddings(image_path, model_name, pretrained) for image_path in database_image_paths]

    # Create a Faiss index for efficient similarity search
    index = faiss.IndexFlatL2(len(query_embeddings))
    index.add(np.array(database_embeddings, dtype=np.float32))

    # Search for similar images
    distances, indices = index.search(np.array([query_embeddings], dtype=np.float32), k=10)  # Find the 10 most similar images

    # Get the paths of the similar images
    similar_image_paths = [database_image_paths[index] for index in indices[0]]

    return similar_image_paths

def plot_images(image_paths):
    """
    Plots the given images.

    Parameters:
    -----------
    image_paths : list
        A list of paths to the images to be plotted.
    """

    num_images = len(image_paths)
    cols = 5
    rows = num_images // cols
    if num_images % cols != 0:
        rows += 1

    plt.figure(figsize=(15, 15))
    for i, image_path in enumerate(image_paths):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(Image.open(image_path))
        plt.axis('off')

    plt.show()

# Example usage
query_image_path = 'path/to/your/query_image.jpg'
database_folder_path = 'path/to/your/database/folder'

similar_image_paths = perform_similarity_search(query_image_path, database_folder_path)
print(similar_image_paths)

plot_images(similar_image_paths)