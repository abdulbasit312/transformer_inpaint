import os
import numpy as np
import torch
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from scipy.linalg import sqrtm


def load_images_from_directory(directory,real_images_dir=None, image_size=(299, 299)):
    """
    Load images from a directory, resize them, and return as a list of NumPy arrays.

    Parameters:
        directory (str): Path to the directory containing images.
        image_size (tuple): Desired image size (H, W).

    Returns:
        list: List of NumPy arrays representing images.
    """
    images = []
    for filename in os.listdir(real_images_dir):
        filepath = os.path.join(directory, filename)
        try:
            img = Image.open(filepath).convert('RGB')
            img = img.resize(image_size)
            images.append(np.array(img))
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
    return images


def preprocess_images_numpy(image_list, device, batch_size=32):
    """
    Preprocess a list of NumPy images for the Inception v3 model.

    Parameters:
        image_list (list): List of NumPy arrays (H, W, C).
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size for processing.

    Returns:
        numpy.ndarray: Preprocessed features from the Inception v3 model.
    """
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    # Transform to resize and normalize images
    transform = Compose([
        Resize((299, 299)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    processed_images = []
    for img in image_list:
        # Convert NumPy array to PIL Image for transformations
        pil_img = Image.fromarray(img.astype('uint8'))
        processed_images.append(transform(pil_img))
    processed_images = torch.stack(processed_images).to(device)

    # Extract features
    with torch.no_grad():
        features = []
        for i in range(0, len(processed_images), batch_size):
            batch = processed_images[i:i + batch_size]
            features.append(model(batch).detach().cpu())
    return torch.cat(features).numpy()


def calculate_fid(real_features, generated_features):
    """
    Compute the Fréchet Inception Distance (FID) score.

    Parameters:
        real_features (numpy.ndarray): Features from real images.
        generated_features (numpy.ndarray): Features from generated images.

    Returns:
        float: The FID score.
    """
    # Compute mean and covariance
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    # Add small value to the diagonal for numerical stability
    epsilon = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * epsilon
    sigma2 += np.eye(sigma2.shape[0]) * epsilon

    # Compute FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    # Handle numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


# Main Function
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories for real and generated images
    real_images_dir = "/w/331/abdulbasit/data/x-medium/test"
    generated_images_dir = "/scratch/expires-2024-Dec-14/abdulbasit/partial_1"

    # Load images
    print("Loading real images...")
    real_images = load_images_from_directory(real_images_dir,real_images_dir=real_images_dir)
    print(f"Loaded {len(real_images)} real images.")

    print("Loading generated images...")
    generated_images = load_images_from_directory(generated_images_dir,real_images_dir=real_images_dir)
    print(f"Loaded {len(generated_images)} generated images.")

    # Preprocess images and extract features
    print("Extracting features for real images...")
    real_features = preprocess_images_numpy(real_images, device)
    print("Extracting features for generated images...")
    generated_features = preprocess_images_numpy(generated_images, device)

    # Compute FID
    print("Calculating FID score...")
    fid_score = calculate_fid(real_features, generated_features)
    print(f"FID Score: {fid_score}")
