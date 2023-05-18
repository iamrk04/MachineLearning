"""
Contains function for downloading data from a url.
"""
import requests
import zipfile
from pathlib import Path
import os


def download_data(url: str, save_path_str: str) -> Path:
    # Setup path to a data folder
    parent_dir = Path(os.path.abspath(__file__)).parent.parent
    data_path = Path(os.path.join(parent_dir, "data"))
    image_path = Path(os.path.join(data_path, save_path_str))

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory already exists... skipping download")
    else: 
        print(f"{image_path} does not exist, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak and suhsi data
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get(url)
            print("Downloading pizza, steak, suhsi data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak and sushi data...")
            zip_ref.extractall(image_path)
    return image_path

if __name__ == "__main__":
    download_data(url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")