import os
import requests

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'


def check_downloaded_data(file_path: str='data/wine.csv') -> bool:
    """
    Check if the data file has already been downloaded.

    Args:
        file_path (str): The path to the data file.
    """
    return os.path.exists(file_path)

def download_data(url: str, file_path: str='data/wine.csv') -> None:
    """
    Download data from the specified URL and save it to the given file path.

    Args:
        url (str): The URL to download the data from.
        file_path (str): The path to save the downloaded data file.
    """

    response = requests.get(url)
    response.raise_for_status()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as file:
        file.write(response.content)