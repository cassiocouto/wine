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

def download_data(url: str, file_path: str='data/wine.csv', timeout: int=30) -> None:
    """
    Download data from the specified URL and save it to the given file path.

    Args:
        url (str): The URL to download the data from.
        file_path (str): The path to save the downloaded data file.
        timeout (int): Timeout for the request in seconds. Default is 30.
    """

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)