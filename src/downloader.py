import os

import requests
from loguru import logger
from tqdm.auto import tqdm


DATA_DIR = "data"
BASE_URL = "https://datasets.grouplens.org/movielens/"

# Files needed for the book recommender
DOWNLOAD_FILES = [
    "goodreads_books.json.gz",
    "goodreads_interactions.csv",
    "book_id_map.csv",
    "user_id_map.csv",
]


def download_file(filename, output_dir=DATA_DIR):
    """Download a file from UCSD McAuley Lab server.

    Args:
        filename (str): Name of the file to download.
        output_dir (str, optional): Directory to save file. Defaults to DATA_DIR.
    """
    url = f"https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/{filename}"
    output_path = os.path.join(output_dir, filename)

    logger.info(f"Downloading {filename} from {url}")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))

            with open(output_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Successfully downloaded {filename}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {filename}: {e}")
        raise


def download_goodreads_data(data_dir: str):
    """Download all required GoodReads datasets.

    Args:
        data_dir (str): Directory to save downloaded files.
    """
    os.makedirs(data_dir, exist_ok=True)

    for filename in DOWNLOAD_FILES:
        output_path = os.path.join(data_dir, filename)
        if os.path.exists(output_path):
            logger.info(f"Skipping {filename} (already exists)")
            continue

        download_file(filename, data_dir)


if __name__ == "__main__":
    download_goodreads_data(DATA_DIR)
