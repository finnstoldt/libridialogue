import os
import csv
import random
import string

from libridialogue import settings


def generate_csv(dir_path) -> None:
    """
    Loader function for librispeech dataset
    Stores list of all files in a .csv file
    Assigns UUID4 to every file
    Expects a .txt file containing file id and transcript in the folder containing audio files

    Args:
    - dir_path (str): Path to root folder of dataset
    - csv_path (str): Path to the .csv file
    """

    print(f"Generating CSV for {dir_path}...")

    csv_path = dir_path + "/dataset.csv"

    if os.path.exists(csv_path):
        print("CSV file already exists, skipping...")
        return

    set = []

    for dirpath, dirnames, filenames in os.walk(dir_path):
        if filenames and not filenames == [".DS_Store"]:

            txt_files = [file for file in filenames if file.endswith(".txt")]
            for file in txt_files:
                with open(os.path.join(dirpath, file), "r") as f:
                    content = f.readlines()
                for line in content:
                    id, text = line.split(" ", 1)
                    set.append(
                        {
                            "id": "".join(
                                random.choices(
                                    string.ascii_letters + string.digits, k=6
                                )
                            ),
                            "audiopath": os.path.join(dirpath, f"{id}.flac"),
                            "text": text[:-1],  # exclude '\n'
                        }
                    )

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["id", "audiopath", "text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(set)
    # return set


def generate_librispeech_csvs(librispeech_path=settings.LIBRISPEECH_PATH):
    """
    Generate CSV files for LibriSpeech dataset
    """
    for dataset in [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]:
        if os.path.exists(f"{librispeech_path}/{dataset}"):
            generate_csv(f"{librispeech_path}/{dataset}")
