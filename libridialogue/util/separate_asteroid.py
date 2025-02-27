import soundfile as sf
import numpy as np
import tempfile
from asteroid.models import BaseModel
import os
from tqdm import tqdm
from time import time
import pandas as pd

from libridialogue import settings


# merge the separated signals into a single mono audio file and write it as a temporary wav file
def merge_and_write(separated_signals, sr=8000):
    # merge the separated signals into a single mono audio file
    merged_signal = np.sum(separated_signals, axis=0)

    # write the merged signal to a temporary wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file_path = temp_file.name
        sf.write(temp_file_path, merged_signal, sr)

    return temp_file_path


def separate_asteroid(
    input_dataset_path,
    output_dataset_path,
    rate,
    model="JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",
):
    # check if the output directory exists
    if os.path.exists(output_dataset_path):
        print("Dataset already separated by asteroid, skipping...")
        return
    else:
        os.makedirs(output_dataset_path)

    df = None
    if os.path.exists(settings.COMPUTATION_TIMES_CSV):
        df = pd.read_csv(settings.COMPUTATION_TIMES_CSV)
    else:
        df = pd.DataFrame(columns=["id", "timestamp", "file_count", "computation_time"])

    start = time()

    model = BaseModel.from_pretrained(model)

    # iterate over all .wav files in the dataset path
    for file in tqdm(os.listdir(input_dataset_path)):
        if file.endswith(".wav"):
            id_1 = file.split("_")[0]
            id_2 = file.split("_")[1].split(".")[0]

            # You can pass a NumPy array:
            mixture, _ = sf.read(
                input_dataset_path + "/" + file,
                dtype="float32",
                always_2d=True,
            )

            # Soundfile returns the mixture as shape (time, channels),
            # and Asteroid expects (batch, channels, time)
            mixture = mixture.transpose()
            mixture = mixture.reshape(1, mixture.shape[0], mixture.shape[1])

            signals = model.separate(mixture)[0]

            save_file1 = f"{output_dataset_path}/{id_1}_{id_2}-1.wav"
            sf.write(save_file1, signals[0], rate)
            save_file2 = f"{output_dataset_path}/{id_1}_{id_2}-2.wav"
            sf.write(save_file2, signals[1], rate)

    end = time()

    result = pd.DataFrame(
        {
            "id": output_dataset_path.split("/")[-1],
            "timestamp": pd.Timestamp.now(),
            "file_count": int(settings.LIBRIDIALOGUE_SIZE) * 2,
            "computation_time": end - start,
        },
        index=[0],
    )
    df = pd.concat([df, result], ignore_index=True)
    df.to_csv(settings.COMPUTATION_TIMES_CSV, index=False)
