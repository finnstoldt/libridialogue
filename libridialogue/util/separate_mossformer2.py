import soundfile as sf
import numpy as np
import librosa
import tempfile
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
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


def separate_mossformer2(
    input_dataset_path,
    output_dataset_path,
    save_all_outputs=False,
):
    # check if the output directory exists
    if os.path.exists(output_dataset_path):
        print("Dataset already separated by mossformer2, skipping...")
        return
    else:
        os.makedirs(output_dataset_path)

    files_to_skip = set()

    df = None
    if os.path.exists(settings.COMPUTATION_TIMES_CSV):
        df = pd.read_csv(settings.COMPUTATION_TIMES_CSV)
    else:
        df = pd.DataFrame(columns=["id", "timestamp", "file_count", "computation_time"])

    start = time()

    # Load the pipeline
    separation_pipe = pipeline(
        Tasks.speech_separation,
        model="damo/speech_mossformer2_separation_temporal_8k",
    )

    # iterate over all .wav files in the dataset path
    for file1 in tqdm(os.listdir(input_dataset_path)):
        if file1 in files_to_skip:
            continue
        if file1.endswith(".wav"):
            id_1 = file1.split("_")[0]
            id_2 = file1.split("_")[1].split(".")[0]
            file2 = f"{id_2}_{id_1}.wav"

            if not save_all_outputs:
                files_to_skip.add(file2)

            audio, sr = librosa.load(input_dataset_path + "/" + file1, sr=None)

            temp_file_path = merge_and_write(
                [
                    librosa.resample(
                        librosa.load(f"{input_dataset_path}/{file1}", sr=None)[0],
                        orig_sr=sr,
                        target_sr=8000,
                    ),
                    librosa.resample(
                        librosa.load(f"{input_dataset_path}/{file2}", sr=None)[0],
                        orig_sr=sr,
                        target_sr=8000,
                    ),
                ]
            )

            # Run the separation
            result = separation_pipe(temp_file_path)

            # Save the separated signals
            signals = result["output_pcm_list"]

            if not save_all_outputs:
                save_file1 = f"{output_dataset_path}/{id_1}_{id_2}.wav"
                sf.write(save_file1, np.frombuffer(signals[0], dtype=np.int16), 8000)
                save_file2 = f"{output_dataset_path}/{id_2}_{id_1}.wav"
                sf.write(save_file2, np.frombuffer(signals[1], dtype=np.int16), 8000)
            else:
                save_file1 = f"{output_dataset_path}/{id_1}_{id_2}-1.wav"
                sf.write(save_file1, np.frombuffer(signals[0], dtype=np.int16), 8000)
                save_file2 = f"{output_dataset_path}/{id_1}_{id_2}-2.wav"
                sf.write(save_file2, np.frombuffer(signals[1], dtype=np.int16), 8000)

            # remove the temporary file
            os.remove(temp_file_path)

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
