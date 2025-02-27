import os
import subprocess
from tqdm import tqdm
import pandas as pd
from time import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from libridialogue import settings

SEPARATOR_BASE_COMMAND = [
    "./bin/cosy_speaker_separator",
    "-c",
    settings.COSY_COMPRESSOR_PLUGIN_FILE,
    "-g",
    settings.COSY_GATE_PLUGIN_FILE,
]


def process_file(
    file1,
    input_dataset_path,
    output_dataset_path,
    rate,
    config_file,
    rms_threshold,
):
    id_1 = file1.split("_")[0]
    id_2 = file1.split("_")[1].split(".")[0]
    file2 = f"{id_2}_{id_1}.wav"
    command = SEPARATOR_BASE_COMMAND + [
        "-i",
        os.path.join(input_dataset_path, file1),
        "-j",
        os.path.join(input_dataset_path, file2),
        "-o",
        os.path.join(output_dataset_path, f"{id_1}_{id_2}.wav"),
        "-f",
        str(rate),
        "--config",
        config_file,
    ]
    # Include rms_threshold if provided
    if rms_threshold is not None:
        command += ["-r", str(rms_threshold)]
    subprocess.call(command, text=True)


def separate_cosy(
    input_dataset_path,
    output_dataset_path,
    rate,
    config_file=settings.COSY_CONFIG_FILE,
    rms_threshold=None,
):
    # Check if the output directory exists
    if os.path.exists(output_dataset_path):
        print("Dataset already separated by cosy, skipping...")
        return
    else:
        os.makedirs(output_dataset_path)

    # Load or create the computation times DataFrame
    if os.path.exists(settings.COMPUTATION_TIMES_CSV):
        df = pd.read_csv(settings.COMPUTATION_TIMES_CSV)
    else:
        df = pd.DataFrame(columns=["id", "timestamp", "file_count", "computation_time"])

    start = time()

    # Build a list of unique files to process
    file_list = [f for f in os.listdir(input_dataset_path) if f.endswith(".wav")]

    # Prepare partial function with fixed arguments
    process_file_partial = partial(
        process_file,
        input_dataset_path=input_dataset_path,
        output_dataset_path=output_dataset_path,
        rate=rate,
        config_file=config_file,
        rms_threshold=rms_threshold,
    )

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_file_partial, file_list),
                total=len(file_list),
                desc="Processing files",
            )
        )

    end = time()

    # Record computation time
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
