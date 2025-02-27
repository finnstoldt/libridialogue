import pandas as pd
import itertools
import random
from libridialogue import settings
from libridialogue.simulate_dialogue_reverb import simulate_libridialogue_reverb
import os
from tqdm import tqdm
from pydub import AudioSegment


def build_libridialogue_clean(
    audio_in_1_path, audio_in_2_path, audio_out_1_path, audio_out_2_path, sample_rate
):
    # Load input audio files
    audio_1_in = AudioSegment.from_file(audio_in_1_path)
    audio_1_in = audio_1_in.set_frame_rate(sample_rate)
    audio_2_in = AudioSegment.from_file(audio_in_2_path)
    audio_2_in = audio_2_in.set_frame_rate(sample_rate)

    # Get lengths in seconds
    audio_1_in_length = len(audio_1_in) / 1000.0
    audio_2_in_length = len(audio_2_in) / 1000.0

    # Determine overlap, ensuring it's valid
    min_overlap = float(settings.LIBRIDIALOGUE_MIN_OVERLAP)
    max_overlap = min(
        float(settings.LIBRIDIALOGUE_MAX_OVERLAP), audio_1_in_length, audio_2_in_length
    )

    if min_overlap > max_overlap:
        # Adjust min_overlap if it's greater than max_overlap
        min_overlap = max_overlap

    # Calculate random overlap within the valid range
    overlap = random.uniform(min_overlap, max_overlap)

    # Compute output length
    audio_out_length = audio_1_in_length + audio_2_in_length - overlap

    # Create silent audio segments for outputs
    audio_1_out = AudioSegment.silent(
        duration=audio_out_length * 1000, frame_rate=sample_rate
    )
    audio_2_out = AudioSegment.silent(
        duration=audio_out_length * 1000, frame_rate=sample_rate
    )

    # Overlay input audios onto output segments
    audio_1_out = audio_1_out.overlay(audio_1_in)
    position = int((audio_1_in_length - overlap) * 1000)
    audio_2_out = audio_2_out.overlay(audio_2_in, position=position)

    # Ensure output directories exist
    if not os.path.exists(os.path.dirname(audio_out_1_path)):
        os.makedirs(os.path.dirname(audio_out_1_path))
    if not os.path.exists(os.path.dirname(audio_out_2_path)):
        os.makedirs(os.path.dirname(audio_out_2_path))

    # Export output audio files
    audio_1_out.export(audio_out_1_path, format="wav")
    audio_2_out.export(audio_out_2_path, format="wav")


def generate(
    librispeech_path=settings.LIBRISPEECH_PATH,
    libridialogue_path=settings.LIBRIDIALOGUE_PATH,
    libridialogue_size=settings.LIBRIDIALOGUE_SIZE,
):

    random.seed(settings.RANDOM_SEED)

    # check if the output directory exists
    if os.path.exists(libridialogue_path):
        print("Dataset already generated, skipping...")
        return
    else:
        os.makedirs(libridialogue_path)

    # read csv file
    df = pd.read_csv(
        librispeech_path + "/dataset.csv", sep=",", encoding="utf-8", header=0
    )

    pairs = list(itertools.combinations(df["id"], 2))

    selected_pairs = random.sample(pairs, int(libridialogue_size))

    # iterate over all pairs
    for pair in tqdm(selected_pairs):
        subset = df.loc[df["id"].isin(pair)]
        id_1 = subset.iloc[0]["id"]
        id_2 = subset.iloc[1]["id"]
        audio_in_1 = subset.iloc[0]["audiopath"]
        audio_in_2 = subset.iloc[1]["audiopath"]
        build_libridialogue_clean(
            audio_in_1,
            audio_in_2,
            f"{libridialogue_path}/8k/clean/{id_1}_{id_2}.wav",
            f"{libridialogue_path}/8k/clean/{id_2}_{id_1}.wav",
            8000,
        )
        build_libridialogue_clean(
            audio_in_1,
            audio_in_2,
            f"{libridialogue_path}/16k/clean/{id_1}_{id_2}.wav",
            f"{libridialogue_path}/16k/clean/{id_2}_{id_1}.wav",
            16000,
        )
        simulate_libridialogue_reverb(
            f"{libridialogue_path}/8k/clean/{id_1}_{id_2}.wav",
            f"{libridialogue_path}/8k/clean/{id_2}_{id_1}.wav",
            f"{libridialogue_path}/8k/reverb-solo/{id_1}_{id_2}.wav",
            f"{libridialogue_path}/8k/reverb-solo/{id_2}_{id_1}.wav",
            f"{libridialogue_path}/8k/reverb-dual/{id_1}_{id_2}.wav",
            f"{libridialogue_path}/8k/reverb-dual/{id_2}_{id_1}.wav",
        )
        simulate_libridialogue_reverb(
            f"{libridialogue_path}/16k/clean/{id_1}_{id_2}.wav",
            f"{libridialogue_path}/16k/clean/{id_2}_{id_1}.wav",
            f"{libridialogue_path}/16k/reverb-solo/{id_1}_{id_2}.wav",
            f"{libridialogue_path}/16k/reverb-solo/{id_2}_{id_1}.wav",
            f"{libridialogue_path}/16k/reverb-dual/{id_1}_{id_2}.wav",
            f"{libridialogue_path}/16k/reverb-dual/{id_2}_{id_1}.wav",
        )
