import jiwer
import os
import csv
import torchaudio
from transformers import pipeline
import torch
import string
from tqdm import tqdm
import pandas as pd
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import (
    SignalDistortionRatio,
)  # Import SDR and SIR
from libridialogue import settings
import numpy as np


def analyze_with_channel_duplicates(
    file_path: str, target_path: str, csv_path: str, rate: int
):
    dataset = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            dataset.append(row)

    file_list = os.listdir(file_path)
    file_list = [file for file in file_list if file != ".DS_Store"]

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        device=torch.device("mps"),
        return_timestamps=True,
    )

    translator = str.maketrans("", "", string.punctuation.replace("'", ""))

    transcripts = {}

    for file in tqdm(file_list, desc="Transcribing files"):
        result = pipe(os.path.join(file_path, file))
        transcript = result["text"].upper().translate(translator)
        transcripts[os.path.basename(file)] = transcript

    speakers = set()

    for key in transcripts.keys():
        speakers.add(key.split("_")[0])

    mers = []

    for speaker in speakers:
        speaker_files = [file for file in transcripts.keys() if speaker in file]
        speaker_transcripts = [transcripts[file] for file in speaker_files]
        speaker_mers = []
        truth = next((entry for entry in dataset if entry.get("id") == speaker), None)

        for speaker_transcript in speaker_transcripts:
            speaker_mer = jiwer.process_words(truth["text"], speaker_transcript).mer
            speaker_mers.append(speaker_mer)

        mers.append(min(speaker_mers))

    mean_mer = sum(mers) / len(mers)

    si_snrs = []
    stois = []
    sdrs = []
    sirs = []

    for speaker in speakers:
        speaker_files = [
            file for file in transcripts.keys() if file.startswith(speaker)
        ]

        speaker_si_snrs = []
        speaker_stois = []
        speaker_sdrs = []

        for speaker_file in speaker_files:
            target_file = os.path.join(
                target_path, speaker_file.replace("-1", "").replace("-2", "")
            )
            if not os.path.exists(target_file):
                print(f"Warning: Target file {target_file} does not exist. Skipping.")
                continue

            pred, pred_rate = torchaudio.load(os.path.join(file_path, speaker_file))
            target, target_rate = torchaudio.load(target_file)

            # Ensure signals are not empty
            if pred.shape[-1] == 0 or target.shape[-1] == 0:
                print(f"Warning: Empty audio signal detected. Skipping {speaker_file}.")
                continue

            # Convert to mono if necessary
            if pred.shape[0] > 1:
                pred = torch.mean(pred, dim=0, keepdim=True)
            if target.shape[0] > 1:
                target = torch.mean(target, dim=0, keepdim=True)

            # Ensure same sampling rate
            if pred_rate != target_rate:
                resample_transform = torchaudio.transforms.Resample(
                    orig_freq=pred_rate, new_freq=target_rate
                )
                pred = resample_transform(pred)
                pred_rate = target_rate  # Update pred_rate

            # Truncate to the same length
            min_length = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_length]
            target = target[..., :min_length]

            # Initialize metrics
            si_snr_metric = ScaleInvariantSignalNoiseRatio()
            stoi_metric = ShortTimeObjectiveIntelligibility(
                fs=target_rate, extended=False
            )
            sdr_metric = SignalDistortionRatio()

            # Compute SI-SNR
            speaker_si_snr = si_snr_metric(
                pred.unsqueeze(0), target.unsqueeze(0)
            ).item()
            speaker_si_snrs.append(speaker_si_snr)

            # Compute STOI
            # Convert tensors to numpy arrays for STOI computation
            pred_np = pred.squeeze().numpy()
            target_np = target.squeeze().numpy()

            # Ensure the signals are in the correct shape (batch, time)
            pred_np = pred_np[np.newaxis, :]
            target_np = target_np[np.newaxis, :]

            # Compute STOI
            speaker_stoi = stoi_metric(
                torch.from_numpy(pred_np), torch.from_numpy(target_np)
            ).item()
            speaker_stois.append(speaker_stoi)

            # Compute SDR
            speaker_sdr = sdr_metric(pred.unsqueeze(0), target.unsqueeze(0)).item()
            speaker_sdrs.append(speaker_sdr)

        if speaker_si_snrs:
            si_snrs.append(max(speaker_si_snrs))
        if speaker_stois:
            stois.append(max(speaker_stois))
        if speaker_sdrs:
            sdrs.append(max(speaker_sdrs))

    mean_si_snr = 0 if not si_snrs else sum(si_snrs) / len(si_snrs)
    mean_stoi = 0 if not stois else sum(stois) / len(stois)
    mean_sdr = 0 if not sdrs else sum(sdrs) / len(sdrs)
    mean_sir = 0 if not sirs else sum(sirs) / len(sirs)

    return mean_mer, mean_si_snr, mean_stoi, mean_sdr, mean_sir


def analyze(
    file_path: str,
    target_path: str,
    csv_path: str,
    rate: int,
    name=None,
) -> None:
    """
    Function to analyze audio files and compute metrics.
    """
    if name is None:
        name = os.path.basename(file_path)

    if os.path.exists(settings.ANALYSIS_CSV):
        df = pd.read_csv(settings.ANALYSIS_CSV)
    else:
        df = pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "mean_mer",
                "mean_si_snr",
                "mean_stoi",
                "mean_sdr",
                "mean_sir",
            ]
        )

    mean_mer, mean_si_snr, mean_stoi, mean_sdr, mean_sir = (
        analyze_with_channel_duplicates(file_path, target_path, csv_path, rate)
    )

    # Add result as a row to the dataframe
    result = pd.DataFrame(
        {
            "id": name,
            "timestamp": pd.Timestamp.now(),
            "mean_mer": mean_mer,
            "mean_si_snr": mean_si_snr,
            "mean_stoi": mean_stoi,
            "mean_sdr": mean_sdr,
            "mean_sir": mean_sir,
        },
        index=[0],
    )
    df = pd.concat([df, result], ignore_index=True)
    df.to_csv(settings.ANALYSIS_CSV, index=False)
