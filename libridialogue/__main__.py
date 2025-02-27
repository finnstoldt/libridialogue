from libridialogue.librispeech.download import download
from libridialogue.librispeech.generate_csv import generate_librispeech_csvs
from libridialogue.generate import generate
from libridialogue.util.separate_cosy import separate_cosy
from libridialogue.util.separate_mossformer2 import separate_mossformer2
from libridialogue.util.separate_asteroid import separate_asteroid
from libridialogue.util.analyze import analyze
from libridialogue import settings
import os
import shutil


def run():
    print("Downloading LibriSpeech dataset...")
    download(["test-clean"])

    print("Generating CSV files...")
    generate_librispeech_csvs()

    librispeech_path = settings.LIBRISPEECH_PATH + "/test-clean"
    librispeech_csv_path = librispeech_path + "/dataset.csv"
    libridialogue_path = settings.LIBRIDIALOGUE_PATH + "/test-clean"
    libridialogue_separated_path = settings.LIBRIDIALOGUE_SEPARATED_PATH + "/test-clean"

    if settings.OVERWRITE_GENERATED_DATA:
        print("Removing existing generated data...")
        if os.path.exists(libridialogue_path):
            shutil.rmtree(libridialogue_path)
        if os.path.exists(libridialogue_separated_path):
            shutil.rmtree(libridialogue_separated_path)

    print("Generating reverb-dual dataset...")
    generate(librispeech_path, libridialogue_path, settings.LIBRIDIALOGUE_SIZE)

    if settings.SEPARATE_COSY:
        print("Separating reverb-dual dataset with cosy...")
        separate_cosy(
            f"{libridialogue_path}/8k/reverb-dual",
            f"{libridialogue_separated_path}/cosy-8k",
            8000,
        )
        separate_cosy(
            f"{libridialogue_path}/16k/reverb-dual",
            f"{libridialogue_separated_path}/cosy-16k",
            16000,
        )

    if settings.SEPARATE_MOSSFORMER2:
        print("Separating reverb-dual dataset with moss...")
        separate_mossformer2(
            f"{libridialogue_path}/8k/reverb-dual",
            f"{libridialogue_separated_path}/mossformer2-8k",
            save_all_outputs=True,
        )

    if settings.SEPARATE_CONVTASNET:
        print("Separating reverb-dual dataset with convtasnet...")
        separate_asteroid(
            f"{libridialogue_path}/8k/reverb-dual",
            f"{libridialogue_separated_path}/convtasnet-8k",
            8000,
            model="JorisCos/ConvTasNet_Libri2Mix_sepnoisy_8k",
        )
        separate_asteroid(
            f"{libridialogue_path}/16k/reverb-dual",
            f"{libridialogue_separated_path}/convtasnet-16k",
            16000,
            model="JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",
        )

    if settings.ANALYZE_CLEAN:
        print("Analyzing clean dataset...")
        analyze(
            f"{libridialogue_path}/8k/clean",
            f"{libridialogue_path}/8k/reverb-solo",
            librispeech_csv_path,
            8000,
            name="clean-8k",
        )
        analyze(
            f"{libridialogue_path}/16k/clean",
            f"{libridialogue_path}/16k/reverb-solo",
            librispeech_csv_path,
            16000,
            name="clean-16k",
        )

    if settings.ANALYZE_REVERB_DUAL:
        print("Analyzing reverb-dual dataset...")
        analyze(
            f"{libridialogue_path}/8k/reverb-dual",
            f"{libridialogue_path}/8k/reverb-solo",
            librispeech_csv_path,
            8000,
            name="reverb-dual-8k",
        )
        analyze(
            f"{libridialogue_path}/16k/reverb-dual",
            f"{libridialogue_path}/16k/reverb-solo",
            librispeech_csv_path,
            16000,
            name="reverb-dual-16k",
        )

    if settings.ANALYZE_COSY:
        print("Analyzing cosy separated reverb-dual dataset...")
        analyze(
            f"{libridialogue_separated_path}/cosy-8k",
            f"{libridialogue_path}/8k/reverb-solo",
            librispeech_csv_path,
            8000,
        )
        analyze(
            f"{libridialogue_separated_path}/cosy-16k",
            f"{libridialogue_path}/16k/reverb-solo",
            librispeech_csv_path,
            16000,
        )

    if settings.ANALYZE_MOSSFORMER2:
        print("Analyzing reverb-dual dataset separated by moss...")
        analyze(
            f"{libridialogue_separated_path}/mossformer2-8k",
            f"{libridialogue_path}/8k/reverb-solo",
            librispeech_csv_path,
            8000,
        )

    if settings.ANALYZE_CONVTASNET:
        print("Analyzing reverb-dual dataset separated by convtasnet...")
        analyze(
            f"{libridialogue_separated_path}/convtasnet-8k",
            f"{libridialogue_path}/8k/reverb-solo",
            librispeech_csv_path,
            8000,
        )
        analyze(
            f"{libridialogue_separated_path}/convtasnet-16k",
            f"{libridialogue_path}/16k/reverb-solo",
            librispeech_csv_path,
            16000,
        )


if __name__ == "__main__":
    run()
