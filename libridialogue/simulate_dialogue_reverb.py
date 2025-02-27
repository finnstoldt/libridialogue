import random
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import uuid
from libridialogue.util import stereo_to_mono
import os
from libridialogue import settings  # Import the settings module


def simulate_libridialogue_reverb(
    audio_in_1,
    audio_in_2,
    audio_out_single_1,
    audio_out_single_2,
    audio_out_1,
    audio_out_2,
):

    if not os.path.exists(os.path.dirname(audio_out_single_1)):
        os.makedirs(os.path.dirname(audio_out_single_1))
    if not os.path.exists(os.path.dirname(audio_out_single_2)):
        os.makedirs(os.path.dirname(audio_out_single_2))
    if not os.path.exists(os.path.dirname(audio_out_1)):
        os.makedirs(os.path.dirname(audio_out_1))
    if not os.path.exists(os.path.dirname(audio_out_2)):
        os.makedirs(os.path.dirname(audio_out_2))

    # Import mono wavfiles as source signals
    audio1, fs = sf.read(audio_in_1)
    audio2, fs = sf.read(audio_in_2)

    # Use environment variables from settings
    min_temperature = float(settings.LIBRIDIALOGUE_ROOM_MIN_TEMPERATURE)
    max_temperature = float(settings.LIBRIDIALOGUE_ROOM_MAX_TEMPERATURE)

    min_humidity = float(settings.LIBRIDIALOGUE_ROOM_MIN_HUMIDITY)
    max_humidity = float(settings.LIBRIDIALOGUE_ROOM_MAX_HUMIDITY)

    min_z = float(settings.LIBRIDIALOGUE_ROOM_MIN_Z)
    max_z = float(settings.LIBRIDIALOGUE_ROOM_MAX_Z)
    z = random.uniform(min_z, max_z)

    min_x = float(settings.LIBRIDIALOGUE_ROOM_MIN_X)
    max_x = float(settings.LIBRIDIALOGUE_ROOM_MAX_X)
    xy_diff = float(
        settings.LIBRIDIALOGUE_ROOM_XY_DIFF
    )  # Range of difference of x and y

    x = random.uniform(min_x, max_x)
    y = random.uniform(x - (xy_diff / 2), x + (xy_diff / 2))

    source_z_options = [float(val) for val in settings.LIBRIDIALOGUE_ROOM_SOURCE_Z]
    step_size_source = float(settings.LIBRIDIALOGUE_ROOM_STEP_SIZE_SOURCE)

    num_steps_source_x = int((x - 1) / step_size_source) + 1
    num_steps_source_y = int((y - 1) / step_size_source) + 1

    room_dim = [x, y, z]  # meters

    rt60_tgt = float(settings.LIBRIDIALOGUE_ROOM_RT60_TGT)  # seconds

    e_absorption, max_order = pra.inverse_sabine(
        rt60_tgt, room_dim
    )  # Invert Sabine's formula

    temperature_val = random.uniform(min_temperature, max_temperature)
    humidity_val = random.uniform(min_humidity, max_humidity)

    room_kwargs = {
        "p": room_dim,
        "fs": fs,
        "materials": pra.Material(e_absorption),
        "max_order": max_order,
        "air_absorption": True,
        "ray_tracing": True,
        "temperature": temperature_val,
        "humidity": humidity_val,
    }

    # Randomly place first source using steps
    source1_x = round(
        random.randint(0, num_steps_source_x - 1) * step_size_source + 1, 2
    )
    source1_y = round(
        random.randint(0, num_steps_source_y - 1) * step_size_source + 1, 2
    )
    source1_z = random.choice(source_z_options)

    x_direction = np.argmax([source1_x, x - source1_x])
    y_direction = np.argmax([source1_y, y - source1_y])

    source_xy_diff = 1.0  # Offset between sources in meters

    if x_direction == 0:
        source2_x = source1_x - source_xy_diff
        mic2_x = source1_x - source_xy_diff + 0.1
        mic1_x = source1_x - 0.1
    else:
        source2_x = source1_x + source_xy_diff
        mic2_x = source1_x + source_xy_diff - 0.1
        mic1_x = source1_x + 0.1

    if y_direction == 0:
        source2_y = source1_y - source_xy_diff
        mic2_y = source1_y - source_xy_diff + 0.1
        mic1_y = source1_y - 0.1
    else:
        source2_y = source1_y + source_xy_diff
        mic2_y = source1_y + source_xy_diff - 0.1
        mic1_y = source1_y + 0.1

    source2_z = random.choice(source_z_options)

    mic1_z = source1_z - 0.1
    mic2_z = source2_z - 0.1

    source_1_kwargs = {
        "position": [source1_x, source1_y, source1_z],
        "signal": audio1,
        "delay": 0,
    }

    source_2_kwargs = {
        "position": [source2_x, source2_y, source2_z],
        "signal": audio2,
        "delay": 0,
    }

    mic_1_position = [mic1_x, mic1_y, mic1_z]
    mic_2_position = [mic2_x, mic2_y, mic2_z]

    # Print all randomized parameters
    # print("Randomized Parameters:")
    # print(f"Temperature: {temperature_val}")
    # print(f"Humidity: {humidity_val}")
    # print(f"Room dimensions (x, y, z): ({x}, {y}, {z})")
    # print(f"Number of steps (x, y): ({num_steps_source_x}, {num_steps_source_y})")
    # print(f"Source 1 position: ({source1_x}, {source1_y}, {source1_z})")
    # print(f"Source 2 position: ({source2_x}, {source2_y}, {source2_z})")
    # print(f"Microphone 1 position: ({mic1_x}, {mic1_y}, {mic1_z})")
    # print(f"Microphone 2 position: ({mic2_x}, {mic2_y}, {mic2_z})")

    # ==========================================
    # Simulate first source and first microphone
    # ==========================================
    room_1 = pra.ShoeBox(**room_kwargs)
    room_1.add_source(**source_1_kwargs)
    mic_locs_1 = np.c_[mic_1_position]
    room_1.add_microphone_array(mic_locs_1)
    room_1.simulate(recompute_rir=True)
    room_1.mic_array.to_wav(audio_out_single_1, norm=True, bitdepth=np.int16)
    out, out_rate = sf.read(audio_out_single_1)
    sf.write(audio_out_single_1, out[: len(audio1)], out_rate)

    # ============================================
    # Simulate second source and second microphone
    # ============================================
    room_2 = pra.ShoeBox(**room_kwargs)
    room_2.add_source(**source_2_kwargs)
    mic_locs_2 = np.c_[mic_2_position]
    room_2.add_microphone_array(mic_locs_2)
    room_2.simulate(recompute_rir=True)
    room_2.mic_array.to_wav(audio_out_single_2, norm=True, bitdepth=np.int16)
    out, out_rate = sf.read(audio_out_single_2)
    sf.write(audio_out_single_2, out[: len(audio2)], out_rate)

    # ========================================
    # Simulate two sources and two microphones
    # ========================================
    room_3 = pra.ShoeBox(**room_kwargs)
    room_3.add_source(**source_1_kwargs)
    room_3.add_source(**source_2_kwargs)
    mic_locs_3 = np.c_[mic_1_position, mic_2_position]
    room_3.add_microphone_array(mic_locs_3)
    room_3.simulate(recompute_rir=True)

    # Create temp file to save the reverberated audio
    tmp_file = f".tmp/{uuid.uuid4()}.wav"
    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")

    room_3.mic_array.to_wav(tmp_file, norm=True, bitdepth=np.int16)
    stereo_to_mono.stereo_to_mono(tmp_file, audio_out_1, audio_out_2)
    out, out_rate = sf.read(audio_out_1)
    sf.write(audio_out_1, out[: len(audio2)], out_rate)
    out, out_rate = sf.read(audio_out_2)
    sf.write(audio_out_2, out[: len(audio2)], out_rate)

    # Remove the tmp directory and its contents
    os.system("rm -rf .tmp")
