from pydub import AudioSegment


def stereo_to_mono(input_file, output_file_1, output_file_2):
    # Load the stereo file using pydub
    audio = AudioSegment.from_wav(input_file)

    # Check if the file is stereo
    if audio.channels != 2:
        print(f"Skipping {input_file}: Not a stereo file.")
        return

    # Split stereo audio into left and right channels
    left_channel = audio.split_to_mono()[0]
    right_channel = audio.split_to_mono()[1]

    # Export left and right channels as mono WAV files
    left_channel.export(output_file_1, format="wav")
    right_channel.export(output_file_2, format="wav")


# Example usage:
# stereo_to_mono("input_stereo_file.wav", "left_channel_mono.wav", "right_channel_mono.wav")
