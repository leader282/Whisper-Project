import sys
from pocketsphinx import AudioFile

def transcribe_audio(input_audio_path):
    # Configure the PocketSphinx speech recognizer
    config = {
        'verbose': False,
        'audio_file': input_audio_path,
        'buffer_size': 2048,
        'no_search': False,
        'full_utt': False
    }

    # Initialize the recognizer with the configuration
    audio_file = AudioFile(**config)
    for phrase in audio_file:
        # Do nothing, just process the audio file into phrases
        pass

    # Get the hypothesis object of the recognized speech
    hypothesis = audio_file.hypothesis()
    return hypothesis

def save_transcription_to_file(audio_path, output_txt_path):
    transcription = transcribe_audio(audio_path)
    with open(output_txt_path, 'w') as f:
        f.write(transcription)

INPUT_AUDIO_PATH = sys.argv[1]
file_name = INPUT_AUDIO_PATH.split('/')[-1].split('.')[0]
OUTPUT_TRANSCRIPT_PATH = 'output_4/'+file_name+'_transcript.txt'

# Transcribe the audio and save the transcription
save_transcription_to_file(INPUT_AUDIO_PATH, OUTPUT_TRANSCRIPT_PATH)