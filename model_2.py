from pyannote.audio.pipelines import SpeakerDiarization
import whisper
import sys

print("Imports Done")

# Load Whisper ASR model
whisper_model_size = "small.en"  # Choose the appropriate model size (e.g., "large", "medium", "small")
model = whisper.load_model(whisper_model_size)

print("Whisper Model Loaded")

# Load PyAnnote Speaker Diarization model
pyannote_model_name = "pyannote/speaker-diarization-3.0"  # Choose the appropriate PyAnnote diarization model
diarization = SpeakerDiarization.from_pretrained(pyannote_model_name, use_auth_token="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")

print("PyAnnote Model Loaded")

# Specify the input and output audio file paths
INPUT_AUDIO_PATH = sys.argv[1]
file_name = INPUT_AUDIO_PATH.split('/')[-1].split('.')[0]
OUTPUT_TRANSCRIPT_PATH = 'output_2/'+file_name+'_transcript.txt'

# Transcribe the input audio
transcription = model.transcribe(INPUT_AUDIO_PATH)

print("Transcription Done")

# Perform speaker diarization
diarization_result = diarization({'audio': INPUT_AUDIO_PATH})

print("Diarization Done")

# Extract speaker segments and labels

last_transcript = None

with open(OUTPUT_TRANSCRIPT_PATH, 'w') as transcript_file:
    for seg in transcription['segments']:
        start_time = seg['start']
        end_time = seg['end']
        segment_transcript = seg['text']
        
        # Initialize speaker_id as None, in case there's no matching diarization segment
        speaker_id = None
        
        for segment, track in diarization_result.itertracks():
            if not (segment.start > end_time or start_time > segment.end):
                speaker_id = track
                break  # Break the inner loop once a matching diarization segment is found
        
        if speaker_id is not None:
            start_time_str = '{:02}:{:02}:{:02}'.format(int(start_time // 3600), int((start_time % 3600) // 60), int(start_time % 60))
            transcript_file.write(f"SPEAKER {speaker_id+1} {start_time_str}\n{segment_transcript}\n")

print(f"Transcript written to {OUTPUT_TRANSCRIPT_PATH}")