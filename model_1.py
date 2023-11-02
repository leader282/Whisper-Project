import whisper
import datetime
import sys
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np

print("\nImports Done")

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu"))

# Specify the input and output audio file paths
INPUT_AUDIO_PATH = sys.argv[1]
file_name = INPUT_AUDIO_PATH.split('/')[-1].split('.')[0]
OUTPUT_TRANSCRIPT_PATH = 'output_1/'+file_name+'_transcript.txt'
num_speakers = int(sys.argv[2])

language = 'English'
model_size = 'small.en'

model = whisper.load_model(model_size)
result = model.transcribe(INPUT_AUDIO_PATH)
segments = result["segments"]

print("Transcription Done")

with contextlib.closing(wave.open(INPUT_AUDIO_PATH,'r')) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames / float(rate)

audio = Audio()

def segment_embedding(segment):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(INPUT_AUDIO_PATH, clip)
  return embedding_model(waveform[None])

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
  embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

print("Embeddings Done")

clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
  segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

print("Clustering Done")

def time(secs):
  return datetime.timedelta(seconds=round(secs))

f = open(OUTPUT_TRANSCRIPT_PATH, "w")
for (i, segment) in enumerate(segments):
  if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
    f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
  f.write(segment["text"][1:] + ' ')
f.close()

print(f"Transcript written to {OUTPUT_TRANSCRIPT_PATH}")