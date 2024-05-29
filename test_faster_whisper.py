import jiwer
from faster_whisper import WhisperModel
import time
from mutagen.mp3 import MP3, HeaderNotFoundError
from mutagen.wave import WAVE
import torch
import os
import sys

# Check that there are 3 arguments
if len(sys.argv) != 4:
    print("Error: Exactly three arguments are required.")
    sys.exit(1)

# Check that each argument is a file that exists
for arg in sys.argv[2:]:
    if not os.path.isfile(arg):
        print(f"Error: File '{arg}' does not exist.")
        sys.exit(1)

#User parameters to aid in testing
model_size = sys.argv[1]
audio_file = sys.argv[2]
referenceion_file = sys.argv[3]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use mutagen to get the length of the audio file and free object after finishing
try:
    audio = MP3(audio_file)
    audio_length = audio.info.length
    del audio
except HeaderNotFoundError:
    audio = WAVE(audio_file)
    audio_length = audio.info.length
    del audio

# Load the model and measure the time taken
loading_start = time.process_time()
model = WhisperModel(model_size,device = device,compute_type = "int8")
loading_time = time.process_time() - loading_start

# Transcribe the audio file and measure the time taken
transcribing_start = time.process_time()
segments,info = model.transcribe(audio_file)
transcribing_time = time.process_time() - transcribing_start
hypothesis = ""
for segment in segments:
    hypothesis += segment.text

# Save the output to output.txt for evaluation
with open('output.txt','w') as file:
    file.write(hypothesis)

# Read reference transcript from files
with open(referenceion_file, 'r') as file:
    reference = file.read()

# Print the resulting statistics from the test
print("Model: Whisper-"+model_size+ "Audio Length:", audio_length ,"Model load time:", loading_time ,"Transcription time:",transcribing_time,"WER:", jiwer.wer(reference, hypothesis))