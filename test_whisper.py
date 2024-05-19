import textdistance as td
import whisper
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
transcription_file = sys.argv[3]

#check cuda availability using torch and set device accordingly
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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

if device == "cuda":
    model = whisper.load_model(model_size, device="cuda")
else:
    model = whisper.load_model(model_size)
    
loading_time = time.process_time() - loading_start

# Transcribe the audio file and measure the time taken
transcribing_start = time.process_time()
result = model.transcribe(audio_file, fp16=False)
transcribing_time = time.process_time() - transcribing_start
predicted = result["text"]

# Save the output to output.txt for evaluation
with open('output.txt','w') as file:
    file.write(predicted)

# Read reference transcript from files
with open(transcription_file, 'r') as file:
    transcript = file.read()

# Compute Damerau-Levenshtein distance using textdistance library
# Damerau Levenshtein distance is a string similarity metric used to measure the minimum number of single-character allowed operations 
# (insertions, deletions, substitutions, transposition)
# required to transform one string into the other.

# The distance is normalized to be in the range [0,1]
dl_similarity = td.damerau_levenshtein.normalized_similarity(transcript,predicted)

# Comute Tversky similarity unsing textdistance library
# Unlike Damerau-Levenshtein this does not account for order of strings
# Tversky similarity is a string similarity metric used to measure the similarity between two strings.
# It is a variation of the Jaccard index that allows to control the importance of false positives and false negatives.
# We are using alpha=0.2 and beta=0.8 to give more importance to false negatives (Word omissions).
tversky = td.Tversky(ks=(0.2, 0.8))
Tversky = tversky(transcript,predicted)

# Print the resulting statistics from the test
print("Model: Whisper-"+model_size+ "Audio Length:", audio_length ,"Model load time:", loading_time ,"Transcription time:",transcribing_time,"Damerau-Levenshtein similarity:", dl_similarity, "Tversky similarity:", Tversky)