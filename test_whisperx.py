import textdistance as td
import whisperx
import gc 
import time
from mutagen.mp3 import MP3, HeaderNotFoundError
from mutagen.wave import WAVE
import os
import sys
import torch

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
batch_size = 4
compute_type = "int8"

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
model = whisperx.load_model(model_size, device, compute_type=compute_type)
loading_time = time.process_time() - loading_start

# save model to local path (optional)
# model_dir = "whisperx_cache/"+model_size
# model = whisperx.load_model(model_size, device, compute_type=compute_type, download_root=model_dir)

# Transcribe the audio file and measure the time taken
transcribing_start = time.process_time()
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
transcribing_time = time.process_time() - transcribing_start
predicted = result["segments"]

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