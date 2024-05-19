# Whisper Testing

This directory contains scripts for testing the performance of the Whisper and Faster Whisper model.

## Files

- `test_whisper.py`: Python test file for base whisper. It takes three command line arguments: the model size, the audio file, and the transcription file. 

- `test_faster_whisper.py`: Python test file for faster-whisper. It takes three command line arguments: the model size, the audio file, and the transcription file.

- `test_whisperx.py`: Python test file for base whisper. It takes three command line arguments: the model size, the audio file, and the transcription file. 

- `test_whisper.sh`: Runs multiple base whisper model sizes from tiny - small 

- `test_faster_whisper.sh`: Runs multiple faster-whisper model sizes from tiny - small 

- `test_whisperx.sh`: Runs multiple base whisper model sizes from tiny - small 


## Usage

To run the testing script, use the following command:

```bash
python test_whisper.py <model_type> <audio_file> <transcription_file>

python test_faster_whisper.py <model_type> <audio_file> <transcription_file>

python test_whisperx.py <model_type> <audio_file> <transcription_file>

```

## Sample Usage

python3 test_whisper.py tiny.en sample_audio/jfk.wav sample_transcripts/jfk-transcript.txt

python3 test_faster_whisper.py tiny.en sample_audio/jfk.wav sample_transcripts/jfk-transcript.txt

python3 test_whisperx.py tiny.en sample_audio/jfk.wav sample_transcripts/jfk-transcript.txt
