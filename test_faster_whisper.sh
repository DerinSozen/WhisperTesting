#!/bin/bash  

# Edit audio file and Transcript for all test cases here
AUDIO_FILE="sample_audio/jfk.wav"
TRANSCRIPT_FILE="sample_transcripts/jfk-transcript.txt"

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

python3 test_faster_whisper.py "tiny.en" $AUDIO_FILE $TRANSCRIPT_FILE
python3 test_faster_whisper.py "tiny" $AUDIO_FILE $TRANSCRIPT_FILE
python3 test_faster_whisper.py "base.en" $AUDIO_FILE $TRANSCRIPT_FILE
python3 test_faster_whisper.py "base" $AUDIO_FILE $TRANSCRIPT_FILE
python3 test_faster_whisper.py "small.en" $AUDIO_FILE $TRANSCRIPT_FILE
ython3 test_faster_whisper.py "small" $AUDIO_FILE $TRANSCRIPT_FILE

deactivate
