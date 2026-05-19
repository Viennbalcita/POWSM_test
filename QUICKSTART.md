# POWSM Quickstart Guide

## Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Input
- Input audio must be a `.wav` file (any sample rate, any length).
- The scripts will automatically resample to 16kHz and trim/pad to 20 seconds.

## Running ASR or G2P

### 1. ASR (Automatic Speech Recognition)
- By default, `ASR.py` expects an audio file in the same directory.
- To run:
  ```bash
  python ASR.py --file "your_audio.wav"
  ```
- If you need to change the input file name, edit the `--file` argument or update the script variable if hardcoded.

### 2. G2P (Grapheme-to-Phoneme)
- By default, `G2P.py` uses a hardcoded transcript prompt.
- To run:
  ```bash
  python G2P.py
  ```
- If you want to use a different transcript, edit the `prompt` variable in `G2P.py`.

## Output
- Results and logs will be printed to the console.
- G2P output is also saved to `g2p_output.txt`.

## Notes
- No manual resampling or trimming needed—just provide a WAV file.
- For advanced options, review script arguments or comments in the code.
