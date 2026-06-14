---
name: run-powsm
description: Run the POWSM ASR or G2P inference script through the project virtualenv. Use when the user wants to transcribe a WAV file (ASR) or convert a transcript to phonemes (G2P), or says "run ASR", "run G2P", "transcribe this audio", or "/run-powsm".
---

Run a POWSM inference script using the project's virtualenv at `powsm/`.

`$ARGUMENTS` selects the task: `asr` (default) or `g2p`, optionally followed by a WAV filename for ASR.

## Steps

1. Confirm the venv exists: `ls powsm/bin/activate`. If missing, tell the user to create it (`python3 -m venv powsm && source powsm/bin/activate && pip install -r requirements.txt`) and stop.

2. **ASR** (`asr [file.wav]`): run
   ```bash
   source powsm/bin/activate && python ASR.py --file "<file.wav>"
   ```
   - Default file is the one already in the repo if none given (see `QUICKSTART.md`).
   - Input may be any sample rate / length; the script resamples to 16 kHz and trims/pads to 20 s automatically. First run downloads the `espnet/powsm` model.

3. **G2P** (`g2p`): `G2P.py` has a hardcoded `prompt` (the ASR transcript) and a hardcoded audio path. If the user wants different input, edit the `prompt` variable and the `audio_path` in `G2P.py` first, then run:
   ```bash
   source powsm/bin/activate && python G2P.py
   ```
   Output also lands in `g2p_output.txt`.

4. Report the console output (transcript or phonemes) back to the user. Note model loading is slow on CPU; it uses CUDA if available.
