# POWSM Spec & Usage Reference

> Canonical reference for running the POWSM speech model in **this repo**.
> Written for future Claude sessions and the maintainer. Last updated: 2026-06-12.
>
> Where this repo's scripts diverge from intended/upstream behavior, both are shown
> **side by side**. Bug details live in [`QA_REPORT.md`](./QA_REPORT.md).

---

## 1. What POWSM is

**POWSM** (Phonetic Open Whisper-Style Speech Model, `espnet/powsm`) is a phonetic
speech *foundation* model built on the OWSM (Open Whisper-style Speech Model) family
and trained on the IPAPack++ corpus. A single model jointly performs **four
phone-related tasks**:

| Task | Symbol | What it does | `text_prev` (prompt) |
|------|--------|--------------|----------------------|
| Phone Recognition | `<pr>` | audio → IPA phones | `<na>` |
| Automatic Speech Recognition | `<asr>` | audio → text | `<na>` |
| Grapheme-to-Phoneme (audio-guided) | `<g2p>` | audio + text → phones | the ASR **transcript** |
| Phoneme-to-Grapheme (audio-guided) | `<p2g>` | audio + phones → text | phoneme string with `/` delimiters |

- **Architecture:** encoder–decoder, Whisper/OWSM-style. An encoder-only CTC variant,
  **POWSM-CTC** (`espnet/powsm_ctc`), also exists.
- **Phone output format:** slash-delimited IPA, e.g. `/pʰ//ɔ//s//ə//m/`.
- **Paper:** *POWSM: A Phonetic Open Whisper-Style Speech Foundation Model*
  (arXiv:2510.24992).

### Hard input requirements (apply to every task)
- **Sample rate: exactly 16 kHz.**
- **Duration: exactly 20 s** — pad (with silence) or truncate to `16000 * 20 = 320000` samples.
- Mono. Mix down multi-channel before inference.

These are not soft preferences — the model was trained on fixed 16 kHz / 20 s inputs.
Off-spec audio produces garbage.

### Language symbols
`lang_sym` is an **ISO 639-3** code in angle brackets: `<eng>` (English), `<spa>`,
`<jpn>`, … Use `<unk>` for an unseen/unknown language. This repo defaults to `<eng>`.

---

## 2. Canonical upstream usage (ground truth)

This is the official `espnet/powsm` API, independent of this repo's wrapper scripts.
When in doubt, trust this over `ASR.py` / `G2P.py`.

```python
from espnet2.bin.s2t_inference import Speech2Text
import soundfile as sf

task = "<pr>"                      # or <asr> / <g2p> / <p2g>
s2t = Speech2Text.from_pretrained(
    "espnet/powsm",
    device="cuda",                 # or "cpu"
    lang_sym="<eng>",
    task_sym=task,
)

speech, rate = sf.read("sample.wav")   # MUST be 16 kHz, 20 s
prompt = "<na>"                        # <na> for PR/ASR; transcript for G2P; phonemes for P2G
pred = s2t(speech, text_prev=prompt)[0][0]
pred = pred.split("<notimestamps>")[1].strip()
print(pred)
```

Notes:
- `task_sym` / `lang_sym` are accepted by `from_pretrained` in current espnet
  (this repo is on **espnet 202511**). Older code paths set them on the instance after
  loading; both appear in `ASR.py` for safety.
- Output is wrapped in special tokens (`<eng>`, `<asr>`, `<notimestamps>`, …). Strip
  them — upstream splits on `<notimestamps>`; this repo regexes out `<[^>]+>`.
- First load downloads **~3 GB** of weights to the Hugging Face cache.

---

## 3. Environment setup (this repo)

| Thing | Value |
|-------|-------|
| Virtualenv | `powsm/` (gitignored) |
| Python | 3.12 |
| Key deps | `espnet==202511`, `torch`, `soundfile`, `numpy`, `librosa` |
| Device | auto: `"cuda" if torch.cuda.is_available() else "cpu"` |
| Model cache | Hugging Face default (`~/.cache/huggingface`), ~3 GB on first run |

```bash
# Activate the project venv (required — it has espnet/torch/librosa)
source powsm/bin/activate

# (Only if recreating) install deps
pip install -r requirements.txt
```

> `requirements.txt` is unpinned and lists stdlib modules (`pathlib`, `argparse`,
> `logging`) as if they were installable — harmless but ignore those three. The venv
> already has working pinned versions.

---

## 4. ASR — `ASR.py`

Transcribe a WAV file to text.

### Command

```bash
source powsm/bin/activate
python ASR.py --file "your_audio.wav"
```

The file is resolved **relative to the script directory**, not your CWD.

### Behavior: intended vs actual

| Aspect | Intended | Actual (current code) |
|--------|----------|------------------------|
| Input | any-rate WAV, any length | same — librosa resamples to 16 kHz, pads/truncates to 20 s |
| Preprocessing (default) | canonical: mono → resample → pad/truncate to 20 s | ✅ **canonical by default.** trim-silence, gain-boost, noise-gate, normalize are now OFF and opt-in via flags |
| `--raw` mode | canonical-only, minimal logging | ✅ mono → resample → pad/truncate (the missing-resample bug is **fixed**). Honest help text. Behaves like the default but guarantees zero optional DSP |
| `--model-id` choices | 3 selectable variants | only `espnet/powsm` works. `espnet/powsm_ctc` is a *different model class* (CTC); `espnet/powsm/textnorm_retrained` is a malformed 3-part repo id and silently falls back to `espnet/powsm`. See QA `M-03` |
| Output | transcript file | writes `asr_transcript.txt` (transcript only, special tokens stripped) ✅ |

### Useful flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--file NAME` | `download.wav` | input WAV in script dir |
| `--diagnose` | off | inspect audio (SR, SNR, clipping) + test model load, no full run |
| `--raw` | off | canonical-only path (mono→resample→pad), minimal logging |
| `--normalize` | off | opt-in: peak-normalize before inference (non-canonical) |
| `--trim-silence` | off | opt-in: trim leading/trailing silence (non-canonical) |
| `--gain-boost` | off | opt-in: RMS gain boost (non-canonical) |
| `--pad-mode {silence,repeat}` | `silence` | how to fill short audio to 20 s |
| `--lang-sym <eng>` | `<eng>` | ISO 639-3 language token |
| `--detect-language` / `--no-detect-language` | on | best-effort `Speech2Language` autodetect |
| `--noise-gate` (+ `--noise-gate-mult`, `--noise-gate-atten`) | off | simple noise gate |
| `--model-id` | `espnet/powsm` | leave as default (see actual-behavior note) |

### Decoding parameters (baked into `build_pretrained_kwargs`)
`ctc_weight=0.3`, `beam_size=10`, `nbest=1`, `minlenratio=0.0`, `maxlenratio=0.0`.
These mirror the demo notebook for output consistency.

### Output
- `asr_transcript.txt` — plain transcript (special tokens stripped). Overwritten each run.
- Console: device, audio stats, decode attempts, preview.

---

## 5. G2P — `G2P.py`

Audio-guided grapheme-to-phoneme: given audio **and** its transcript, emit IPA phones.

### Command

```bash
source powsm/bin/activate
python G2P.py --file "your_audio.wav"
```

### Behavior: intended vs actual

| Aspect | Intended | Actual (current code) |
|--------|----------|------------------------|
| Transcript source | hand-edited file, decoupled from ASR output | ✅ reads **`g2p_prompt.txt`** (override with `--prompt-file`). No code edits needed. Errors clearly if missing/blank |
| Audio input | any-rate WAV | ✅ has its own resampler → 16 kHz, pad/truncate to 20 s |
| `text_prev` | the transcript guides phonemization | ✅ `s2t(speech, text_prev=prompt)` |
| Output cleanup | strip special tokens + `/` delimiters | ✅ regex strip + remove `/` |

### The ASR → G2P pipeline

G2P reads its transcript from `g2p_prompt.txt` (deliberately **not** auto-coupled to
ASR's output — you paste/edit it by hand). No source edits required:

```bash
source powsm/bin/activate

# 1. Transcribe → writes asr_transcript.txt
python ASR.py --file "your_audio.wav"

# 2. Paste that transcript into g2p_prompt.txt (a plain text file you edit),
#    then run G2P on the SAME audio:
python G2P.py --file "your_audio.wav"
```

- `g2p_prompt.txt` ships seeded with a sample transcript, so G2P runs out of the box.
- Use `--prompt-file other.txt` to point at a different transcript file.
- If the file is missing or blank, G2P exits with a clear "edit g2p_prompt.txt" message.

### Output
- `g2p_output.txt` — formatted: original text + phoneme output. Overwritten each run.
- Console: same content via logging.

---

## 6. Quick reference

```bash
# Setup (once per shell)
source powsm/bin/activate

# ASR: WAV → text  (writes asr_transcript.txt)
python ASR.py --file "audio.wav"

# ASR diagnostics only (no full inference)
python ASR.py --file "audio.wav" --diagnose

# G2P: audio + transcript → IPA phones  (writes g2p_output.txt)
#   edit g2p_prompt.txt with the transcript first (or pass --prompt-file)
python G2P.py --file "audio.wav"
```

| Item | Value |
|------|-------|
| Model id | `espnet/powsm` (CTC variant: `espnet/powsm_ctc`) |
| Loader | `espnet2.bin.s2t_inference.Speech2Text` |
| Input | 16 kHz mono, padded/truncated to 20 s |
| Task tokens | `<pr>` `<asr>` `<g2p>` `<p2g>` |
| Lang token | ISO 639-3 in brackets, e.g. `<eng>`; `<unk>` if unknown |
| Default prompt | `<na>` (PR/ASR); transcript (G2P); `/`-phonemes (P2G) |
| ASR output file | `asr_transcript.txt` |
| G2P input file | `g2p_prompt.txt` (hand-edited; `--prompt-file` to override) |
| G2P output file | `g2p_output.txt` |
| Model download | ~3 GB on first run (HF cache) |

---

## 7. Known issues / gotchas

Full list with line numbers in [`QA_REPORT.md`](./QA_REPORT.md). Status of the
usage-affecting ones:

- ✅ **FIXED — G2P prompt** — `G2P.py` now reads `g2p_prompt.txt` (or `--prompt-file`)
  instead of a hardcoded string; errors clearly if missing/blank. (was `C-02`, `M-04`)
- ✅ **FIXED — `--raw` mode** — now does true canonical preprocessing
  (mono→resample→pad/truncate) with honest help text. (was `H-02`)
- ✅ **CHANGED — default preprocessing** — ASR default is now canonical-only;
  trim/gain/gate/normalize are opt-in flags, off by default.
- **`--model-id` extra choices don't work** — `espnet/powsm/textnorm_retrained` is an
  invalid repo id (silently falls back to `espnet/powsm`); `espnet/powsm_ctc` is a
  different model class. Stick with the default. (`M-03`)
- **First-run file defaults** — both scripts default `--file` to a real file
  (`download.wav`) that ships in the repo; always pass `--file` explicitly for clarity.
- **Best-effort diagnostics may be noisy** — SNR/noise estimation is informational; a
  failure there shouldn't (and mostly doesn't) abort a run.

---

## Sources

- [espnet/powsm · Hugging Face](https://huggingface.co/espnet/powsm)
- [POWSM: A Phonetic Open Whisper-Style Speech Foundation Model (arXiv:2510.24992)](https://arxiv.org/pdf/2510.24992)
- [Using ESPnet at Hugging Face](https://huggingface.co/docs/hub/en/espnet)
- Repo: `ASR.py`, `G2P.py`, `QA_REPORT.md`, local venv (`espnet 202511`, Python 3.12)
