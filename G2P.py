from espnet2.bin.s2t_inference import Speech2Text
import soundfile as sf
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import re
import logging
import argparse

# Optional high-quality resampler
try:
    import librosa

    _HAS_LIBROSA = True
except Exception:
    librosa = None
    _HAS_LIBROSA = False

logging.basicConfig(level=logging.INFO, format="%(message)s")

parser = argparse.ArgumentParser(description="Run POWSM G2P (grapheme-to-phoneme) conversion")
parser.add_argument("--file", default="download.wav", help="Audio filename in the script directory")
parser.add_argument(
    "--prompt-file",
    dest="prompt_file",
    default="g2p_prompt.txt",
    help="Text file (in the script directory) containing the ASR transcript used to guide G2P",
)
args = parser.parse_args()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return audio

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Prefer librosa if available for higher-quality resampling
    if _HAS_LIBROSA:
        try:
            logging.info(f"Resampling {orig_sr}Hz -> {target_sr}Hz (librosa)")
            return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        except Exception as e:
            logging.info(
                f"librosa resample failed ({e}); falling back to PyTorch linear interpolation"
            )

    # Fallback: linear interpolation via PyTorch
    logging.info(f"Resampling {orig_sr}Hz -> {target_sr}Hz (PyTorch linear, fallback)")
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # shape (1,1,T)
    new_len = int(round(x.shape[-1] * target_sr / orig_sr))
    x_res = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return x_res.squeeze().cpu().numpy()


def prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz when needed (POWSM expects 16 kHz input)
    if sample_rate != 16000:
        audio = resample_audio(audio, sample_rate, 16000)

    target_samples = 16000 * 20
    if audio.shape[0] < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.shape[0]))
    elif audio.shape[0] > target_samples:
        audio = audio[:target_samples]

    return audio


def clean_special_tokens(text: str) -> str:
    """Remove special tokens like <notimestamps> from text."""
    return re.sub(r"<[^>]+>", "", text).strip()


task = "<g2p>"
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.info("=" * 80)
logging.info("G2P (Grapheme to Phoneme) Conversion")
logging.info("=" * 80)

# Load model
logging.info(f"\nLoading POWSM G2P model on device: {device}")
s2t = Speech2Text.from_pretrained(
    "espnet/powsm",
    device=device,
    lang_sym="<eng>",
    task_sym=task,
)
logging.info("✓ Model loaded successfully")

# Load audio
audio_path = Path(__file__).resolve().with_name(args.file)
try:
    speech, rate = sf.read(audio_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Audio file not found: {audio_path}")
except Exception as e:
    raise RuntimeError(f"Failed to read audio file: {e}") from e
speech = prepare_audio(np.asarray(speech), rate)

logging.info(f"\n[Input Audio]")
logging.info(f"  File: {audio_path.name}")
logging.info(f"  Sample rate: {rate} Hz")
logging.info(f"  Audio samples: {len(speech)}")

# Get the ASR transcript (prompt for G2P model) from an editable text file.
# Edit g2p_prompt.txt (or pass --prompt-file) to change the transcript — no code edits.
prompt_path = Path(__file__).resolve().with_name(args.prompt_file)
try:
    prompt = prompt_path.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    raise FileNotFoundError(
        f"G2P prompt file not found: {prompt_path}\n"
        f"Create it and paste the ASR transcript inside, then re-run."
    )
if not prompt:
    raise ValueError(
        f"G2P prompt file is empty: {prompt_path}\nPaste the ASR transcript into it, then re-run."
    )

logging.info("\n[ASR Input Text]")
logging.info(f"  Prompt file: {prompt_path.name}")
logging.info(f"  Prompt length: {len(prompt)} characters")
logging.info(f"  Text: {prompt[:100]}...")

# Run G2P inference
logging.info(f"\n[Running G2P Inference]")
try:
    result = s2t(speech, text_prev=prompt)
    if not result or not result[0] or not result[0][0]:
        raise ValueError("Model returned unexpected format or empty result")
    raw_pred = result[0][0]
except Exception as e:
    raise RuntimeError(f"Inference failed: {e}") from e

# Post-processing: strip special tokens and remove slash delimiters
phonemes = clean_special_tokens(raw_pred)
phonemes = phonemes.replace("/", "")

logging.info(f"  ✓ Inference complete")
logging.info(f"  Raw output length: {len(raw_pred)} chars")
logging.info(f"  Phonemes length: {len(phonemes)} chars")

# Save to output file
output_file = Path(__file__).resolve().parent / "g2p_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("G2P (Grapheme to Phoneme) Conversion Output\n")
    f.write("=" * 80 + "\n\n")
    f.write("[Original Text]\n")
    f.write(f"{prompt}\n\n")
    f.write("[Phoneme Output]\n")
    f.write(f"{phonemes}\n\n")
    f.write("=" * 80 + "\n")

logging.info(f"\n✅ Output saved to: {output_file}")
logging.info(f"   File path: {output_file.resolve()}")

# Print results to console
logging.info(f"\n[RESULTS]")
logging.info(f"\nOriginal Text:\n{prompt}\n")
logging.info(f"Phoneme Output:\n{phonemes}")
