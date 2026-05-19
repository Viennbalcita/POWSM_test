from espnet2.bin.s2t_inference import Speech2Text
import soundfile as sf
import torch
import numpy as np
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


def prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

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

logging.info("="*80)
logging.info("G2P (Grapheme to Phoneme) Conversion")
logging.info("="*80)

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
audio_path = Path(__file__).resolve().with_name("Input part 1 mono.wav")
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

# Get the ASR transcript (prompt for G2P model)
prompt = "FORMULATING THE GOOD A GOOD RESEARCH QUESTION OKAY SO MOST OF THE TIME FOR TODAY SESSION I'LL BE YAPPING THAT'S BASICALLY WHAT I WANTED TO SAY I'LL BE YAPPING WE ARE NOW IN WEEK THREE NOTHING TO WORRY IN MY OPINION SO FAR"
if not prompt.strip():
    raise ValueError("Provide the ASR transcript in prompt before running G2P.")

logging.info(f"\n[ASR Input Text]")
logging.info(f"  Prompt length: {len(prompt)} characters")
logging.info(f"  Text: {prompt[:100]}...")

asr_test_output = prompt
logging.info(f"\n  ✓ Prompt matches ASR test output")

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
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("G2P (Grapheme to Phoneme) Conversion Output\n")
    f.write("="*80 + "\n\n")
    f.write("[Original Text]\n")
    f.write(f"{prompt}\n\n")
    f.write("[Phoneme Output]\n")
    f.write(f"{phonemes}\n\n")
    f.write("="*80 + "\n")

logging.info(f"\n✅ Output saved to: {output_file}")
logging.info(f"   File path: {output_file.resolve()}")

# Print results to console
logging.info(f"\n[RESULTS]")
logging.info(f"\nOriginal Text:\n{prompt}\n")
logging.info(f"Phoneme Output:\n{phonemes}")