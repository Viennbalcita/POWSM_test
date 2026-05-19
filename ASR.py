from espnet2.bin.s2t_inference import Speech2Text
import soundfile as sf
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import math
import inspect
from typing import Optional

# Optional high-quality resampler
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    librosa = None
    _HAS_LIBROSA = False


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        print(f"Resampler: original sample rate {orig_sr} Hz equals target {target_sr} Hz — no resampling needed")
        return audio

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Prefer librosa if available for higher-quality resampling
    if _HAS_LIBROSA:
        try:
            print(f"Resampler: using librosa to resample {orig_sr}Hz -> {target_sr}Hz")
            return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        except Exception as e:
            print(f"Resampler: librosa failed ({e}), falling back to PyTorch linear resampler")

    # Fallback: linear interpolation via PyTorch
    print(f"Resampler: using PyTorch linear interpolation to resample {orig_sr}Hz -> {target_sr}Hz (fallback)")
    x = torch.from_numpy(audio.astype(np.float32))
    x = x.unsqueeze(0).unsqueeze(0)  # shape (1,1,T)
    new_len = int(round(x.shape[-1] * target_sr / orig_sr))
    x_res = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    y = x_res.squeeze().cpu().numpy()
    return y


def trim_silence(audio: np.ndarray, threshold: float = 0.01, min_keep: int = 1600) -> np.ndarray:
    if audio.size == 0:
        return audio

    abs_audio = np.abs(audio)
    non_silent = np.where(abs_audio >= threshold)[0]
    if non_silent.size == 0:
        return audio

    start = max(int(non_silent[0]) - min_keep, 0)
    end = min(int(non_silent[-1]) + min_keep + 1, audio.shape[0])
    trimmed = audio[start:end]
    if trimmed.shape[0] != audio.shape[0]:
        import logging
        logging.info(f"Trimmed leading/trailing silence from {audio.shape[0]} to {trimmed.shape[0]} samples")
    return trimmed


def apply_gain(audio: np.ndarray, target_rms: float = 0.08, max_peak: float = 0.95) -> np.ndarray:
    if audio.size == 0:
        return audio

    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms <= 1e-9:
        return audio

    gain = target_rms / rms
    boosted = audio * gain
    peak = float(np.max(np.abs(boosted)))
    if peak > max_peak and peak > 0:
        boosted = boosted * (max_peak / peak)
        import logging
        logging.info(f"Applied conservative gain and limited peak to {max_peak:.2f}")
    elif gain != 1.0:
        import logging
        logging.info(f"Applied RMS gain factor {gain:.2f}")

    return boosted


def estimate_noise_rms(audio: np.ndarray, sr: int, window_sec: float = 0.5):
    if audio.size == 0:
        return 0.0, 0
    win = int(round(window_sec * sr))
    if win <= 0:
        return 0.0, 0

    # Compute RMS over sliding windows and pick the minimum-energy window as noise
    # Use non-overlapping windows for speed
    n_windows = max(1, audio.shape[0] // win)
    rms_vals = []
    starts = []
    for i in range(n_windows):
        s = i * win
        e = min(s + win, audio.shape[0])
        chunk = audio[s:e]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        rms_vals.append(rms)
        starts.append(s)

    if not rms_vals:
        return 0.0, 0

    idx = int(np.argmin(rms_vals))
    return float(rms_vals[idx]), int(starts[idx])


def apply_noise_gate(audio: np.ndarray, noise_rms: float, mult: float = 3.0, attenuation: float = 0.1) -> np.ndarray:
    if audio.size == 0:
        return audio
    thresh = noise_rms * mult
    mask = np.abs(audio) < thresh
    if np.any(mask):
        audio = audio.copy()
        audio[mask] = audio[mask] * attenuation
        import logging
        logging.info(f"Applied noise gate (threshold={thresh:.5f}, attenuation={attenuation})")
    return audio


# Refactored for readability

def prepare_audio_minimal(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple:
    """Minimal preprocessing matching demo notebook: convert to mono, pass raw (NO resampling)."""
    # Basic validation
    if audio is None or len(audio) == 0:
        raise ValueError("Audio array is empty")

    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        raise ValueError("Audio contains NaN or Inf values")

    # Mix to mono (same as notebook)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        print(f"[RAW MODE] Converted multi-channel audio to mono")

    print(f"[RAW MODE] Audio shape: {audio.shape}, sample rate: {sample_rate} Hz")
    print(f"[RAW MODE] Duration: {len(audio) / sample_rate:.2f}s")
    
    return audio, sample_rate


def prepare_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: int = 16000,
    duration_sec: float = 20.0,
    normalize: bool = False,
    trim_silence_enabled: bool = True,
    gain_boost_enabled: bool = True,
    noise_gate_enabled: bool = False,
    noise_gate_mult: float = 3.0,
    noise_gate_atten: float = 0.1,
    pad_mode: str = "silence",  # 'silence' or 'repeat'
):
    # Basic validation
    if audio is None or len(audio) == 0:
        raise ValueError("Audio array is empty")

    if sample_rate is None or sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")

    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        raise ValueError("Audio contains NaN or Inf values")

    # Mix to mono (preserves info from all channels)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample using librosa when available, otherwise fallback
    if sample_rate != target_sr:
        audio = resample_audio(audio, sample_rate, target_sr)
        sample_rate = target_sr

    # Trim obvious silence before loudness adjustment and padding
    if trim_silence_enabled:
        audio = trim_silence(audio)

    # Optional normalization
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

    # Conservative RMS boost to make quiet speech more usable
    if gain_boost_enabled:
        audio = apply_gain(audio)

    # Estimate noise from quietest 0.5s window and optionally apply noise gate
    try:
        noise_rms, noise_start = estimate_noise_rms(audio, sample_rate, window_sec=0.5)
        print(f"Estimated noise RMS: {noise_rms:.6f} (window start {noise_start})")
        # compute SNR using signal RMS excluding the noise window
        s = noise_start
        e = min(s + int(0.5 * sample_rate), audio.shape[0])
        if audio.shape[0] - (e - s) > 0:
            signal_parts = np.concatenate([audio[:s], audio[e:]]) if s > 0 else audio[e:]
            signal_rms = float(np.sqrt(np.mean(np.square(signal_parts)))) if signal_parts.size > 0 else 0.0
        else:
            signal_rms = float(np.sqrt(np.mean(np.square(audio))))
        if noise_rms > 0 and signal_rms > 0:
            snr = 20 * math.log10(signal_rms / noise_rms)
            print(f"Estimated SNR (quietest-0.5s method): {snr:.1f} dB")
    except Exception as e:
        import logging
        logging.error(f"Exception occurred in noise estimation: {e}")
        noise_rms = 0.0
        raise

    if noise_gate_enabled and noise_rms > 0:
        audio = apply_noise_gate(audio, noise_rms, mult=noise_gate_mult, attenuation=noise_gate_atten)

    # Amplitude and quality checks
    max_amplitude = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if max_amplitude < 1e-4:
        print("Warning: Audio appears to be silent or very quiet")
    elif max_amplitude > 1.0:
        print(f"Warning: Audio amplitude exceeds 1.0 (max: {max_amplitude:.3f})")

    # SNR estimation (assume first 0.5s is silence/noise)
    try:
        noise_len = min(int(0.5 * target_sr), audio.shape[0])
        noise = audio[:noise_len]
        noise_std = float(np.std(noise)) if noise_len > 0 else 0.0
        signal_std = float(np.std(audio)) if audio.size > 0 else 0.0
        snr = 20 * math.log10((signal_std / (noise_std + 1e-9)) + 1e-9)
        print(f"Estimated SNR: {snr:.1f} dB")
    except Exception:
        pass

    # Clipping and silence percentage
    try:
        clipped = int(np.sum(np.abs(audio) >= 0.99))
        pct_clipped = 100.0 * clipped / audio.shape[0]
        silent = int(np.sum(np.abs(audio) < 0.01))
        pct_silent = 100.0 * silent / audio.shape[0]
        print(f"Clipped samples: {clipped} ({pct_clipped:.2f}%)")
        print(f"Silent samples: {pct_silent:.2f}%")
    except Exception:
        pass

    # Pad or truncate to fixed duration required by POWSM
    target_samples = int(target_sr * duration_sec)
    if audio.shape[0] < target_samples:
        pad_len = target_samples - audio.shape[0]
        if pad_mode == "repeat" and audio.shape[0] > 0:
            repeats = (target_samples // audio.shape[0]) + 1
            audio = np.tile(audio, repeats)[:target_samples]
            import logging
            logging.info(f"Audio repeated to reach {duration_sec}s (pad_mode=repeat)")
        else:
            audio = np.pad(audio, (0, pad_len), mode="constant", constant_values=0)
            import logging
            logging.info(f"Audio padded with {pad_len} samples ({pad_len/target_sr:.2f}s) to reach {duration_sec}s")
    elif audio.shape[0] > target_samples:
        audio = audio[:target_samples]

    return audio, sample_rate

task = "<asr>"
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "espnet/powsm"


def build_pretrained_kwargs(device: str, lang_sym: str, task_sym: str):
    return {
        "device": device,
        "lang_sym": lang_sym,
        "task_sym": task_sym,
        # Notebook decoding parameters for better consistency
        "minlenratio": 0.0,
        "maxlenratio": 0.0,
        "ctc_weight": 0.3,
        "beam_size": 10,
        "batch_size": 0,
        "nbest": 1,
    }


def get_model(device: str = None, lang_sym: str = "<eng>", task_sym: str = "<asr>", model_id: str = MODEL_ID):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")
    print(f"Model id: {model_id}")
    pretrained_kwargs = build_pretrained_kwargs(device, lang_sym, task_sym)

    supported_params = set(inspect.signature(Speech2Text.from_pretrained).parameters)
    filtered_kwargs = {key: value for key, value in pretrained_kwargs.items() if key in supported_params}

    try:
        s2t = Speech2Text.from_pretrained(model_id, **filtered_kwargs)
    except Exception as e:
        err = str(e)
        if "Repo id must be" in err or "repo id" in err:
            base_id = resolve_language_detector_model_id(model_id)
            if base_id != model_id:
                print(f"Warning: model id '{model_id}' rejected; retrying with base id '{base_id}'")
                try:
                    s2t = Speech2Text.from_pretrained(base_id, **filtered_kwargs)
                except Exception as e2:
                    print(f"Retry with base id failed: {e2}")
                    raise
        else:
            raise
    
    # CRITICAL: Set language and task symbols AFTER loading
    # These are not supported by from_pretrained(), must be set on the instance
    for attr, val in [("lang_sym", lang_sym), ("task_sym", task_sym)]:
        if hasattr(s2t, attr):
            try:
                setattr(s2t, attr, val)
                print(f"Set model.{attr} = {val}")
            except Exception as e:
                print(f"Warning: could not set {attr}: {e}")
    
    return s2t


def resolve_language_detector_model_id(model_id: str) -> str:
    parts = model_id.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return model_id


def maybe_detect_language(audio: np.ndarray, device: str, lang_sym: str, task_sym: str, model_id: str) -> str:
    try:
        from espnet2.bin.s2t_inference_language import Speech2Language
    except Exception as e:
        print(f"Language detection unavailable: {e}")
        return lang_sym

    try:
        print("Attempting best-effort language detection...")
        detector_model_id = resolve_language_detector_model_id(model_id)
        if detector_model_id != model_id:
            print(f"Language detector model id: {detector_model_id} (derived from {model_id})")
        detect_kwargs = {"device": device}
        if "nbest" in inspect.signature(Speech2Language.from_pretrained).parameters:
            detect_kwargs["nbest"] = 1
        s2lang = Speech2Language.from_pretrained(detector_model_id, **detect_kwargs)
        detected = s2lang(audio)
        if detected and detected[0]:
            detected_lang = detected[0][0]
            print(f"Detected language symbol: {detected_lang}")
            return detected_lang
    except Exception as e:
        print(f"Language detection failed: {e}")

    return lang_sym

def diagnose_audio_and_model(audio_path_str: str, model_id: str, lang_sym: str):
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        return

    try:
        speech, rate = sf.read(audio_path)
    except Exception as e:
        print(f"❌ Failed to read audio: {e}")
        return

    print(f"✓ File readable: {audio_path.name}")
    print(f"  Sample rate: {rate} Hz")
    print(f"  Duration: {len(speech) / rate:.2f} seconds")
    print(f"  Shape: {speech.shape}")
    print(f"  Dtype: {speech.dtype}")
    print(f"  Range: [{float(np.min(speech)):.6f}, {float(np.max(speech)):.6f}]")

    # Resample check
    if rate != 16000 and _HAS_LIBROSA:
        try:
            speech_16k = librosa.resample(speech, orig_sr=rate, target_sr=16000)
            print(f"✓ Resampled from {rate} Hz to 16000 Hz (librosa)")
        except Exception as e:
            print(f"Resample failed: {e}")
            return
    elif rate != 16000:
        print(f"Info: rate {rate} != 16000 but librosa not available; will use fallback in prepare_audio")
        speech_16k = speech
    else:
        speech_16k = speech

    # Basic quality checks
    try:
        noise = speech_16k[:int(0.5 * 16000)]
        noise_std = np.std(noise) if noise.size > 0 else 0.0
        signal_std = np.std(speech_16k) if speech_16k.size > 0 else 0.0
        snr = 20 * math.log10((signal_std / (noise_std + 1e-9)) + 1e-9)
        print(f"Estimated SNR: {snr:.1f} dB")
    except Exception:
        pass

    clipped = int(np.sum(np.abs(speech_16k) >= 0.99))
    pct_clipped = 100.0 * clipped / speech_16k.shape[0]
    silent = int(np.sum(np.abs(speech_16k) < 0.01))
    pct_silent = 100.0 * silent / speech_16k.shape[0]
    print(f"Clipped samples: {clipped} ({pct_clipped:.2f}%)")
    print(f"Silent samples: {pct_silent:.2f}%")

    # Pad/truncate preview
    target_samples = 16000 * 20
    print(f"Target samples (20s @16k): {target_samples}")
    if speech_16k.shape[0] < target_samples:
        print(f"Audio ({speech_16k.shape[0]}) < target ({target_samples}) — will pad with silence by default")
    elif speech_16k.shape[0] > target_samples:
        print(f"Audio will be truncated to {target_samples} samples (20s)")

    active_lang_sym = maybe_detect_language(np.asarray(speech_16k), device=device, lang_sym=lang_sym, task_sym=task, model_id=model_id)
    if active_lang_sym != lang_sym:
        print(f"Using detected language symbol: {active_lang_sym}")

    # Model load check
    print("\nAttempting to load POWSM model (this may download ~3GB)...")
    try:
        s2t = get_model(device=device, lang_sym=active_lang_sym, task_sym=task, model_id=model_id)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Run a short inference if possible (may be slow)
    try:
        speech_final, _ = prepare_audio(
            np.asarray(speech),
            rate,
            target_sr=16000,
            duration_sec=20.0,
            normalize=False,
            trim_silence_enabled=True,
            gain_boost_enabled=True,
            noise_gate_enabled=False,
            noise_gate_mult=3.0,
            noise_gate_atten=0.1,
            pad_mode="silence",
        )
        raw_pred = s2t(speech_final, text_prev="<na>")[0][0]
        print(f"Inference raw output length: {len(raw_pred)}")
        print(repr(raw_pred[:300]))
    except Exception as e:
        print(f"Inference failed: {e}")


def main(args):
    audio_path = Path(__file__).resolve().with_name(args.file)

    if args.diagnose:
        diagnose_audio_and_model(audio_path, args.model_id, args.lang_sym)
        return

    print(f"Using device: {device}")
    print(f"Audio path: {audio_path.name}")

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        speech, rate = sf.read(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file: {e}") from e

    print(f"Original sample rate: {rate}")
    print(f"Original samples: {len(speech)}")

    if args.raw:
        print("\n[RAW MODE ENABLED] Mirroring demo notebook: mono conversion only, NO resampling/padding/trimming/gain/normalization")
        speech, rate = prepare_audio_minimal(
            np.asarray(speech),
            rate,
        )
    else:
        speech, rate = prepare_audio(
            np.asarray(speech),
            rate,
            target_sr=16000,
            duration_sec=20.0,
            normalize=args.normalize,
            trim_silence_enabled=args.trim_silence,
            gain_boost_enabled=args.gain_boost,
            noise_gate_enabled=args.noise_gate,
            noise_gate_mult=args.noise_gate_mult,
            noise_gate_atten=args.noise_gate_atten,
            pad_mode=args.pad_mode,
        )

    print(f"Preprocessed sample rate: {rate}")
    print(f"Preprocessed samples: {len(speech)}")

    active_lang_sym = args.lang_sym
    if args.detect_language:
        active_lang_sym = maybe_detect_language(np.asarray(speech), device=device, lang_sym=args.lang_sym, task_sym=task, model_id=args.model_id)
        print(f"Using language symbol: {active_lang_sym}")

    # Ensure float32
    speech = speech.astype(np.float32)
    
    # Skip normalization in --raw mode (matches notebook behavior)
    if not args.raw:
        max_val = float(np.max(np.abs(speech))) if speech.size else 0.0
        if max_val > 1.0 and not args.normalize:
            # Only normalize here if not already normalized in prepare_audio
            speech = speech / max_val
            print(f"Normalised audio (peak was {max_val:.4f})")
    else:
        print("[RAW MODE] Skipping post-inference normalization")

    rms = float(np.sqrt(np.mean(np.square(speech)))) if speech.size else 0.0
    print(f"Audio RMS after normalization: {rms:.6f}")
    if rms < 1e-4:
        raise ValueError("Audio appears silent — check source file and channel mix")

    print(f"Audio peak: {np.abs(speech).max()}, First 200 samples: {speech[:200]}")

    # load model and run inference
    s2t = get_model(device=device, lang_sym=active_lang_sym, task_sym=task, model_id=args.model_id)

    # Try several decode variants to avoid prompt-related early termination
    import re
    
    if args.raw:
        # Notebook approach: single call with NO text_prev
        print("\n[RAW MODE] Single inference with NO text_prev (matching notebook)")
        try:
            raw_pred = s2t(speech)[0][0]
            print(f"Raw output: {repr(raw_pred[:200])}")
            processed = re.sub(r"<[^>]+>", "", raw_pred).strip()
            print(f"Processed output: '{processed}'")
        except Exception as e:
            print(f"Inference failed: {e}")
            processed = ""
    else:
        # ASR.py approach: try multiple text_prev variants
        attempts = [
            (None, "no text_prev (default)"),
            ("", "empty string text_prev"),
            ("<na>", "'<na>' text_prev"),
        ]

        found = False
        processed = None
        for tp, desc in attempts:
            try:
                if tp is None:
                    raw_pred = s2t(speech)[0][0]
                else:
                    raw_pred = s2t(speech, text_prev=tp)[0][0]
            except Exception as e:
                print(f"Decode attempt ({desc}) failed: {e}")
                continue

            print(f"Decode attempt ({desc}) raw: {repr(raw_pred[:200])}")
            # remove special tokens like <eng>, <asr>, <notimestamps>
            processed = re.sub(r"<[^>]+>", "", raw_pred).strip()
            print(f"Processed prediction ({desc}): '{processed}'")
            if processed:
                found = True
                break

        if not found:
            print("Warning: all decode attempts returned no transcript. Consider trying another model variant or check audio quality.")
            processed = ""
    
    # Save transcript to file with fixed name
    output_file = Path(__file__).resolve().parent / "asr_transcript.txt"
    with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
        f.write(processed if processed else "")
    
    print(f"\n{'='*80}")
    print("✅ ASR Processing Complete")
    print(f"{'='*80}")
    print(f"Output file: asr_transcript.txt")
    print(f"Full path: {output_file.resolve()}")
    print(f"Transcript length: {len(processed) if processed else 0} characters")
    if processed:
        print(f"Preview: {processed[:100]}...")
    else:
        print("Preview: (empty - no transcript extracted)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run POWSM ASR with preprocessing and diagnostics")
    parser.add_argument("--file", default="input 4.wav", help="Audio filename in the script directory")
    parser.add_argument("--diagnose", action="store_true", help="Run diagnostics instead of full inference")
    parser.add_argument("--raw", action="store_true", help="Skip all preprocessing (minimal: resample + pad/truncate only)")
    parser.add_argument("--normalize", action="store_true", help="Normalize audio before inference")
    parser.add_argument("--no-trim-silence", dest="trim_silence", action="store_false", help="Disable trimming leading/trailing silence")
    parser.add_argument("--no-gain-boost", dest="gain_boost", action="store_false", help="Disable conservative RMS gain boosting")
    parser.add_argument("--pad-mode", dest="pad_mode", choices=["silence", "repeat"], default="silence", help="How to pad short audio")
    parser.add_argument("--lang-sym", dest="lang_sym", default="<eng>", help="Language symbol for model")
    parser.add_argument("--detect-language", dest="detect_language", action="store_true", help="Attempt best-effort language detection before inference")
    parser.add_argument("--no-detect-language", dest="detect_language", action="store_false", help="Disable automatic language detection")
    parser.add_argument("--model-id", default=MODEL_ID, choices=["espnet/powsm", "espnet/powsm_ctc", "espnet/powsm/textnorm_retrained"], help="POWSM model variant to load")
    parser.add_argument("--noise-gate", dest="noise_gate", action="store_true", help="Enable simple noise-gate preprocessing")
    parser.add_argument("--noise-gate-mult", dest="noise_gate_mult", type=float, default=3.0, help="Noise gate threshold multiplier of estimated noise RMS")
    parser.add_argument("--noise-gate-atten", dest="noise_gate_atten", type=float, default=0.1, help="Noise gate attenuation factor for samples below threshold")
    parser.set_defaults(trim_silence=True, gain_boost=True, detect_language=True)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
