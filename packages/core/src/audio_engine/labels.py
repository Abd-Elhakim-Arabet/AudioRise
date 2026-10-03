"""Fine-grained instrument labeling (Increment 3).

Best-effort LOCAL heuristic — no model download, numpy only.
Never invents stems: tags the proven Demucs stems and records
`{stem} -> [{label, score}]` in `stems.json`, plus alias symlinks like
`other[flute,synth].wav` (fallback: copy) for DAW convenience.

Honesty note: flute-vs-violin from a finished mix is unsolved;
treat scores as organizing hints, not ground truth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FINE_LABELS = (
    "flute",
    "piano",
    "guitar",
    "violin",
    "trumpet",
    "saxophone",
    "synth",
    "speech",
    "singing",
    "drums",
    "bass",
)

_SILENCE_DB = -60.0
_ALIAS_MIN_SCORE = 0.25


def _load_mono(path: str | Path) -> tuple["object", int]:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True)  # [T, C]
    mono = data.mean(axis=1).astype("float64")
    return mono, int(sr)


def extract_features(path: str | Path) -> dict:
    """Numpy-only spectral + temporal descriptors."""
    import numpy as np

    mono, sr = _load_mono(path)
    n = mono.size
    if n == 0:
        return {"rms_db": -120.0, "silent": True}
    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    rms_db = 20.0 * math.log10(rms + 1e-12)
    peak = float(np.max(np.abs(mono)) + 1e-12)
    if rms_db < _SILENCE_DB:
        return {"rms_db": round(rms_db, 2), "silent": True}

    zcr = float(np.mean(np.abs(np.diff(np.sign(mono)))) / 2.0)

    # Framed magnitude spectra (4096 / 2048 hop).
    flen, hop = 4096, 2048
    frames = []
    for start in range(0, max(n - flen, 1), hop):
        frame = mono[start : start + flen] * np.hanning(min(flen, n - start))
        spec = np.abs(np.fft.rfft(frame, n=flen))
        frames.append(spec)
    spec_mat = np.stack(frames)  # [F, B]
    freqs = np.fft.rfftfreq(flen, 1.0 / sr)
    mag_sum = spec_mat.sum(axis=1, keepdims=True) + 1e-12
    centroid = (spec_mat * freqs).sum(axis=1) / mag_sum[:, 0]
    spread = np.sqrt(((spec_mat * (freqs**2)).sum(axis=1) / mag_sum[:, 0]) - centroid**2 + 1e-12)
    gmean = np.exp(np.log(spec_mat + 1e-12).mean(axis=1))
    flatness = gmean / (spec_mat.mean(axis=1) + 1e-12)
    cumsum = np.cumsum(spec_mat, axis=1)
    rolloff = freqs[np.argmax(cumsum >= 0.85 * mag_sum, axis=1)]

    # Energy envelope onsets (rms per hop frame).
    env = np.array([np.sqrt(np.mean(mono[i : i + hop] ** 2) + 1e-12) for i in range(0, n - hop, hop)])
    thresh = env.max() * 0.25
    onsets = int(np.sum((env[1:] - env[:-1]) > thresh * 0.5))
    dur = n / float(sr)
    onset_rate = onsets / max(dur, 1e-6)
    sustain = float(np.mean(env > env.max() * 0.1))  # high = legato/sustained

    low_band = spec_mat[:, freqs < 250].sum()
    low_ratio = float(low_band / (spec_mat.sum() + 1e-12))
    env_var = float(env.std() / (env.mean() + 1e-12))  # speech has gappy envelope

    return {
        "rms_db": round(rms_db, 2),
        "silent": False,
        "sr": sr,
        "zcr": round(float(zcr), 4),
        "centroid_med": round(float(np.median(centroid)), 1),
        "centroid_std": round(float(centroid.std()), 1),
        "bandwidth_mean": round(float(spread.mean()), 1),
        "flatness_mean": round(float(np.clip(flatness.mean(), 0, 1)), 4),
        "rolloff_med": round(float(np.median(rolloff)), 1),
        "onset_rate": round(float(onset_rate), 2),
        "sustain": round(float(np.clip(sustain, 0, 1)), 3),
        "low_ratio": round(float(np.clip(low_ratio, 0, 1)), 3),
        "env_var": round(float(env_var), 3),
        "crest_db": round(20.0 * math.log10(peak / (rms + 1e-12)), 2),
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_features(f: dict) -> dict[str, float]:
    """Rule-based scores in [0, 1]. Returns {} for silence."""
    if f.get("silent"):
        return {}
    bright = _clamp01(f["centroid_med"] / 6000.0)
    mid_bright = 1.0 - abs(f["centroid_med"] - 1800.0) / 3000.0
    mid_bright = _clamp01(mid_bright)
    tonal = 1.0 - _clamp01(f["flatness_mean"] * 3.0)
    noisy = _clamp01(f["flatness_mean"] * 2.5)
    sustained = _clamp01(f["sustain"])
    percussive = _clamp01(f["onset_rate"] / 6.0) * (1.0 - sustained * 0.6)
    vibrato = _clamp01(f["centroid_std"] / 350.0)
    stable = 1.0 - vibrato
    low = _clamp01(f["low_ratio"] * 2.0)
    zcr = f["zcr"]
    speechy_zcr = _clamp01(1.0 - abs(zcr - 0.12) / 0.12)  # speech ~0.05-0.2
    gappy = _clamp01((f["env_var"] - 0.6) / 1.2)  # speech pauses vs legato music

    scores = {
        # Pure sustained mellow tone, few onsets.
        "flute": tonal * (0.4 + 0.6 * sustained) * (1.0 - bright * 0.6) * (1.0 - percussive) * (0.7 + 0.3 * stable),
        # Sustained + pitch movement.
        "violin": tonal * sustained * (0.35 + 0.65 * vibrato) * (1.0 - percussive) * (0.4 + 0.6 * mid_bright),
        # Bright sustained brass.
        "trumpet": tonal * sustained * (0.3 + 0.7 * bright) * (1.0 - percussive * 0.7),
        "saxophone": tonal * sustained * (0.3 + 0.7 * mid_bright) * (0.5 + 0.5 * vibrato) * (1.0 - percussive * 0.7),
        # Very stable pitched sustain, machine-like.
        "synth": tonal * sustained * stable * (1.0 - percussive) * (0.5 + 0.5 * _clamp01(f["bandwidth_mean"] / 2500.0 + 0.3)),
        # Harmonic onsets with decay.
        "piano": tonal * _clamp01(0.25 + percussive) * (1.0 - sustained * 0.45) * (0.5 + 0.5 * _clamp01(f["crest_db"] / 20.0)),
        "guitar": tonal * _clamp01(0.2 + percussive * 0.9) * (0.4 + 0.6 * mid_bright) * (0.4 + 0.6 * sustained),
        # Noisy + transient.
        "drums": _clamp01(percussive * 0.75 + noisy * 0.55) * (0.5 + 0.5 * _clamp01(f["crest_db"] / 18.0)),
        # Low-end energy, dark.
        "bass": low * (1.0 - bright) * (0.4 + 0.6 * _clamp01(sustained + 0.3)),
        # Gappy envelope + consonant noise.
        "speech": _clamp01(speechy_zcr * 0.6 + gappy * 0.6 + noisy * 0.35) * (1.0 - sustained * 0.55),
        # Sustained vocal-like + vibrato + gaps less than speech.
        "singing": tonal * sustained * (0.3 + 0.7 * vibrato) * (0.4 + 0.6 * speechy_zcr) * (0.5 + 0.5 * mid_bright),
    }
    return {k: round(_clamp01(v), 3) for k, v in scores.items()}


def label_stem(stem_path: str | Path, top_k: int = 2) -> list[dict]:
    """Return top-k `[{label, score}]` for one stem file."""
    feats = extract_features(stem_path)
    if feats.get("silent"):
        return []
    scores = score_features(feats)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(top_k, 1)]
    return [{"label": k, "score": v} for k, v in ranked]


def _alias_for(stem_path: Path, tags: list[dict]) -> Path | None:
    """Create `stem[a,b].wav` symlink (fallback: copy). None if skipped."""
    if not tags or tags[0]["score"] < _ALIAS_MIN_SCORE:
        return None
    names = ",".join(t["label"] for t in tags)
    alias = stem_path.with_name(f"{stem_path.stem}[{names}]{stem_path.suffix}")
    if alias.exists():
        return alias
    try:
        alias.symlink_to(stem_path.name)
    except (OSError, NotImplementedError):
        import shutil

        shutil.copy2(stem_path, alias)
    return alias


def label_path(path: str | Path, top_k: int = 2) -> dict:
    """Tag a single stem `.wav` or every stem in a dir.

    Single file → `{stem: tags}` + alias next to the file.
    Directory → same as :func:`label_stems`.
    """
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() != ".wav":
            raise ValueError(f"not a .wav stem file: {p}")
        tags = label_stem(p, top_k=top_k)
        if tags:
            _alias_for(p, tags)
        return {p.stem: tags}
    if p.is_dir():
        return label_stems(p, top_k=top_k)
    raise FileNotFoundError(f"stem file or dir not found: {p}")


def label_stems(stems_dir: str | Path, top_k: int = 2) -> dict:
    """Tag every `*.wav` in a stems dir; update `stems.json`; make aliases.

    Returns `{stem_name: [{label, score}], ...}`.
    """
    d = Path(stems_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"stems dir not found: {d}")
    wavs = sorted(p for p in d.glob("*.wav") if "[" not in p.stem)
    if not wavs:
        raise FileNotFoundError(f"no stem wavs in {d}")
    labels: dict[str, list[dict]] = {}
    for wav in wavs:
        tags = label_stem(wav, top_k=top_k)
        labels[wav.stem] = tags
        if tags:
            _alias_for(wav, tags)

    meta_path = d / "stems.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
        meta["labels"] = labels
        meta_path.write_text(json.dumps(meta, indent=2))
    return labels
