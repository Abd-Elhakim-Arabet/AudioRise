"""Local Demucs separation (Increment 2).

Fully local: PyTorch + Demucs `htdemucs` on CPU by default.
First run downloads the model (~80 MB), afterwards works offline.

Output contract (stable since Increment 1):
    separate(input, outdir) -> dict with keys:
      stems: list of {name, path, rms_db}
      stems_json: path to stems.json
      source: probe_audio() dict
      model, device, sample_rate
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .probe import probe_audio

EXPECTED_STEMS = ("vocals", "drums", "bass", "other")
TARGET_SR = 44100


class SeparationNotAvailable(RuntimeError):
    pass


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
    except ImportError as exc:
        raise SeparationNotAvailable(
            "separation needs torch + torchaudio + demucs: "
            "`pip install -e packages/core[separation]`"
        ) from exc


def _load_stereo_wav(path: Path, target_sr: int = TARGET_SR):
    """Load any soundfile-readable audio as float32 stereo Tensor [2, T]."""
    import torch

    try:
        import soundfile as sf
    except ImportError as exc:
        raise SeparationNotAvailable(
            "separation needs soundfile: `pip install -e packages/core[separation]`"
        ) from exc

    data, sr = sf.read(str(path), always_2d=True)  # [T, C] float64
    wav = torch.from_numpy(data.T).float()  # [C, T]
    if sr != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav


def _rms_db(wav) -> float:
    import torch

    rms = torch.sqrt(torch.mean(wav**2) + 1e-12).item()
    return round(20.0 * math.log10(rms + 1e-12), 2)


def _run_model(wav, model_name: str, device: str):
    """Run Demucs. Split out for tests to monkeypatch (no download)."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model(model_name)
    model.to(device)
    model.eval()
    mix = wav.unsqueeze(0).to(device)  # [1, C, T]
    with __import__("torch").no_grad():
        sources = apply_model(
            model, mix, device=device, split=True, overlap=0.25, progress=False
        )[0]  # [nsrc, C, T]
    return model.sources, sources.cpu(), int(model.samplerate)


def _save_wav(path: Path, wav, sr: int) -> None:
    import numpy as np

    try:
        import soundfile as sf
    except ImportError as exc:
        raise SeparationNotAvailable(
            "separation needs soundfile: `pip install -e packages/core[separation]`"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    data = wav.detach().cpu().clamp(-1.0, 1.0).t().numpy()  # [T, C]
    sf.write(str(path), data, sr, subtype="PCM_16")


def separate(
    input_path: str | Path,
    outdir: str | Path = "stems",
    model: str = "htdemucs",
    device: str | None = None,
) -> dict:
    _require_torch()
    import torch

    src = Path(input_path)
    if not src.is_file():
        raise FileNotFoundError(f"audio file not found: {src}")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wav = _load_stereo_wav(src)
    names, sources, sr = _run_model(wav, model, device)

    stems: list[dict] = []
    for name, audio in zip(names, sources):
        stem_path = out / f"{name}.wav"
        _save_wav(stem_path, audio, sr)
        stems.append({"name": name, "path": str(stem_path), "rms_db": _rms_db(audio)})

    # Guarantee the 4-stem contract even if a model returns extras/fewer.
    have = {s["name"] for s in stems}
    for missing in EXPECTED_STEMS:
        if missing not in have:
            silence = torch.zeros((2, sources.shape[-1]))
            p = out / f"{missing}.wav"
            if not Path(p).exists():
                _save_wav(p, silence, sr)
                stems.append({"name": missing, "path": str(p), "rms_db": -120.0})

    result = {
        "source": probe_audio(src),
        "model": model,
        "device": device,
        "sample_rate": sr,
        "stems": sorted(stems, key=lambda s: s["name"]),
        "stems_json": str(out / "stems.json"),
        "labels": {},  # reserved for Increment 3
    }
    Path(result["stems_json"]).write_text(json.dumps(result, indent=2))
    return result
