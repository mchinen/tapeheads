"""Shared file operations and audio decoding."""

import gzip
import hashlib
import json
import pathlib
import shutil
import subprocess

import numpy as np

from tapeheads import trace_types

ROOT = pathlib.Path(__file__).resolve().parents[1]


def executable(name: str) -> str:
    """Find FFmpeg tooling on PATH or in the adjacent treemusic toolchain."""
    found = shutil.which(name)
    if found:
        return found
    for base in (
        ROOT / '.tools',
        ROOT.parent / 'treemusic' / '.toolchain' / 'bin',
    ):
        path = base / name
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f'Install {name} or add it to PATH')


def sha256(path: trace_types.PathLike) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: trace_types.PathLike, data: object) -> None:
    """Write strict, indented JSON atomically."""
    path = pathlib.Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    payload = json.dumps(data, indent=2, allow_nan=False) + '\n'
    if path.suffix == '.gz':
        temporary.write_bytes(
            gzip.compress(payload.encode(), compresslevel=1, mtime=0)
        )
    else:
        temporary.write_text(payload)
    temporary.replace(path)


def decode(
    path: trace_types.PathLike,
    start_seconds: float = 0,
    duration_seconds: float | None = None,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Decode an explicitly bounded mono analysis signal with FFmpeg."""
    command = [
        executable('ffmpeg'),
        '-v',
        'error',
        '-i',
        str(path),
        '-ss',
        str(start_seconds),
    ]
    if duration_seconds is not None:
        command.extend(['-t', str(duration_seconds)])
    command.extend(
        [
            '-map',
            '0:a:0',
            '-ac',
            '1',
            '-ar',
            str(sample_rate),
            '-f',
            'f32le',
            '-',
        ]
    )
    result = subprocess.run(command, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if audio.size < 400 or not np.isfinite(audio).all():
        raise ValueError('Audio must contain at least 25 ms of finite samples')
    return audio


def read_json(path: trace_types.PathLike) -> trace_types.JsonObject:
    """Read a plain or compressed JSON object; reject other root values."""
    path = pathlib.Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            value = json.load(stream)
    else:
        value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return value
