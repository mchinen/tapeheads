#!/usr/bin/env python3
"""Download manifest audio and create lossless, mono analysis derivatives.

Usage:
    python3 tools/prepare_audio.py --resolve
    python3 tools/prepare_audio.py --download
    python3 tools/prepare_audio.py --download --only dodge_image

Search results are saved for review. Downloads require a resolved URL in the
manifest. Source audio remains at its native channel count and sample rate.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]


def save_json(path: pathlib.Path, value: object) -> None:
    """Write JSON atomically so interrupted runs keep a valid manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def find_tool(name: str, supplied: str | None = None) -> str:
    """Find a supplied tool, project-local tool, or executable on PATH."""
    candidates = [supplied, shutil.which(name), ROOT / '.tools' / name]
    if name in ('ffmpeg', 'ffprobe'):
        candidates.append(
            ROOT.parent / 'treemusic' / '.toolchain' / 'bin' / name
        )
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).is_file():
            return str(pathlib.Path(candidate).resolve())
    option = name.replace('_', '-')
    raise FileNotFoundError(f'{name} unavailable; pass --{option}')


def run(
    command: list[str],
    log_path: pathlib.Path | None = None,
    timeout: float = 900,
) -> str:
    """Run an argument list, optionally retaining stdout and stderr in a log."""
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return result.stdout


def describe_audio(path: pathlib.Path, ffprobe: str) -> dict[str, Any]:
    """Return file provenance and the decoded audio stream properties."""
    metadata = json.loads(
        run(
            [
                ffprobe,
                '-v',
                'error',
                '-select_streams',
                'a:0',
                '-show_entries',
                'stream=sample_rate,channels,duration:format=duration',
                '-of',
                'json',
                str(path),
            ]
        )
    )
    stream = metadata['streams'][0]
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return {
        'path': str(path.relative_to(ROOT)),
        'sha256': digest.hexdigest(),
        'bytes': path.stat().st_size,
        'sample_rate_hz': int(stream['sample_rate']),
        'channels': int(stream['channels']),
        'duration_seconds': float(metadata['format']['duration']),
    }


def resolve(entry: dict[str, Any], downloader: str) -> None:
    """Save five search candidates without silently selecting a recording."""
    response = json.loads(
        run(
            [
                downloader,
                '--ignore-config',
                '--flat-playlist',
                '--dump-single-json',
                '--socket-timeout',
                '15',
                '--retries',
                '1',
                'ytsearch5:' + entry['query'],
            ],
            timeout=120,
        )
    )
    entry['candidates'] = [
        {
            key: candidate.get(key)
            for key in ('id', 'title', 'url', 'channel', 'duration')
        }
        for candidate in response.get('entries', [])
    ]
    if not entry['candidates']:
        raise RuntimeError('No search results')
    entry['status'] = 'resolved' if entry.get('url') else 'needs_selection'


def prepare(
    entry: dict[str, Any],
    downloader: str,
    ffmpeg: str,
    ffprobe: str,
    cookies: str | None = None,
    browser: str | None = None,
) -> None:
    """Download one selected recording, decode it, and record provenance."""
    if not entry.get('url'):
        raise ValueError('Select a candidate URL in data/sources.json first')
    directory = ROOT / 'data' / 'downloads' / entry['id']
    directory.mkdir(parents=True, exist_ok=True)
    log = ROOT / 'data' / 'logs' / (entry['id'] + '.log')
    command = [
        downloader,
        '--ignore-config',
        '--no-playlist',
        '--no-progress',
        '--js-runtimes',
        'node',
        '--cache-dir',
        str(ROOT / '.cache' / 'yt-dlp'),
        '--socket-timeout',
        '20',
        '--retries',
        '2',
        '--fragment-retries',
        '2',
        '--ffmpeg-location',
        str(pathlib.Path(ffmpeg).parent),
        '-f',
        'bestaudio/best',
        '--write-info-json',
        '--print',
        'after_move:filepath',
        '--no-simulate',
        '-o',
        str(directory / '%(id)s.%(ext)s'),
        entry['url'],
    ]
    if cookies:
        command.extend(['--cookies', str(cookies)])
    if browser:
        command.extend(['--cookies-from-browser', browser])
    excerpt = entry.get('excerpt')
    if excerpt:
        start = excerpt['start_seconds']
        end = start + excerpt['duration_seconds']
        command.extend(['--download-sections', f'*{start}-{end}'])
    output = run(command, log, timeout=1800)
    paths = [
        pathlib.Path(line)
        for line in output.splitlines()
        if line.strip() and pathlib.Path(line).is_file()
    ]
    if not paths:
        raise RuntimeError(f'Downloader returned no media path; see {log}')
    original = paths[-1]
    info_files = list(directory.glob('*.info.json'))
    if info_files:
        metadata = json.loads(info_files[0].read_text())
        entry['recording'] = {
            key: metadata.get(key)
            for key in (
                'id',
                'title',
                'channel',
                'channel_url',
                'webpage_url',
                'duration',
                'upload_date',
                'license',
                'format_id',
                'acodec',
                'abr',
            )
        }
    target = ROOT / 'data' / 'audio' / entry['id']
    target.mkdir(parents=True, exist_ok=True)
    files = {'download': describe_audio(original, ffprobe)}
    for label, sample_rate in (
        ('source', None),
        ('mono_16000', 16000),
        ('mono_24000', 24000),
    ):
        destination = target / (label + '.flac')
        temporary = target / (label + '.partial.flac')
        command = [
            ffmpeg,
            '-v',
            'error',
            '-y',
            '-i',
            str(original),
            '-map',
            '0:a:0',
            '-vn',
        ]
        if excerpt:
            command.extend(['-t', str(excerpt['duration_seconds'])])
        if sample_rate:
            command.extend(['-ac', '1', '-ar', str(sample_rate)])
        command.extend(['-c:a', 'flac', str(temporary)])
        run(command)
        temporary.replace(destination)
        files[label] = describe_audio(destination, ffprobe)
    entry['files'] = files
    entry['processing'] = {
        'normalization': 'none',
        'resampling': 'ffmpeg default resampler',
        'downmix': 'ffmpeg default mono channel mapping',
        'ffmpeg_version': run([ffmpeg, '-version']).splitlines()[0],
        'yt_dlp_version': run([downloader, '--version']).strip(),
        'prepared_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    entry['status'] = 'ready'
    entry.pop('error', None)


def main() -> int:
    """Resolve or download selected manifest entries; continue after failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=pathlib.Path, default=ROOT / 'data' / 'sources.json'
    )
    parser.add_argument('--resolve', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--only', nargs='+', help='Manifest IDs to process.')
    parser.add_argument('--yt-dlp')
    parser.add_argument('--ffmpeg')
    parser.add_argument('--ffprobe')
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument(
        '--cookies',
        type=pathlib.Path,
        help='Explicit Netscape cookie file; not saved in manifest.',
    )
    authentication.add_argument(
        '--cookies-from-browser',
        help='Local browser, optionally with profile: firefox[:PROFILE].',
    )
    arguments = parser.parse_args()
    if not (arguments.resolve or arguments.download):
        parser.error('Specify --resolve or --download')
    manifest = json.loads(arguments.manifest.read_text())
    entries = manifest['entries']
    if arguments.only:
        unknown = set(arguments.only) - {entry['id'] for entry in entries}
        if unknown:
            parser.error(f'Unknown IDs: {sorted(unknown)}')
    downloader = find_tool('yt-dlp', arguments.yt_dlp)
    ffmpeg = (
        find_tool('ffmpeg', arguments.ffmpeg) if arguments.download else None
    )
    ffprobe = (
        find_tool('ffprobe', arguments.ffprobe) if arguments.download else None
    )
    failed = False
    for entry in entries:
        if arguments.only and entry['id'] not in arguments.only:
            continue
        files = entry.get('files', {})
        if (
            entry['status'] == 'ready'
            and files
            and all(
                (ROOT / value['path']).is_file() for value in files.values()
            )
        ):
            print(entry['id'], 'already ready', flush=True)
            continue
        print(entry['id'], 'starting', flush=True)
        try:
            if arguments.resolve and not entry.get('url'):
                resolve(entry, downloader)
            if arguments.download:
                prepare(
                    entry,
                    downloader,
                    ffmpeg,
                    ffprobe,
                    arguments.cookies,
                    arguments.cookies_from_browser,
                )
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as error:
            entry['status'] = (
                'blocked_auth' if 'confirm you' in str(error) else 'failed'
            )
            entry['error'] = str(error)
            failed = True
        save_json(arguments.manifest, manifest)
        print(entry['id'], entry['status'], flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
