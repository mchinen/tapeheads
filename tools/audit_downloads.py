"""Verify downloaded audio against its recorded hashes and decoded metadata."""

import soundfile

from tapeheads import io


def main() -> None:
    """Write a reproducible inventory without changing source recordings."""
    manifest = io.read_json(io.ROOT / 'data/sources.json')
    results = []
    for entry in manifest['entries']:
        files = []
        for role, metadata in entry.get('files', {}).items():
            path = io.ROOT / metadata['path']
            record = {
                'role': role,
                'path': metadata['path'],
                'exists': path.is_file(),
            }
            if path.is_file():
                record['hash_matches'] = io.sha256(path) == metadata['sha256']
                record['size_matches'] = (
                    path.stat().st_size == metadata['bytes']
                )
                if role != 'download':
                    audio = soundfile.info(path)
                    record['duration_seconds'] = audio.duration
                    record['metadata_matches'] = (
                        audio.samplerate == metadata['sample_rate_hz']
                        and audio.channels == metadata['channels']
                        and abs(audio.duration - metadata['duration_seconds'])
                        < 1 / audio.samplerate
                    )
            files.append(record)
        results.append(
            {'id': entry['id'], 'title': entry['title'], 'files': files}
        )
    passed = all(
        record.get('exists', False)
        and record.get('hash_matches', False)
        and record.get('size_matches', False)
        and record.get('metadata_matches', True)
        for entry in results
        for record in entry['files']
    ) and all(entry['files'] for entry in results)
    io.write_json(
        io.ROOT / 'data/download-audit.json',
        {
            'generator': 'python -m tools.audit_downloads',
            'manifest_sha256': io.sha256(io.ROOT / 'data/sources.json'),
            'passed': passed,
            'entries': results,
        },
    )
    print(f'{len(results)} recordings checked; passed={passed}')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
