# Audio collection

Prepared September 18, 2026. The selected sources and download state are in
[`data/sources.json`](../data/sources.json). Synthetic fixture parameters and
checksums are in [`data/signals.json`](../data/signals.json).

## Current state

Four individual synthetic fixtures and a composite repeated-chirp sequence are
ready, each with 48, 24, and 16 kHz mono WAV files.
All 14 YouTube recordings were downloaded for the original local studies.
Each recording has its compressed download, source FLAC, and 16/24 kHz mono
FLAC derivatives. Recording choices have not yet been manually auditioned.
Logs are under `data/logs/`. Downloads use about 79 MiB and prepared audio uses
about 1.2 GiB, including synthetic fixtures. No paid service was used.

| ID | Recording | Selection |
| --- | --- | --- |
| `dodge_image` | Charles Dodge, He Destroyed Her Image | Speech Songs II, Charles Dodge Topic upload |
| `bach_cello` | Bach, Cello Suite No. 1 in G major, BWV 1007 | Complete suite, Lucia Swarts / Netherlands Bach Society |
| `chopin_nocturne` | Chopin, Nocturne in E-flat major, Op. 9 No. 2 | Seong-Jin Cho / Deutsche Grammophon |
| `espresso` | Sabrina Carpenter, Espresso | Official audio |
| `so_what` | Miles Davis, So What | Official audio |
| `fake_plastic_trees` | Radiohead, Fake Plastic Trees | Radiohead upload |
| `sometimes` | My Bloody Valentine, Sometimes | Lost in Translation soundtrack release |
| `bohemian_rhapsody` | Queen, Bohemian Rhapsody | Queen Official audio |
| `blue_suede_shoes` | Elvis Presley, Blue Suede Shoes | Official audio, complete 122.55-second recording |
| `beach` | Beach waves | First 180 seconds; uploader describes real-time, non-looped sound |
| `rain` | Light rain without music or thunder | First 180 seconds |
| `traffic` | Night traffic on a Da Nang bridge | First 180 seconds |
| `cage_speech` | John Cage interview about silence | April 2, 1991 interview clip, about 4 minutes |
| `obama_speech` | Barack Obama, 2004 DNC keynote | C-SPAN recording, about 19 minutes |

Music and speech downloads retain the full selected recording. The Bach choice
is the complete first suite, about 17 minutes. For model development, begin
with short analysis windows and label their context; do not silently substitute
those windows for full-song attention. Environmental selections are explicit
excerpts rather than downloads of multi-hour videos.

## Synthetic fixtures

| ID | Definition |
| --- | --- |
| `sine` | 440 Hz, amplitude 0.25, 30 seconds, 10 ms edge fades |
| `chirp` | Linear sweep from 40 Hz to 7 kHz over 30 seconds, amplitude 0.25, 10 ms edge fades |
| `gaussian_noise` | Gaussian white noise, mean 0, standard deviation 0.1, seed 20260918, 30 seconds |
| `gaussian_pulse` | Gaussian pulse centered at 15 seconds, sigma 10 ms, amplitude 0.5 |

The pulse is an additional interpretation of “Gaussian”; white noise is the
main noise fixture. All signals originate at 48 kHz in PCM16. FFmpeg resamples
the same waveform to 16 and 24 kHz so the models receive corresponding audio.
The noise is quantized Gaussian noise; clipping counts are recorded. It is
not regenerated independently at each rate.

## Reproduce and resume

Install the pinned downloader:

```sh
bash tools/setup_audio.sh
```

The scripts require Python 3.10 or later, FFmpeg, and FFprobe. YouTube extraction
also uses Node.js. FFmpeg tools are found on `PATH`, in `.tools`, or in the
adjacent `treemusic/.toolchain/bin` installation. Downloader and FFmpeg paths
can also be supplied to `prepare_audio.py` explicitly.

Generate signals:

```sh
python3 -m tools.generate_signals
```

Resume the selected recordings:

```sh
python3 tools/prepare_audio.py --download
```

If YouTube requires authentication, use an explicitly supplied Netscape-format
cookies file kept outside this repository:

```sh
python3 tools/prepare_audio.py --download --cookies /path/to/youtube-cookies.txt
```

You can also authorize use of a local browser session directly:

```sh
python3 tools/prepare_audio.py --download --cookies-from-browser firefox
```

Browser profiles are read only when this option is supplied. Cookies and their
contents are not written to manifests or exported to files. Authentication may
still fail depending on YouTube's response. It is also possible to run these scripts on another machine with
working YouTube access and transfer the resulting local data directory.

To retry one source or refresh unresolved search candidates:

```sh
python3 tools/prepare_audio.py --download --only espresso
python3 tools/prepare_audio.py --resolve
```

Entries with selected URLs keep those URLs. To search again, remove an entry's
`url` and run `--resolve --only ID`. Review its candidates and set `url` before
downloading. The script never automatically substitutes the first search hit.

Run only one manifest writer at a time. Each completed or failed entry is saved
atomically. Ready entries with existing files are skipped on subsequent runs;
this existence check is not a checksum revalidation. Partial media can resume
through yt-dlp. Per-entry failures do not stop the remaining collection, and
the process exits nonzero if any entry fails.

## Files and provenance

For each downloaded recording, preserve the compressed YouTube source and its
metadata under `data/downloads/ID/`. Create these files in `data/audio/ID/`:

- `source.flac`: decoded source channel count and sample rate.
- `mono_16000.flac`: wav2vec analysis input.
- `mono_24000.flac`: MuQ analysis input.

FLAC avoids another lossy encoding step; it cannot restore information lost by
YouTube compression. No loudness normalization is applied. The manifest stores
URLs, recording titles, channel metadata, excerpt bounds, SHA-256 checksums,
duration, channel count, sample rate, and tool versions when preparation succeeds.
Synthetic fixtures use `mono_RATE.wav` instead.

Binary audio, downloads, tool executables, and logs are ignored by Git. The
scripts, selection manifest, and fixture manifest are retained. Downloaded
media is local input data; no redistribution license is inferred from a public
YouTube URL.

## Repeated chirp sequence

`data/audio/chirp_sequence/` contains a 32-second test sequence with four
identical three-second linear chirps from 40 Hz to 7 kHz. The chirps start at
1, 10, 19, and 28 seconds. Each intervening six-second interval contains:

1. A 1.5-second sine tone: 220 Hz, then 440 Hz, then 880 Hz.
2. 1.5 seconds of seeded Gaussian white noise at standard deviation 0.1.
3. One second of silence.
4. Two seconds of DTMF: `123A`, then `456B`, then `789C`. Each symbol plays for
   350 ms followed by 150 ms of silence.

The sequence also has one second of leading and trailing silence. Chirps and
sine tones have amplitude 0.25; DTMF uses amplitude 0.125 per sinusoid. Active
segments have 10 ms edge fades. The master is a float32, 48 kHz mono WAV;
16/24 kHz files are resampled copies. No level normalization is applied.

Generate and analyze it with:

```sh
.venv/bin/python -m tools.generate_chirp_sequence
.venv/bin/python -m tapeheads analyze \
  data/audio/chirp_sequence/mono_48000.wav --encoder wav2vec2 \
  --output outputs/chirp_sequence_wav2vec2 --video
```

[`data/chirp_sequence.json`](../data/chirp_sequence.json) records every segment,
its parameters and sample boundaries, generator checksum, and audio checksums.
The generated wav2vec trace uses the full 32-second context and the usual
±100 ms exclusion, allowing attention to earlier and later repeated chirps.
