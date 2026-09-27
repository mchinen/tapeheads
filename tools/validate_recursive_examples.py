"""Record native-attention checks and audits of recursive example assets."""

import json

from tapeheads import io
from tapeheads.validation import validate


def main() -> None:
    """Write a reproducible report for the two performances and short checks."""
    names = (
        'chirp_sequence_wav2vec2_layer2',
        'blue_suede_shoes_full_wav2vec2_layer2',
        'recursive_verified_wav2vec2',
        'recursive_verified_hubert',
        'recursive_verified_muq',
    )
    report = {
        'generator': 'python -m tools.validate_recursive_examples',
        'validation': {},
        'native_attention_max_error': {},
    }
    for name in names:
        path = io.ROOT / 'outputs' / name / 'trace.json'
        trace = json.loads(path.read_text())
        report['validation'][name] = validate(path)
        errors = [
            layer['model'].get('native_attention_max_error')
            for layer in trace['layers']
        ]
        if all(error is not None for error in errors):
            report['native_attention_max_error'][trace['encoder']] = errors
    io.write_json(io.ROOT / 'data' / 'recursive-validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
