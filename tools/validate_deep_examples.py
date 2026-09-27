"""Record native checks and reconstructed audio checks for deep examples."""

import json

from tapeheads import io
from tapeheads import validation


def main() -> None:
    """Write a reproducible report for eight-layer MuQ studies."""
    names = (
        'muq_depth8_verified',
        'chirp_sequence_muq_depth8',
        'blue_suede_shoes_full_muq_depth8',
    )
    report = {
        'generator': 'python -m tools.validate_deep_examples',
        'native_attention_errors': [],
        'validation': {},
    }
    for name in names:
        path = io.ROOT / 'outputs' / name / 'trace.json'
        trace = json.loads(path.read_text())
        report['validation'][name] = validation.validate(path)
        if name == 'muq_depth8_verified':
            report['native_attention_errors'] = [
                layer['model']['native_attention_max_error']
                for layer in trace['layers']
            ]
    io.write_json(io.ROOT / 'data/deep-validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
