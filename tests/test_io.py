"""Check reproducible trace archives and atomic strict JSON writes."""

import pathlib
import tempfile
import unittest

from tapeheads import io


class TraceIoTest(unittest.TestCase):
    """Protect trace integrity when compression or serialization fails."""

    def test_compressed_and_plain_traces_agree(self):
        data = {'title': 'Birds / forêt', 'samples': [0, 400, 720]}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            plain = root / 'trace.json'
            archive = root / 'trace.json.gz'
            duplicate = root / 'copy.json.gz'
            for path in (plain, archive, duplicate):
                io.write_json(path, data)
                self.assertEqual(io.read_json(path), data)
            self.assertEqual(archive.read_bytes(), duplicate.read_bytes())

    def test_invalid_replacement_preserves_existing_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'trace.json.gz'
            io.write_json(path, {'sample': 400})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                io.write_json(path, {'sample': float('nan')})
            self.assertEqual(path.read_bytes(), before)

    def test_reader_rejects_non_object_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('trace.json', 'trace.json.gz'):
                with self.subTest(name=name):
                    path = pathlib.Path(directory) / name
                    io.write_json(path, [1, 2, 3])
                    with self.assertRaisesRegex(ValueError, 'JSON object'):
                        io.read_json(path)


if __name__ == '__main__':
    unittest.main()
