import unittest

from app.config import _validate_chunking


class ChunkingValidationTests(unittest.TestCase):
    def test_rejects_non_positive_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "CHUNK_SIZE"):
            _validate_chunking(0, 0)

    def test_rejects_overlap_larger_than_chunk(self) -> None:
        with self.assertRaisesRegex(ValueError, "CHUNK_OVERLAP"):
            _validate_chunking(100, 100)

    def test_accepts_valid_chunking(self) -> None:
        _validate_chunking(1000, 200)


if __name__ == "__main__":
    unittest.main()
