import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import ingestion


class IngestionFilenameTests(unittest.TestCase):
    def test_sanitize_filename_removes_path_separators_and_reserved_chars(self) -> None:
        self.assertEqual(ingestion._sanitize_filename("../bad:name?.pdf"), "bad_name")

    def test_sanitize_filename_falls_back_to_document(self) -> None:
        self.assertEqual(ingestion._sanitize_filename("!!!.txt"), "document")

    def test_save_uploaded_file_uses_sanitized_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ingestion, "DOCUMENTS_DIR", Path(temp_dir)):
                document_id, stored_path = ingestion._save_uploaded_file(
                    "../bad:name?.txt", ".txt", b"content"
                )

        self.assertTrue(document_id)
        self.assertEqual(stored_path.parent, Path(temp_dir))
        self.assertTrue(stored_path.name.startswith("bad_name_"))
        self.assertTrue(stored_path.name.endswith(".txt"))


if __name__ == "__main__":
    unittest.main()
