"""Small offline unit tests for the real-checkpoint smoke admission path."""
import os
import unittest
from unittest.mock import patch
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment


class RealHookResearchGuardTests(unittest.TestCase):
    def test_offline_flags_and_sensitive_mounts_refused(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "1"}):
            with self.assertRaises(ResearchRefused):
                verify_restricted_environment()
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            with patch("src.lfm3_real_checkpoint_smoke.Path.exists", return_value=True):
                with self.assertRaises(ResearchRefused):
                    verify_restricted_environment()

    def test_admission_requires_offline_variables_and_no_sensitive_mounts(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            with patch("src.lfm3_real_checkpoint_smoke.Path.exists", return_value=False):
                verify_restricted_environment()


if __name__ == "__main__":
    unittest.main()
