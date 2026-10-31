"""No-network tests: FA comparison must deny wrong binaries or build flags."""
import importlib.util
import pathlib
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lfm3_compare_hip_fa", ROOT / "compare_hip_fa.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

BASE_HASH = "a" * 64
CANDIDATE_HASH = "b" * 64


def fake_ssh(command, timeout=20):
    if "build-hip-gfx1151-fa-q8" in command:
        return types.SimpleNamespace(
            returncode=0,
            stdout=CANDIDATE_HASH + "  /fake/path\n"
                   "GGML_CUDA_FA:BOOL=ON\n"
                   "GGML_CUDA_FA_QUANTS:STRING=f16-f16;q8_0-q8_0\n"
                   "GGML_HIP:BOOL=ON\n",
        )
    return types.SimpleNamespace(
        returncode=0,
        stdout=BASE_HASH + "  /fake/path\n"
               "GGML_CUDA_FA:BOOL=OFF\n"
               "GGML_HIP:BOOL=ON\n",
    )


class TestHIPFAPreflight(unittest.TestCase):
    def test_accepts_distinct_pinned_hip_variants(self):
        with patch.object(module, "ssh", side_effect=fake_ssh) as ssh:
            hashes = module.prepare()
        self.assertEqual(hashes["HIP-noFA"], BASE_HASH)
        self.assertEqual(hashes["HIP-FA"], CANDIDATE_HASH)
        self.assertEqual(ssh.call_count, 2)

    def test_refuses_same_binary_without_candidate(self):
        def same(command, timeout=20):
            result = fake_ssh(command, timeout)
            if "build-hip-gfx1151-fa-q8" in command:
                result.stdout = result.stdout.replace(CANDIDATE_HASH, BASE_HASH, 1)
            return result
        with patch.object(module, "ssh", side_effect=same):
            with self.assertRaisesRegex(RuntimeError, "same as baseline"):
                module.prepare()

    def test_refuses_candidate_without_q8_kernel(self):
        def no_q8(command, timeout=20):
            result = fake_ssh(command, timeout)
            result.stdout = result.stdout.replace("q8_0-q8_0", "q4_0-q4_0")
            return result
        with patch.object(module, "ssh", side_effect=no_q8):
            with self.assertRaisesRegex(RuntimeError, "Q8 kernel"):
                module.prepare()

    def test_refuses_candidate_compiled_without_flash_attention(self):
        def no_fa(command, timeout=20):
            result = fake_ssh(command, timeout)
            if "build-hip-gfx1151-fa-q8" in command:
                result.stdout = result.stdout.replace("GGML_CUDA_FA:BOOL=ON", "GGML_CUDA_FA:BOOL=OFF")
            return result
        with patch.object(module, "ssh", side_effect=no_fa):
            with self.assertRaisesRegex(RuntimeError, "incorrect Flash Attention"):
                module.prepare()

    def test_refuses_missing_binary(self):
        def missing(command, timeout=20):
            if "build-hip-gfx1151-fa-q8" in command:
                return types.SimpleNamespace(returncode=1, stdout="")
            return fake_ssh(command, timeout)
        with patch.object(module, "ssh", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                module.prepare()


if __name__ == "__main__":
    unittest.main()
