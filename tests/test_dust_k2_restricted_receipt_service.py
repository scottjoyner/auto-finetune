"""Security policy unit tests; no privileged account mutations."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from experiments.dust.k2_restricted_receipt_service import (
    parse_forced_command, verify_separate_identity,
)


class RestrictedReceiptTests(unittest.TestCase):
    def test_digest_only_command_allowed(self):
        self.assertEqual(
            parse_forced_command("receive " + "a" * 32 + " " + "b" * 64),
            ("a" * 32, "b" * 64),
        )

    def test_no_remote_shell_path_setup_or_verification(self):
        for attack in (
            None, "", "verify " + "a" * 32,
            "setup", "cat /var/lib/dust-receipt/receiver-owned.key",
            "receive " + "a" * 32 + " " + "b" * 64 + "; id",
            "receive " + "a" * 32 + " " + "b" * 64 + "\nwhoami",
            "receive " + "A" * 32 + " " + "b" * 64,
            "receive " + "a" * 32 + " " + "x" * 64,
            "receive " + "a" * 32 + " --root /home/scott/git",
        ):
            with self.subTest(attack=attack):
                with self.assertRaises(ValueError):
                    parse_forced_command(attack)

    def test_unique_uid_private_mode_policy(self):
        verify_separate_identity(
            current_uid=2501, service_uid=2501, producer_uid=1000,
            service_key_uid=2501, data_root_uid=2501,
            service_root_mode=0o700, signing_key_mode=0o600)

    def test_shared_login_or_producer_readable_key_denied(self):
        base = dict(
            current_uid=2501, service_uid=2501, producer_uid=1000,
            service_key_uid=2501, data_root_uid=2501,
            service_root_mode=0o700, signing_key_mode=0o600,
        )
        variants = [
            {"service_uid": 1000, "current_uid": 1000,
             "data_root_uid": 1000, "service_key_uid": 1000},
            {"service_uid": 0, "current_uid": 0,
             "data_root_uid": 0, "service_key_uid": 0},
            {"current_uid": 1000},
            {"service_key_uid": 1000},
            {"data_root_uid": 1000},
            {"service_root_mode": 0o750},
            {"signing_key_mode": 0o640},
        ]
        for override in variants:
            with self.subTest(override=override):
                with self.assertRaises(PermissionError):
                    verify_separate_identity(**{**base, **override})

    def test_isolated_ssh_client_uses_dedicated_forced_command_only(self):
        from experiments.dust.k2_direction_witness import IsolatedSSHReceiver
        with tempfile.TemporaryDirectory() as temp:
            identity = Path(temp) / "receipt-only-ssh.key"
            identity.write_text("synthetic testing only")
            identity.chmod(0o600)
            client = IsolatedSSHReceiver("x1-370", "a" * 32, identity)
            expected = {
                "schema": "auto-finetune.dust-k2-independent-receipt.v1",
                "run_id": "a" * 32, "batch_sha256": "b" * 64,
                "receiver_hmac_sha256": "c" * 64, "batch_index": 0,
            }
            class Result:
                stdout = json.dumps(expected)
            with patch(
                "experiments.dust.k2_direction_witness.subprocess.run",
                return_value=Result(),
            ) as invoked:
                self.assertEqual(client("b" * 64), expected)
                command = invoked.call_args.args[0]
                self.assertIn("dustreceipt", command)
                self.assertIn("IdentitiesOnly=yes", command)
                self.assertIn("ClearAllForwardings=yes", command)
                self.assertIn("StrictHostKeyChecking=yes", command)
                self.assertNotIn("/home/scott/git/wt-dust-k2-direction-witness-20261009/", 
                                 " ".join(command))
                self.assertEqual(command[-1], "receive " + "a" * 32 +
                                 " " + "b" * 64)
            identity.chmod(0o644)
            with self.assertRaises(PermissionError):
                IsolatedSSHReceiver("x1-370", "a" * 32, identity)
            with self.assertRaises(ValueError):
                IsolatedSSHReceiver("evil.example", "a" * 32, identity)

    def test_legacy_receipt_verifier_never_claims_independent_key_custody(self):
        from experiments.dust.k2_receipt_receiver import (
            prepare_receiver, receive, verify)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "legacy-receiver"
            prepare_receiver(root)
            receive(root, "a" * 32, "b" * 64)
            result = verify(root, "a" * 32)
            self.assertEqual(result["signed_batch_receipts"], 1)
            self.assertTrue(result["receiver_hmac_chain_verified"])
            self.assertIsNone(result["producer_has_receiver_key"])
            self.assertFalse(result["independent_custody_for_precommit"])
            self.assertFalse(result["authorizes_real_classifier_training"])

    def test_git_worktree_must_never_be_installed_as_service_program(self):
        from experiments.dust.k2_restricted_receipt_service import (
            verify_installed_code)
        with self.assertRaises(PermissionError):
            verify_installed_code(Path(__file__))

    def test_service_is_explicitly_not_provisioned(self):
        from experiments.dust.k2_restricted_receipt_service import main
        with self.assertRaisesRegex(SystemExit, "RECEIVER_NOT_PROVISIONED"):
            main()


if __name__ == "__main__":
    unittest.main()
