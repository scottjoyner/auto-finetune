"""Security policy unit tests; no privileged account mutations."""
import unittest

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

    def test_service_is_explicitly_not_provisioned(self):
        from experiments.dust.k2_restricted_receipt_service import main
        with self.assertRaisesRegex(SystemExit, "RECEIVER_NOT_PROVISIONED"):
            main()


if __name__ == "__main__":
    unittest.main()
