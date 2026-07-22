"""Concurrency tests for shared auto-finetune runtime leases."""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile

import pytest

from src.locking import Lease, LeaseSet, ResourceBusy, ResourceRequest, command_resources


def _hold(lock_dir: str, resource: str, ready, release) -> None:
    with Lease(lock_dir, ResourceRequest(resource)):
        ready.set()
        release.wait(10)


def test_second_process_cannot_acquire_exclusive_lease():
    with tempfile.TemporaryDirectory() as tmp:
        ready = mp.Event()
        release = mp.Event()
        proc = mp.Process(target=_hold, args=(tmp, "pipeline", ready, release))
        proc.start()
        assert ready.wait(5)
        try:
            with pytest.raises(ResourceBusy, match="pipeline"):
                Lease(tmp, ResourceRequest("pipeline")).acquire()
        finally:
            release.set()
            proc.join(5)
        assert proc.exitcode == 0


def test_shared_dataset_readers_can_coexist_but_writer_is_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        first = Lease(tmp, ResourceRequest("datasets", shared=True)).acquire()
        second = Lease(tmp, ResourceRequest("datasets", shared=True)).acquire()
        try:
            with pytest.raises(ResourceBusy, match="datasets"):
                Lease(tmp, ResourceRequest("datasets")).acquire()
        finally:
            second.release()
            first.release()


def test_lease_set_releases_earlier_resources_on_partial_failure():
    with tempfile.TemporaryDirectory() as tmp:
        blocker = Lease(tmp, ResourceRequest("gpu")).acquire()
        try:
            leases = LeaseSet(tmp, [ResourceRequest("datasets"), ResourceRequest("gpu")])
            with pytest.raises(ResourceBusy):
                leases.acquire()
            # datasets was acquired first and must have been rolled back.
            probe = Lease(tmp, ResourceRequest("datasets")).acquire()
            probe.release()
        finally:
            blocker.release()


def test_training_uses_shared_dataset_and_exclusive_gpu_checkpoint():
    reqs = {r.name: r for r in command_resources("train", "hermes-reasoning")}
    assert reqs["datasets"].shared
    assert not reqs["gpu"].shared
    assert not reqs["checkpoint-hermes-reasoning"].shared


def test_worktree_path_does_not_change_lock_identity():
    reqs_a = [r.name for r in command_resources("format", "ssd")]
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            reqs_b = [r.name for r in command_resources("format", "ssd")]
        finally:
            os.chdir(old)
    assert reqs_a == reqs_b == ["datasets"]
