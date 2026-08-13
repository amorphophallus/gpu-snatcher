import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gpu_reservation_worker as worker


class QueryComputePidsTest(unittest.TestCase):
    @mock.patch("gpu_reservation_worker.subprocess.run")
    def test_filters_by_exact_gpu_uuid_and_ignores_malformed_rows(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "GPU-a, 101\n"
                "GPU-b, 202\n"
                "GPU-a, not-a-pid\n"
                "malformed\n"
            ),
            stderr="",
        )

        self.assertEqual(worker.query_compute_pids("GPU-a"), {101})

    @mock.patch("gpu_reservation_worker.pid_owner_uid")
    @mock.patch("gpu_reservation_worker.query_compute_pids")
    def test_finds_same_user_requesters_but_ignores_worker(
        self,
        query_compute_pids,
        pid_owner_uid,
    ):
        query_compute_pids.return_value = {100, 200, 300}
        pid_owner_uid.side_effect = {200: 1007, 300: 1008}.get

        result = worker.find_requester_pids(
            "GPU-a",
            yield_uid=1007,
            ignored_pids={100},
        )

        self.assertEqual(result, {200})


class StateFileTest(unittest.TestCase):
    def test_state_write_is_complete_json_and_replaces_previous_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reservation.json"
            worker.write_state(path, state="starting", pid=1)
            worker.write_state(path, state="holding", pid=1, held_mib=1024)

            payload = __import__("json").loads(path.read_text())
            self.assertEqual(payload["state"], "holding")
            self.assertEqual(payload["held_mib"], 1024)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
