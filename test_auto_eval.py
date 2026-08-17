import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT = REPO_ROOT / "auto_eval.sh"


class AutoEvalStaticTests(unittest.TestCase):
    def test_vlm_options_and_depth_contract_are_declared(self):
        script = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "--save-depth-image",
            "--annotation-source",
            "--vlm-base-url",
            "--vlm-timeout-seconds",
            "--vlm-query-interval",
            "--vlm-noise-projection-samples",
            "--tracking-metric-type",
            "--task-summary-out",
            "--print-command",
        ):
            self.assertIn(token, script)


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "requires bash")
class AutoEvalCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "robust-rearrangement"
        self.root.mkdir()
        self.checkpoint = Path(self.temp_dir.name) / "actor.pt"
        self.checkpoint.touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_auto_eval(self, *extra_args, env=None):
        command = [
            "bash",
            str(SCRIPT),
            "--steps",
            "eval",
            "--local-path",
            str(self.root),
            "--overwrite-wt-path",
            str(self.checkpoint),
            "--task",
            "one_leg",
            "--n-envs",
            "3",
            "--n-rollouts",
            "3",
            "--randomness",
            "low",
            "--print-command",
            *extra_args,
        ]
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_vlm_command_contains_complete_rgbd_contract(self):
        summary_path = Path(self.temp_dir.name) / "summaries" / "one_leg.json"
        result = self.run_auto_eval(
            "--annotation-source",
            "vlm",
            "--tracking-metric-type",
            "pose",
            "--vlm-base-url",
            "http://vlm.test:8000",
            "--vlm-timeout-seconds",
            "30",
            "--vlm-query-interval",
            "0",
            "--vlm-noise-projection-samples",
            "200",
            "--task-summary-out",
            str(summary_path),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = shlex.split(result.stdout.strip().splitlines()[-1])
        expected_pairs = {
            "--n-envs": "3",
            "--n-rollouts": "3",
            "--max-rollout-steps": "1000",
            "--randomness": "low",
            "--annotation-source": "vlm",
            "--tracking-metric-type": "pose",
            "--vlm-base-url": "http://vlm.test:8000",
            "--vlm-timeout-seconds": "30",
            "--vlm-query-interval": "0",
            "--vlm-noise-projection-samples": "200",
            "--task-summary-out": str(summary_path),
            "--wt-path": str(self.checkpoint),
        }
        self.assertEqual(argv[:3], ["python", "-m", "src.eval.evaluate_model"])
        self.assertIn("--save-depth-image", argv)
        self.assertIn("--save-rollouts", argv)
        for flag, value in expected_pairs.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_scripted_command_does_not_emit_vlm_transport_options(self):
        result = self.run_auto_eval("--annotation-source", "scripted")
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = shlex.split(result.stdout.strip().splitlines()[-1])
        self.assertEqual(argv[argv.index("--annotation-source") + 1], "scripted")
        self.assertNotIn("--vlm-base-url", argv)
        self.assertNotIn("--vlm-timeout-seconds", argv)
        self.assertNotIn("--vlm-query-interval", argv)
        self.assertIn("--save-depth-image", argv)

    def test_vlm_source_requires_base_url(self):
        env = os.environ.copy()
        env.pop("VLM_GUIDANCE_URL", None)
        result = self.run_auto_eval("--annotation-source", "vlm", env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --vlm-base-url", result.stderr)


if __name__ == "__main__":
    unittest.main()
