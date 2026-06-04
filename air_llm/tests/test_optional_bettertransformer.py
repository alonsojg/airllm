import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestOptionalBetterTransformer(unittest.TestCase):
    def test_import_succeeds_without_optimum_bettertransformer(self):
        repo_root = Path(__file__).resolve().parents[2]
        package_root = repo_root / "air_llm"

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_optimum = Path(temp_dir) / "optimum"
            fake_optimum.mkdir(parents=True, exist_ok=True)
            (fake_optimum / "__init__.py").write_text("", encoding="utf-8")

            env = os.environ.copy()
            original_pythonpath = env.get("PYTHONPATH", "")
            if original_pythonpath:
                env["PYTHONPATH"] = f"{temp_dir}:{package_root}:{original_pythonpath}"
            else:
                env["PYTHONPATH"] = f"{temp_dir}:{package_root}"

            cmd = [
                sys.executable,
                "-c",
                "import airllm.airllm_base as m; print(m.BetterTransformer is None)",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            self.assertEqual(result.returncode, 0, msg=result.stderr + "\n" + result.stdout)
            self.assertIn("True", result.stdout)


if __name__ == "__main__":
    unittest.main()
