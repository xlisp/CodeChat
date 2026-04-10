"""Docker-based SWE-bench reward computation for RL training.

Rewards model-generated patches by:
1. (Fast) Checking patch syntax — reject garbage immediately
2. (Docker) Applying patch in the instance's Docker image and running tests

Reward scale (configurable via apply_bonus / test_weight):
  0.0  — empty patch or patch doesn't apply
  0.3  — patch applies cleanly (apply_bonus)
  0.3 + 0.7 * (passed/total) — partial credit for FAIL_TO_PASS tests
  1.0  — all FAIL_TO_PASS tests pass

Three modes:
  "docker"     — full: apply patch + run tests in Docker (slow, ~1-5 min/eval)
  "apply-only" — medium: only `git apply --check` in Docker (fast, ~5-10s/eval)
  "syntax"     — fast: regex check for valid unified-diff syntax (no Docker)

Pre-requisites (for docker / apply-only modes):
  pip install swebench
  Docker installed (`docker info` must work without sudo)
  Images pre-built: python -m scripts.rl_swebench --prepare-images ...
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class RewardResult:
    """Detailed reward breakdown for logging / debugging."""

    reward: float
    patch_applies: bool
    tests_passed: int
    tests_total: int
    error: str = ""


class SWEBenchReward:
    """Compute RL rewards for SWE-bench patches via Docker evaluation.

    Usage::

        env = SWEBenchReward(mode="docker", max_workers=4)
        result = env.compute_reward(instance_dict, patch_str)
        print(result.reward)  # 0.0 .. 1.0

        # batch (parallel Docker for a GRPO group):
        results = env.compute_rewards_batch(instance_dict, [p1, p2, p3, p4])
    """

    def __init__(
        self,
        mode: str = "docker",
        max_workers: int = 2,
        timeout: int = 300,
        apply_bonus: float = 0.3,
        test_weight: float = 0.7,
    ):
        assert mode in ("docker", "apply-only", "syntax")
        self.mode = mode
        self.max_workers = max_workers
        self.timeout = timeout
        self.apply_bonus = apply_bonus
        self.test_weight = test_weight
        self._spec_cache: dict[str, Any] = {}
        self._pool = (
            ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
        )

        if mode != "syntax":
            self._check_docker()
        if mode == "docker":
            self._check_swebench()

    # ------------------------------------------------------------------
    # Environment checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_docker():
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            raise RuntimeError(
                "Docker is not available. Install Docker and ensure the current "
                "user can run `docker info` without sudo."
            ) from e

    @staticmethod
    def _check_swebench():
        try:
            import swebench  # noqa: F401
        except ImportError:
            raise ImportError(
                "pip install swebench  (required for Docker-based reward)"
            )

    # ------------------------------------------------------------------
    # Test spec helpers
    # ------------------------------------------------------------------

    def _get_test_spec(self, instance: dict):
        iid = instance["instance_id"]
        if iid not in self._spec_cache:
            from swebench.harness.test_spec import make_test_spec

            self._spec_cache[iid] = make_test_spec(instance)
        return self._spec_cache[iid]

    def _get_image_name(self, instance: dict) -> str:
        """Derive the Docker image name for an instance via swebench TestSpec."""
        try:
            spec = self._get_test_spec(instance)
            # swebench >=2.x exposes the image name differently across versions
            for attr in ("instance_image_key", "image_key", "base_image_key"):
                val = getattr(spec, attr, None)
                if val:
                    return val
        except Exception:
            pass
        # Fallback convention used by official harness
        iid = instance["instance_id"]
        return f"sweb.eval.x86_64.{iid}:latest"

    @staticmethod
    def _patch_is_valid(patch: str) -> bool:
        """Quick syntax check: does the string look like a unified diff?"""
        if not patch or not patch.strip():
            return False
        return bool(re.search(r"^@@\s", patch, re.MULTILINE))

    @staticmethod
    def _parse_fail_to_pass(instance: dict) -> list[str]:
        f2p = instance.get("FAIL_TO_PASS", [])
        if isinstance(f2p, str):
            f2p = json.loads(f2p)
        return f2p

    # ------------------------------------------------------------------
    # Docker-based evaluation (full)
    # ------------------------------------------------------------------

    def _build_eval_script(self, test_cmd: str) -> str:
        """Shell script executed inside the Docker container."""
        return f"""#!/bin/bash
set -e
cd /testbed

# --- Apply the model's patch ---
echo "@@SWEBRL_APPLY_START@@"
git apply /tmp/model.patch 2>&1
APPLY_EXIT=$?
echo "@@SWEBRL_APPLY_EXIT=$APPLY_EXIT@@"

if [ $APPLY_EXIT -ne 0 ]; then
    echo "@@SWEBRL_PATCH_FAILED@@"
    exit 0
fi
echo "@@SWEBRL_PATCH_APPLIED@@"

# --- Run the test suite ---
echo "@@SWEBRL_TESTS_START@@"
{test_cmd} 2>&1 || true
echo "@@SWEBRL_TESTS_DONE@@"
"""

    def _evaluate_docker(self, instance: dict, patch: str) -> RewardResult:
        """Full Docker evaluation: apply patch + run tests."""
        fail_to_pass = self._parse_fail_to_pass(instance)
        total_f2p = len(fail_to_pass)

        if not self._patch_is_valid(patch):
            return RewardResult(0.0, False, 0, total_f2p, "invalid patch syntax")

        try:
            spec = self._get_test_spec(instance)
        except Exception as e:
            return RewardResult(0.0, False, 0, total_f2p, f"test_spec error: {e}")

        image = self._get_image_name(instance)
        test_cmd = getattr(spec, "test_cmd", "") or ""
        if not test_cmd:
            return RewardResult(0.0, False, 0, total_f2p, "no test_cmd in spec")

        tmpdir = tempfile.mkdtemp(prefix="swebrl_")
        try:
            patch_path = os.path.join(tmpdir, "model.patch")
            with open(patch_path, "w") as f:
                f.write(patch)
            script_path = os.path.join(tmpdir, "eval.sh")
            with open(script_path, "w") as f:
                f.write(self._build_eval_script(test_cmd))

            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "-v", f"{patch_path}:/tmp/model.patch:ro",
                    "-v", f"{script_path}:/tmp/eval.sh:ro",
                    image,
                    "bash", "/tmp/eval.sh",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return self._parse_docker_output(
                result.stdout, result.stderr, fail_to_pass, total_f2p
            )
        except subprocess.TimeoutExpired:
            return RewardResult(0.0, False, 0, total_f2p, "docker timeout")
        except Exception as e:
            return RewardResult(0.0, False, 0, total_f2p, f"docker error: {e}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _parse_docker_output(
        self,
        stdout: str,
        stderr: str,
        fail_to_pass: list[str],
        total_f2p: int,
    ) -> RewardResult:
        if "@@SWEBRL_PATCH_FAILED@@" in stdout:
            return RewardResult(0.0, False, 0, total_f2p, "patch apply failed")

        if "@@SWEBRL_PATCH_APPLIED@@" not in stdout:
            return RewardResult(
                0.0, False, 0, total_f2p, "unexpected output (no apply marker)"
            )

        # Patch applied — check tests
        if "@@SWEBRL_TESTS_DONE@@" not in stdout:
            return RewardResult(
                self.apply_bonus, True, 0, total_f2p, "tests did not complete"
            )

        test_output = (
            stdout.split("@@SWEBRL_TESTS_START@@")[-1]
            .split("@@SWEBRL_TESTS_DONE@@")[0]
        )
        passed = self._count_passing_tests(test_output, fail_to_pass)

        if total_f2p == 0:
            reward = self.apply_bonus
        else:
            reward = self.apply_bonus + self.test_weight * (passed / total_f2p)
        return RewardResult(reward, True, passed, total_f2p)

    @staticmethod
    def _count_passing_tests(test_output: str, fail_to_pass: list[str]) -> int:
        """Heuristic count of FAIL_TO_PASS tests that now pass.

        Supports pytest-style (PASSED/FAILED) and unittest-style (ok/FAIL)
        output. Conservative: only counts a test as passed if we find an
        explicit pass marker or the test is mentioned without a fail marker.
        """
        passed = 0
        for test_name in fail_to_pass:
            short = (
                test_name.split("::")[-1]
                if "::" in test_name
                else test_name.split(".")[-1]
            )
            esc_full = re.escape(test_name)
            esc_short = re.escape(short)

            # pytest verbose: "test_name PASSED"
            if re.search(rf"{esc_full}.*PASSED", test_output):
                passed += 1
                continue
            if re.search(rf"{esc_short}.*PASSED", test_output):
                passed += 1
                continue
            # unittest: "test_name ... ok"
            if re.search(rf"{esc_full}.*\bok\b", test_output):
                passed += 1
                continue
            if re.search(rf"{esc_short}.*\bok\b", test_output):
                passed += 1
                continue
        return passed

    # ------------------------------------------------------------------
    # Apply-only evaluation (medium speed)
    # ------------------------------------------------------------------

    def _evaluate_apply_only(self, instance: dict, patch: str) -> RewardResult:
        fail_to_pass = self._parse_fail_to_pass(instance)
        total_f2p = len(fail_to_pass)

        if not self._patch_is_valid(patch):
            return RewardResult(0.0, False, 0, total_f2p, "invalid patch syntax")

        image = self._get_image_name(instance)
        tmpdir = tempfile.mkdtemp(prefix="swebrl_")
        try:
            patch_path = os.path.join(tmpdir, "model.patch")
            with open(patch_path, "w") as f:
                f.write(patch)

            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "-v", f"{patch_path}:/tmp/model.patch:ro",
                    image,
                    "bash", "-c",
                    "cd /testbed && git apply --check /tmp/model.patch 2>&1; "
                    "echo EXIT_CODE=$?",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if "EXIT_CODE=0" in result.stdout:
                return RewardResult(self.apply_bonus, True, 0, total_f2p)
            return RewardResult(
                0.0, False, 0, total_f2p, "patch apply --check failed"
            )
        except subprocess.TimeoutExpired:
            return RewardResult(0.0, False, 0, total_f2p, "docker timeout")
        except Exception as e:
            return RewardResult(0.0, False, 0, total_f2p, f"docker error: {e}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_reward(self, instance: dict, patch: str) -> RewardResult:
        """Compute reward for a single (instance, patch) pair."""
        if self.mode == "syntax":
            valid = self._patch_is_valid(patch)
            return RewardResult(self.apply_bonus if valid else 0.0, valid, 0, 0)
        elif self.mode == "apply-only":
            return self._evaluate_apply_only(instance, patch)
        else:
            return self._evaluate_docker(instance, patch)

    def compute_rewards_batch(
        self, instance: dict, patches: list[str]
    ) -> list[RewardResult]:
        """Compute rewards for a batch of patches (same instance, multiple
        completions). Uses a thread pool for parallel Docker evaluation."""
        if self._pool is not None and self.mode != "syntax":
            futures = {
                self._pool.submit(self.compute_reward, instance, p): i
                for i, p in enumerate(patches)
            }
            results: list[RewardResult | None] = [None] * len(patches)
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = RewardResult(0.0, False, 0, 0, str(e))
            return results  # type: ignore[return-value]
        return [self.compute_reward(instance, p) for p in patches]

    def close(self):
        if self._pool:
            self._pool.shutdown(wait=False)
