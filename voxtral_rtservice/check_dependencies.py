"""Fail on dependency conflicts except the exact upstream NCCL override.

vLLM 0.32.0's Dockerfile pins NCCL 2.30.7 for DeepEPv2, intentionally overriding
torch 2.13.0+cu130's 2.29.7 metadata requirement. Do not downgrade that runtime
or suppress arbitrary pip-check failures. Review this exception on upgrades.
"""

from importlib.metadata import version
import subprocess
import sys


def main():
    expected = {"vllm": "0.32.0", "torch": "2.13.0+cu130", "nvidia-nccl-cu13": "2.30.7"}
    for package, pin in expected.items():
        if version(package) != pin:
            raise SystemExit(f"Unexpected runtime version: {package}")
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    allowed = ('torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; '
               'platform_system == "Linux", but you have nvidia-nccl-cu13 2.30.7.')
    errors = [line for line in result.stdout.splitlines()
              if line not in (allowed, "No broken requirements found.")]
    if errors or result.stderr or result.returncode not in (0, 1) or not result.stdout.strip():
        print(result.stdout, result.stderr)
        raise SystemExit(1)
    print("Dependencies validated; retained the exact upstream NCCL 2.30.7 override")


if __name__ == "__main__":
    main()
