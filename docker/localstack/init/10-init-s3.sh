#!/usr/bin/env bash
set -euo pipefail

# Ensure enrollment bucket exists (idempotent).
# LocalStack provides the `awslocal` wrapper.
BUCKET="${XAMURAI_ENROLLMENT_BUCKET:-xamurai-enrollment}"

awslocal s3 mb "s3://${BUCKET}" 2>/dev/null || true

# Optional debug output
awslocal s3 ls || true
