import subprocess


def _render(*args: str) -> str:
    # Use helm template to render charts; assumes `helm` is available in PATH.
    cmd = ["helm", "template", "nanosamurai", "./charts/nanosamurai-stack", *args]
    return subprocess.check_output(cmd, text=True)


def test_nanosamurai_chart_propagates_hf_token_to_diarization_workers() -> None:
    # Ensure hfTokenSecret is passed to rtservice AND to whisperx/finalizer.
    rendered = _render(
        "-f",
        "./charts/nanosamurai-stack/values.local.docker-desktop.yaml",
        "--set",
        "rtservice.hfTokenSecret.name=nanosamurai-hf",
        "--set",
        "rtservice.hfTokenSecret.key=HF_TOKEN",
    )

    # must appear 3x: rtservice + whisperx_worker + finalizer_worker
    assert rendered.count("- name: HF_TOKEN") >= 3

    # Local overlay should also relax rollout/probe settings for slow cold starts.
    assert "progressDeadlineSeconds: 3600" in rendered
    assert "failureThreshold: 360" in rendered


def test_nanosamurai_chart_renders_enrollment_env_when_backend_s3_manifest() -> None:
    rendered = _render(
        "-f",
        "./charts/nanosamurai-stack/values.local.docker-desktop.yaml",
        "--set",
        "rtservice.hfTokenSecret.name=nanosamurai-hf",
        "--set",
        "rtservice.hfTokenSecret.key=HF_TOKEN",
    )

    # values.local.docker-desktop.yaml sets enrollment.backend=s3_manifest
    assert "- name: ENROLL_BACKEND" in rendered
    assert "- name: ENROLL_S3_ENDPOINT" in rendered
    assert "- name: ENROLL_S3_BUCKET" in rendered


def test_nanosamurai_chart_renders_bff_guest_tenant_id_env() -> None:
    rendered = _render(
        "-f",
        "./charts/nanosamurai-stack/values.local.docker-desktop.yaml",
        "--set",
        "bff.auth.required=false",
        "--set",
        "bff.auth.guestTenantId=00000000-0000-0000-0000-000000000000",
    )

    assert "- name: SAMURAIBFF_AUTH_GUEST_TENANT_ID" in rendered
    assert "00000000-0000-0000-0000-000000000000" in rendered
