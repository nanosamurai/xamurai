{{- define "nanosamurai-stack.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "nanosamurai-stack.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s" (include "nanosamurai-stack.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "nanosamurai-stack.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "nanosamurai-stack.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Enrollment reader env vars (rtservice/whisperx_worker/finalizer_worker)

This helper emits ENROLL_* env vars depending on `.Values.enrollment.backend`.

Supported backends:
- legacy_dir: dev-only flat dir
- local_manifest: per-tenant manifest layout on a shared filesystem
- s3_manifest: per-tenant manifests + samples on S3/S3-compatible storage

Notes:
- S3 credentials are optional. If credentialsSecret.name is set, access/secret are read from it.
- Force path style is recommended for LocalStack/MinIO.
*/}}
{{- define "nanosamurai-stack.enrollmentEnv" }}
- name: ENROLL_BACKEND
  value: {{ .Values.enrollment.backend | default "legacy_dir" | quote }}

{{- $backend := (.Values.enrollment.backend | default "legacy_dir") }}
{{- if or (eq $backend "legacy_dir") (eq $backend "local_manifest") }}
- name: ENROLL_DIR
  value: {{ .Values.enrollment.dir | default "/app/enrolled_speakers" | quote }}
{{- end }}

- name: ENROLL_CACHE_TTL_S
  value: {{ (.Values.enrollment.cache.ttlSeconds | default 300) | toString | quote }}
- name: ENROLL_CACHE_MAX_TENANTS
  value: {{ (.Values.enrollment.cache.maxTenants | default 128) | toString | quote }}

{{- if eq $backend "s3_manifest" }}
- name: ENROLL_S3_BUCKET
  value: {{ .Values.enrollment.s3.bucket | default "" | quote }}
- name: ENROLL_S3_PREFIX
  value: {{ .Values.enrollment.s3.prefix | default "enrollment" | quote }}
- name: ENROLL_S3_REGION
  value: {{ .Values.enrollment.s3.region | default "" | quote }}
- name: ENROLL_S3_ENDPOINT
  value: {{ .Values.enrollment.s3.endpoint | default "" | quote }}

{{- if .Values.enrollment.s3.credentialsSecret.name }}
- name: ENROLL_S3_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.enrollment.s3.credentialsSecret.name | quote }}
      key: {{ .Values.enrollment.s3.credentialsSecret.accessKeyKey | default "accessKey" | quote }}
- name: ENROLL_S3_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.enrollment.s3.credentialsSecret.name | quote }}
      key: {{ .Values.enrollment.s3.credentialsSecret.secretKeyKey | default "secretKey" | quote }}
{{- end }}

- name: ENROLL_S3_FORCE_PATH_STYLE
  value: {{ ternary "true" "false" (.Values.enrollment.s3.forcePathStyle | default true) | quote }}
{{- end }}
{{- end }}
