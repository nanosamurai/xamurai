-- Tenants (companies, clinics, etc.)
CREATE TABLE tenants (
    tenant_id      uuid PRIMARY KEY,
    name           text NOT NULL,
    email          text NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- Users within a tenant (doctors, coaches, etc.)
CREATE TABLE app_users (
    user_id        uuid PRIMARY KEY,
    tenant_id      uuid NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    external_id    text UNIQUE,          -- e.g. auth provider ID or email
    display_name   text,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_app_users_tenant ON app_users(tenant_id);

CREATE TABLE sessions (
    session_id     uuid PRIMARY KEY,      -- app-generated UUIDv7/ULID
    tenant_id      uuid NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    user_id        uuid REFERENCES app_users(user_id) ON DELETE SET NULL,

    -- business key used across Kafka messages (can also be the same as session_id if you want)
    session_key    text NOT NULL UNIQUE,

    title          text,                  -- optional: "Cardio followup with John"
    status         text NOT NULL DEFAULT 'active',   -- active | finished | failed
    started_at     timestamptz NOT NULL DEFAULT now(),
    ended_at       timestamptz,           -- set when finalized
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_sessions_tenant_user ON sessions(tenant_id, user_id);
CREATE INDEX idx_sessions_status ON sessions(status);

CREATE TABLE recordings (
    recording_id   uuid PRIMARY KEY,
    session_id     uuid NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,

    recording_url  text NOT NULL,        -- file://..., s3://...
    duration_s     double precision NOT NULL,
    sample_rate    integer NOT NULL,
    lang           text,                 -- "en", "cs", ...
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_recordings_session ON recordings(session_id);

CREATE TABLE session_transcripts (
    transcript_id  uuid PRIMARY KEY,
    session_id     uuid NOT NULL UNIQUE REFERENCES sessions(session_id) ON DELETE CASCADE,
    recording_id   uuid REFERENCES recordings(recording_id) ON DELETE SET NULL,

    tenant_id      uuid NOT NULL,        -- duplicated for simpler querying
    user_id        uuid,                 -- duplicated as well

    full_text      text NOT NULL,        -- concatenated transcript
    lang           text,
    duration_s     double precision,
    segments       jsonb NOT NULL,       -- [{start_s, end_s, text, speaker}, ...]
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_session_transcripts_tenant ON session_transcripts(tenant_id);
CREATE INDEX idx_session_transcripts_user ON session_transcripts(user_id);
-- If we want full-text search in Postgres:
-- ALTER TABLE session_transcripts ADD COLUMN full_text_tsv tsvector;
-- CREATE INDEX idx_session_transcripts_fts ON session_transcripts USING GIN(full_text_tsv);

CREATE TABLE speakers (
    speaker_id     uuid PRIMARY KEY,
    tenant_id      uuid NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    user_id        uuid REFERENCES app_users(user_id) ON DELETE SET NULL,
    audio_url      text NOT NULL,

    -- e.g. "Dr. Novak", "Miroslav", "Patient A"
    label          text NOT NULL,

    -- optional, for later: global embedding for this speaker (vector extension, etc.)
    -- embedding      vector(256),

    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_speakers_tenant_user ON speakers(tenant_id, user_id);
