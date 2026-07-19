# Contributor agent guidance

- Read `README.MD` and the relevant documents under `docs/` before changing
  behavior.
- Create a dedicated imperative branch from `master` for each task.
- Make small, atomic commits with imperative commit messages.
- Update the relevant Markdown documentation whenever behavior, configuration,
  tests, or public interfaces change.
- Preserve unrelated working-tree changes.
- Treat security review as part of completion. Never commit credentials,
  customer data, recordings, transcripts, or private infrastructure details.
- Bind local test services to `127.0.0.1` or `localhost`; never expose test
  ports on the local network.
- Use `xamurai` for these speech services and `nanosamurai` for the complete
  solution. The internal `drsynth_common` import remains only for compatibility;
  do not introduce new public names using the historical prefix.
