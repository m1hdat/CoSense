# Repository reporting rule

After every material repository change, update `docs/IMPLEMENTATION_REPORT.tex` in the same task.

A material change includes data or split changes, preprocessing/model/metric changes, artifact or submission-contract changes, notebook or Streamlit behavior changes, dependency/environment changes, and new validation results.

For each update:

1. Bump the report version and `Last updated` date when the documented state changes.
2. Update the relevant current-state section; do not leave superseded facts presented as current.
3. Append one dated entry under `Change log` containing scope, files, evidence/tests, generated artifact name/version/size, known limitations, and the next action.
4. Record facts supported by repository state or observed tool output. Mark inferences and unresolved external rules explicitly.
5. Do not include raw private contexts, credentials, tokens, or uncommitted personal paths beyond repository-relative paths.
6. Do not include SHA values or other cryptographic hashes in the report. Keep such values in machine-readable metadata only.
7. Run `git diff --check` and a LaTeX build when a TeX engine is available. If LaTeX is unavailable or blocked, report that limitation rather than claiming a successful PDF build.

Small typo-only edits that do not change behavior may be grouped into the next material report entry.
