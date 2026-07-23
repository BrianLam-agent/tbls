English | [简体中文](./release-process.zh-CN.md)

# Release process

This describes how `tbls` is versioned, how the changelog is generated, and
the exact automated pipeline that runs when a release tag is pushed. It is
maintainer-facing — read this before pushing a `v*` tag.

## Versioning

`tbls` uses [Semantic Versioning](https://semver.org/) with tags of the form
`vMAJOR.MINOR.PATCH[-PRERELEASE]`, for example:

- `v0.1.0-alpha.1` — pre-1.0, alpha pre-release (the initial release).
- `v0.1.0` — pre-1.0 stable point release.
- `v1.0.0` — first stable API.

The version is declared in exactly two places, which must always agree:

1. `pyproject.toml`'s `[project] version = "..."` (no `v` prefix).
2. `src/tbls/__init__.py`'s `__version__ = "..."` (no `v` prefix).

The release workflow's `validate` job **fails the release** if
`pyproject.toml`'s version does not match the pushed tag (stripped of its `v`
prefix) — this is a hard gate, not a lint warning.

## Changelog: Conventional Commits + `git-cliff`

The changelog is generated automatically from git history by
[`git-cliff`](https://git-cliff.org/), configured in `cliff.toml`. It groups
commits by their [Conventional Commits](https://www.conventionalcommits.org/)
prefix:

| Prefix | Changelog section |
|---|---|
| `feat` | Features |
| `fix` | Bug fixes |
| `perf` | Performance |
| `refactor` | Refactoring |
| `docs` | Documentation |
| `test` | Tests |
| `build`, `ci` | Build and CI |
| `chore` | Maintenance |
| `revert` | Reverts |

Commits that don't follow this convention are silently excluded from the
changelog (`filter_unconventional = true` in `cliff.toml`) — this is a strong
reason to actually follow Conventional Commits on `master` (see
[`development.md` §3](./development.md#3-project-conventions)), not just a
style preference.

You can preview the changelog for unreleased commits locally without any CI:

```bash
uvx --from git-cliff git-cliff --config cliff.toml --unreleased
```

## The tag-triggered release pipeline

Pushing a tag matching `v*` to `origin` triggers
`.github/workflows/release.yml`, which runs, in order:

```
validate  →  verify (= ci.yml)  →  build  →  changelog  →  publish
```

1. **`validate`**: checks the tag is well-formed SemVer, resolves to the
   commit that triggered the push, and is reachable from `master` (guards
   against tagging a stray branch/PR commit). Fails the whole pipeline if
   `pyproject.toml`'s version doesn't match the tag.
2. **`verify`**: re-runs the entire CI suite (`.github/workflows/ci.yml`,
   invoked as a reusable workflow) — lint, `mypy`, the full test matrix
   (Python 3.10–3.13), and a build/`twine check` smoke test. A release never
   skips CI, even though the pushing developer presumably already ran it
   locally.
3. **`build`**: `uv build` produces the wheel and sdist, `twine check`
   validates them, both are uploaded as a workflow artifact named `dist`.
4. **`changelog`** (`.github/workflows/changelog.yml`, reusable): downloads
   the `dist` artifact, runs `git-cliff --latest` to render *only this
   release's* changelog section, appends a distribution-artifact list
   (filenames + SHA-256 hashes) and the source commit hash, and publishes (or
   updates, if re-run) a GitHub Release for the tag — **with the wheel and
   sdist attached as release assets** — via `gh release create/edit ... dist/*`.
   A tag containing a pre-release suffix (`-alpha.1`, `-rc.1`, ...) is marked
   as a GitHub pre-release automatically.
5. **`publish`**: publishes the built `dist/*` to PyPI using
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC —
   `id-token: write` permission, `environment: pypi`, no stored API token).

If any job before `publish` fails, nothing is published to PyPI and no GitHub
Release is created/updated with that tag — the pipeline is deliberately
ordered so PyPI is the *last* externally-visible, hardest-to-undo step.

## One-time setup (already done for this repository)

- **PyPI Trusted Publisher**: configured on pypi.org for the `tbls` project,
  pointing at `BrianLam-agent/tbls` + workflow file
  `.github/workflows/release.yml` + environment `pypi`. If you ever rename
  the repository, workflow file, or environment, update the Trusted
  Publisher configuration on PyPI to match, or `publish` will fail
  authentication.
- **GitHub environment** named `pypi` (referenced by the `publish` job) —
  create it under repository Settings → Environments if it doesn't already
  exist; add required reviewers there if you want a manual approval gate
  before publishing.

## Cutting a release (maintainer checklist)

1. Ensure `master` is green (`ci.yml` passing).
2. Bump the version in **both** `pyproject.toml` and `src/tbls/__init__.py`
   to the same value, commit (`chore(release): bump version to X.Y.Z`), push
   to `master`.
3. Tag the commit and push the tag:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```
4. Watch the `Release` workflow run in the Actions tab. On success: a GitHub
   Release exists at `vX.Y.Z` with the changelog, wheel, and sdist attached,
   and the version is live on PyPI (`pip install tbls==X.Y.Z`).
5. If `validate` or `verify` fails, delete the tag (`git push --delete origin
   vX.Y.Z && git tag -d vX.Y.Z`), fix the issue on `master`, and re-tag —
   nothing external has been published yet at that point.

## Initial release

The first release of this package is planned as `v0.1.0-alpha.1` (pre-release,
signaling "just published, API may still shift before `v0.1.0`"). Subsequent
`0.x` releases may still contain breaking changes between minor versions per
SemVer's pre-1.0 convention; `tbls.genoptim`/`tbls.ensemble` remain
experimental regardless of the core package's version (see
[`experimental-modules.md`](./experimental-modules.md)).
