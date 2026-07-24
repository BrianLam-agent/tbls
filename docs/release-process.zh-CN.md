[English](./release-process.md) | 简体中文

# 发布流程

本文档描述 `tbls` 的版本管理、变更日志生成方式，以及推送发布标签时运行的自动化流水线。面向维护者--推送 `v*` 标签前请先阅读。

## 版本管理

`tbls` 采用[语义化版本](https://semver.org/)，标签形如 `vMAJOR.MINOR.PATCH[-PRERELEASE]`，例如：

- `v0.1.0-alpha.1`——1.0 之前，alpha 预发布（最初的计划路线，已弃用，见下文）。
- `v0.1.0`--1.0 之前的稳定点版本。
- `v1.0.0`--首个稳定 API。

版本号在且仅在两处声明，且必须始终一致：

1. `pyproject.toml` 的 `[project] version = "..."`（无 `v` 前缀）。
2. `src/tbls/__init__.py` 的 `__version__ = "..."`（无 `v` 前缀）。

发布工作流的 `validate` 作业在 `pyproject.toml` 版本与所推标签（去掉 `v` 前缀）不一致时**令发布失败**--这是硬性关卡，而非 lint 警告。

## 变更日志：Conventional Commits + `git-cliff`

变更日志由 [`git-cliff`](https://git-cliff.org/) 从 git 历史自动生成，配置见 `cliff.toml`。它按 [Conventional Commits](https://www.conventionalcommits.org/) 前缀对提交分组：

| 前缀 | 变更日志分区 |
|---|---|
| `feat` | Features |
| `fix` | Bug fixes |
| `perf` | Performance |
| `refactor` | Refactoring |
| `docs` | Documentation |
| `test` | Tests |
| `build`、`ci` | Build and CI |
| `chore` | Maintenance |
| `revert` | Reverts |

不合规范的提交会被静默排除在变更日志之外（`cliff.toml` 中 `filter_unconventional = true`）--这正是要在 `master` 上切实遵循 Conventional Commits 的有力理由（见 [`development.md` 第 3 节](./development.zh-CN.md)），而非仅出于风格偏好。

无需任何 CI，即可在本地预览未发布提交的变更日志：

```bash
uvx --from git-cliff git-cliff --config cliff.toml --unreleased
```

## 由标签触发的发布流水线

向 `origin` 推送匹配 `v*` 的标签会触发 `.github/workflows/release.yml`，依次执行：

```
validate  ->  verify (= ci.yml)  ->  build  ->  changelog  ->  publish
```

1. **`validate`**：检查标签是否为良构的 SemVer，解析到触发推送的提交，且该提交可由 `master` 到达（防止给游离分支/PR 提交打标签）。若 `pyproject.toml` 版本与标签不符，则令整个流水线失败。
2. **`verify`**：重跑整套 CI 套件（`.github/workflows/ci.yml`，作为可复用工作流调用）--lint、`mypy`、完整测试矩阵（Python 3.10–3.13）、以及 build/`twine check` 冒烟测试。发布绝不跳过 CI，即便推送的开发者想必已在本地跑过。
3. **`build`**：`uv build` 产出 wheel 与 sdist，`twine check` 校验，二者作为名为 `dist` 的工作流产物上传。
4. **`changelog`**（`.github/workflows/changelog.yml`，可复用）：下载 `dist` 产物，运行 `git-cliff --latest` 仅渲染*本次发布*的变更日志段落，附加产物清单（文件名 + SHA-256 哈希）与源提交哈希，并经 `gh release create/edit ... dist/*` 为该标签发布（或重跑时更新）GitHub Release--**wheel 与 sdist 作为发布资产附挂**。含预发布后缀（`-alpha.1`、`-rc.1` 等）的标签会自动标记为 GitHub 预发布。
5. **`publish`**：使用 [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) 将构建好的 `dist/*` 发布至 PyPI（OIDC--`id-token: write` 权限、`environment: pypi`、不存储 API token）。

若 `publish` 之前的任一作业失败，则不会向 PyPI 发布任何内容，也不会以该标签创建/更新 GitHub Release--流水线刻意如此排序，使 PyPI 成为最后一步对外可见、最难撤销的步骤。

## 一次性配置（本仓库已完成）

- **PyPI Trusted Publisher**：已在 pypi.org 上为 `tbls` 项目配置，指向 `BrianLam-agent/tbls` + 工作流文件 `.github/workflows/release.yml` + 环境 `pypi`。若你重命名仓库、工作流文件或环境，须在 PyPI 上同步更新 Trusted Publisher 配置，否则 `publish` 将鉴权失败。
- **名为 `pypi` 的 GitHub 环境**（被 `publish` 作业引用）--若尚不存在，在仓库 Settings -> Environments 下创建；如需在发布前设置人工审批关卡，可在该环境添加必需审批者。

## 发布一个版本（维护者清单）

1. 确认 `master` 为绿（`ci.yml` 通过）。
2. 在 `pyproject.toml` 与 `src/tbls/__init__.py` **两处**将版本号升至同一值，提交（`chore(release): bump version to X.Y.Z`），推送至 `master`。
3. 为该提交打标签并推送：
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```
4. 在 Actions 标签页观察 `Release` 工作流运行。成功后：`vX.Y.Z` 处存在一个 GitHub Release，附有变更日志、wheel 与 sdist，且版本已上线 PyPI（`pip install tbls==X.Y.Z`）。
5. 若 `validate` 或 `verify` 失败，删除标签（`git push --delete origin vX.Y.Z && git tag -d vX.Y.Z`），在 `master` 上修复问题后重新打标签--此时尚未发布任何对外内容。

## 初始版本

本包首个版本为 `v0.1.0`（1.0 之前的点发布；原计划的 `v0.1.0-alpha.1` 预发布已改路为点发布）。依据 SemVer 的 1.0 前约定，后续 `0.x` 版本在次版本间仍可能含破坏性变更；无论核心包版本如何，`tbls.genoptim`/`tbls.ensemble` 始终为实验性（见 [`experimental-modules.md`](./experimental-modules.zh-CN.md)）。
