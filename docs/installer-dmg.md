# Tokenity Installer and Runtime Distribution

Tokenity ships one user-facing installer, `Install Tokenity.pkg`. It installs
`Tokenity.app`, the privileged Node Agent and watchdog, the pinned Python/MLX
Runtime, JACCL, and the native MiniMax H3 runtime. The component package is
also embedded in the app so **Install or Repair** can reopen the same verified
payload later. Model weights remain separate.

The application never uses SSH to install software on another Mac. A new Mac
does not yet have a trusted Node Agent, so its first privileged installation
must be approved locally by an administrator.

## Runtime identity

The release lock is:

```text
packaging/runtime/runtime-lock.json
```

Runtime `2026.09.15.2` pins:

| Component | Value |
| --- | --- |
| Platform | macOS / arm64 |
| Minimum macOS | 26.2 |
| CPython | 3.12.13 |
| MLX | 0.32.0 |
| MLX-LM | 0.31.3 |
| FastAPI | 0.139.0 |
| Uvicorn | 0.50.2 |
| Native MiniMax H3 protocol | 1 |
| H3 Turbo protocol / presets | 1 / 4, 6, 8 steps |
| Isolated MLX-VLM / MLX | 0.7.1 / 0.32.2 |
| VLM Transformers | 5.16.1 |

The macOS requirement is not an arbitrary app setting. The validated
`mlx/core`, `libmlx.dylib`, and `libjaccl.dylib` binaries report `minos 26.2`.
Supporting macOS 14 requires rebuilding those binaries with a lower deployment
target and validating them on that OS.

## Importing the validated Runtime

The importer accepts local paths or rsync-style SSH sources. It copies both
sources into isolated staging directories, removes machine-specific editable
installs and caches, normalizes symlinks and entry-point shebangs, validates
the pinned versions and arm64 Mach-O files, and compares deterministic tree
identities.

For a two-node cluster:

```bash
cd /path/to/Tokenity-Stable
./scripts/import-tokenity-runtime.sh \
  user@node-a.local:/path/to/TokenityRuntime \
  user@node-b.local:/path/to/TokenityRuntime
```

Only one canonical copy is retained at:

```text
dist/runtime-cache/TokenityRuntime
```

Mac A and Mac B are verification peers; their directories are never merged.
The normalized Runtime must also contain the arm64 native H3 executable at
`current/bin/mlx-serve` and its relocatable libraries below `current/lib`.
`runtime-manifest.json` pins its SHA-256, minimum macOS, and distributed
protocol in addition to the deterministic whole-tree SHA-256. The manifest
excludes itself from the tree digest.

## Building the full offline DMG

After importing the Runtime:

```bash
./scripts/package-tokenity-dmg.sh
```

The default mode is `required`. A release build fails instead of silently
producing a controller-only image when the Runtime is absent or invalid.

Outputs:

```text
dist/Tokenity-0.1.2.dmg
dist/Tokenity-0.1.2.dmg.sha256
dist/Tokenity-0.1.2-macos-arm64.pkg
dist/Tokenity-0.1.2-macos-arm64.pkg.sha256
dist/Tokenity-NodeAgent-Runtime-2026.09.15.2-macos-arm64.pkg
dist/Tokenity-NodeAgent-Runtime-2026.09.15.2-macos-arm64.pkg.sha256
dist/Tokenity-RuntimeCatalog-2026.09.15.2.json
```

Mounted layout:

```text
Install Tokenity.pkg
Runtime Catalog.json
Tokenity Installer.sha256
README.txt
```

`Install Tokenity.pkg` is a distribution package containing both the app and
the system Runtime component. Users do not need to drag an app or locate a
second installer.

## First installation

On every Mac that will execute models:

1. Download the full `Tokenity-0.1.2-macos-arm64.pkg`, or open the full DMG.
2. Open the PKG (`Install Tokenity.pkg` inside the DMG).
3. Approve the installation in Installer.app.
4. Place compatible MLX model folders under
   `/Library/Tokenity/Models`, or select a model folder in the app.
5. Open Tokenity and confirm the component readiness checks pass.

The published v0.1.2 PKG is Developer ID signed and Apple notarized, with its
notarization ticket stapled. The app is always installed at `/Applications/Tokenity.app`, even
when another copy has been moved elsewhere. Xcode, Homebrew and a separate
Python installation are not required.

The first-launch guide locates and verifies the embedded package. It never
invokes `sudo` or bypasses Installer.app.

## Online Runtime fallback

The packaged app contains a fixed `RuntimeCatalog.json`. It records the exact:

- Runtime ID and compatibility requirements
- package filename and package identifier
- byte size
- SHA-256
- HTTPS release URL

If the package is not embedded, Tokenity downloads that exact artifact to its
user cache, verifies size and SHA-256, and only then opens Installer.app.
Tokenity does not use a mutable `latest` URL.

The catalog currently points to:

```text
https://github.com/HeyZhey/Tokenity/releases/download/runtime-2026.09.15.2/
```

Before distributing a thin/controller-only build, publish the pkg at the URL
recorded in the catalog. The offline DMG does not depend on that URL.

`TOKENITY_RUNTIME_DOWNLOAD_BASE_URL` can select another immutable HTTPS
location when packaging:

```bash
TOKENITY_RUNTIME_DOWNLOAD_BASE_URL=https://downloads.example.com/tokenity/runtime-2026.09.15.2 \
./scripts/package-tokenity-dmg.sh
```

The online and offline paths must use the same pkg bytes and SHA-256.

## Package modes

`TOKENITY_NODE_AGENT_PACKAGE` accepts:

- `required` (default): build a full release DMG or fail.
- `auto`: include the package when a valid Runtime exists.
- `skip`: intentionally build a thin controller-only DMG.

Use `skip` only when the immutable catalog artifact has already been published.

## What the Node Agent package installs

```text
${TOKENITY_CODE_ROOT}
${TOKENITY_RUNTIME_ROOT}
${TOKENITY_MODEL_ROOT}
/Library/LaunchDaemons/ai.tokenity.node-agent.plist
/Library/LaunchDaemons/ai.tokenity.node-agent-watchdog.plist
/usr/local/bin/tokenity-uninstall-node-agent
```

The preinstall script:

- rejects Intel Macs
- rejects macOS older than the Runtime minimum
- checks free disk space
- asks an existing Agent to stop model ranks
- refuses to overwrite a still-running inference process
- publishes a bounded maintenance window and stops the watchdog before updating

The postinstall script:

- makes the model directory writable by the console user
- imports MLX, MLX-LM and Tokenity
- initializes a small MLX array
- executes the native H3 binary and verifies distributed protocol 1
- starts the launchd-managed Node Agent on port `9100`
- verifies `/v1/node/health`, then starts the independent watchdog on `9101`

The watchdog probes only `127.0.0.1:9100/v1/node/health`. Three consecutive
failures are required before a controlled `launchctl kickstart`; restart
attempts use exponential backoff, a sliding-window limit, and a circuit-open
state. LAN failure observed only by TokenityControl cannot trigger this local
restart path. Agent instance journals live under
`${TOKENITY_STATE_ROOT}/instances` and are used to fence restart adoption
by PID start time, instance/operation identity, model revision, rank topology,
and runtime heartbeat.

Upgrades create `${TOKENITY_STATE_ROOT}/maintenance.json` before stopping
jobs, preventing watchdog/update restart races. The marker expires if an
installer aborts and is removed only after the new Agent health contract is
verified.

To remove the system jobs while preserving all model files:

```bash
sudo /usr/local/bin/tokenity-uninstall-node-agent
```

The uninstaller refuses to proceed while an inference process remains active,
and removes both the Agent and watchdog LaunchDaemons so no duplicate legacy
job is left behind.

## Model weights

Model weights are not included by default. To build a very large offline image:

```bash
TOKENITY_INCLUDE_MODEL=1 \
TOKENITY_MODEL_SOURCE=/path/to/model \
./scripts/package-tokenity-dmg.sh
```

## Signed v0.1.2 release

Download `Tokenity-0.1.2-macos-arm64.pkg` and its `.sha256` file from the
[v0.1.2 Release](https://github.com/HeyZhey/Tokenity/releases/tag/v0.1.2).
The release app/runtime use Developer ID Application signatures and the PKG
uses Developer ID Installer signing. Apple notarization is accepted and its
ticket is stapled to the installer. Users do not need an Apple Developer account.

The final package passed a standard installation on a separate M5 Ultra with
256 GiB memory and macOS 27.0, including Gatekeeper checks with a download
quarantine attribute, ordinary-user permissions, LM/VLM short inference, and
H3 startup/Turbo readiness. No installed-payload permission repair was needed.
See [release notes](release-0.1.2.md) for the precise test scope.

## Unsigned development builds

`scripts/package-tokenity-dmg.sh` produces an unsigned development package
with an ad-hoc-signed app. Developer ID signing and notarization are separate
release steps; running the source script alone does not produce the published
notarized artifact. The following instructions apply only to such unsigned builds.

A browser download may trigger Gatekeeper. On a Mac where local policy allows
exceptions:

1. Try opening the downloaded PKG once.
2. Open **System Settings → Privacy & Security** and choose **Open Anyway**
   for that installer (on some macOS versions, choose **Open** first).
3. Authenticate and reopen the PKG. After installation, launch Tokenity from
   Applications. If macOS also blocks the app, repeat the same per-app approval.

This is Apple's documented [unknown-developer opening procedure](https://support.apple.com/guide/mac-help/mh40616/mac).
It does not provide a guarantee of zero prompts on every Mac; organization
policies may prohibit unsigned software. The package does not change Gatekeeper
or SIP settings.

The packaging approach uses Apple's `pkgbuild` / `productbuild`, as do
[Munki's packaging tools](https://github.com/munki/munki-pkg/blob/main/README.md).
Munki makes signing and notarization optional (`--skip-signing` and
`--skip-notarization`). This is a packaging option, not a Gatekeeper exemption.
No additional packaging framework is needed for Tokenity.

Keep the `.sha256` file beside the downloaded package, then verify:

```bash
shasum -a 256 -c Tokenity-0.1.2-macos-arm64.pkg.sha256
```

For the renamed PKG inside the DMG, run this from the mounted image:

```bash
shasum -a 256 -c "Tokenity Installer.sha256"
```

SHA-256 detects corruption; it does not replace publisher identity signing.
`pkgutil --check-signature` reporting "no signature" is expected only for an
unsigned development build. The published release reports a trusted Developer ID
signature and Apple notarization. The installed app must also pass
`codesign --verify --deep --strict`.

## Verification

```bash
hdiutil verify dist/Tokenity-0.1.2.dmg
./scripts/tokenity-runtime-manifest.py verify-artifact \
  --catalog dist/Tokenity-RuntimeCatalog-2026.09.15.2.json \
  --artifact dist/Tokenity-NodeAgent-Runtime-2026.09.15.2-macos-arm64.pkg
pkgutil --check-signature \
  dist/Tokenity-0.1.2-macos-arm64.pkg
pkgutil --payload-files \
  dist/Tokenity-NodeAgent-Runtime-2026.09.15.2-macos-arm64.pkg
```

After installing on a compatible test Mac:

```bash
curl --noproxy "*" -sS http://127.0.0.1:9100/v1/node/info
```
