# Tokenity Installer and Runtime Distribution

Tokenity ships two components:

- `TokenityControl.app`, installed by dragging it to Applications.
- A privileged Node Agent and MLX Runtime package, installed locally with
  Installer.app on every Mac that will execute model ranks.

The application never uses SSH to install software on another Mac. A new Mac
does not yet have a trusted Node Agent, so its first privileged installation
must be approved locally by an administrator.

## Runtime identity

The release lock is:

```text
packaging/runtime/runtime-lock.json
```

Runtime `2026.07.26.1` pins:

| Component | Value |
| --- | --- |
| Platform | macOS / arm64 |
| Minimum macOS | 26.2 |
| CPython | 3.12.13 |
| MLX | 0.32.0 |
| MLX-LM | 0.31.3 |
| FastAPI | 0.139.0 |
| Uvicorn | 0.50.2 |

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

For the validated Mango/Kiwi cluster:

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
The normalized Runtime contains `runtime-manifest.json`, including its
deterministic tree SHA-256. The manifest excludes itself from the tree digest.

## Building the full offline DMG

After importing the Runtime:

```bash
./scripts/package-tokenity-dmg.sh
```

The default mode is `required`. A release build fails instead of silently
producing a controller-only image when the Runtime is absent or invalid.

Outputs:

```text
dist/Tokenity-0.1.0.dmg
dist/Tokenity-0.1.0.dmg.sha256
dist/Tokenity-NodeAgent-Runtime-2026.07.26.1-macos-arm64.pkg
dist/Tokenity-NodeAgent-Runtime-2026.07.26.1-macos-arm64.pkg.sha256
dist/Tokenity-RuntimeCatalog-2026.07.26.1.json
```

Mounted layout:

```text
TokenityControl.app
Applications -> /Applications
Install Tokenity Node Agent.pkg
Runtime Catalog.json
Runtime Installer.sha256
README.txt
```

The visible pkg is a relative symlink to the exact same pkg embedded under the
app's `Contents/Resources`. This avoids duplicate payload bytes and means the
installer is still available after the app is copied to Applications.

## First installation

On every Mac that will execute models:

1. Open the full DMG.
2. Drag `TokenityControl.app` to Applications on the controller Mac.
3. Open `Install Tokenity Node Agent.pkg`.
4. Approve the installation in Installer.app.
5. Place compatible MLX model folders under
   `${TOKENITY_MODEL_ROOT}`.
6. Open Tokenity and confirm each Node Agent is reachable on port `9100`.

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
https://github.com/HeyZhey/Tokenity/releases/download/runtime-2026.07.26.1/
```

Before distributing a thin/controller-only build, publish the pkg at the URL
recorded in the catalog. The offline DMG does not depend on that URL.

`TOKENITY_RUNTIME_DOWNLOAD_BASE_URL` can select another immutable HTTPS
location when packaging:

```bash
TOKENITY_RUNTIME_DOWNLOAD_BASE_URL=https://downloads.example.com/tokenity/runtime-2026.07.26.1 \
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

## Signing status

The current app is ad-hoc signed and the pkg is unsigned. This is suitable for
internal validation only. Public distribution still requires:

- Developer ID Application signing
- Developer ID Installer signing
- notarization and stapling of the final DMG

SHA-256 detects corruption; it does not replace publisher identity signing.

## Verification

```bash
hdiutil verify dist/Tokenity-0.1.0.dmg
./scripts/tokenity-runtime-manifest.py verify-artifact \
  --catalog dist/Tokenity-RuntimeCatalog-2026.07.26.1.json \
  --artifact dist/Tokenity-NodeAgent-Runtime-2026.07.26.1-macos-arm64.pkg
pkgutil --check-signature \
  dist/Tokenity-NodeAgent-Runtime-2026.07.26.1-macos-arm64.pkg
pkgutil --payload-files \
  dist/Tokenity-NodeAgent-Runtime-2026.07.26.1-macos-arm64.pkg
```

After installing on a compatible test Mac:

```bash
curl --noproxy "*" -sS http://127.0.0.1:9100/v1/node/info
```
