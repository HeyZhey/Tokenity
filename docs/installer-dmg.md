# Tokenity Drag-Install DMG

Build from the Stable source of truth:

```bash
cd /Users/zxc/Documents/Tokenity-Stable
./scripts/package-tokenity-dmg.sh
```

The build always produces:

```text
dist/Tokenity-0.1.0.dmg
dist/Tokenity-0.1.0.dmg.sha256
```

The mounted DMG uses the conventional macOS layout:

```text
TokenityControl.app
Applications -> /Applications
README.txt
```

Users install the control app by dragging `TokenityControl.app` onto
`Applications`.

## Control App And Node Agent

Tokenity separates the unprivileged control app from the privileged inference
service:

- `TokenityControl.app` is always present and is installed by dragging.
- When a runtime source is available, the DMG also contains
  `Install Tokenity Node Agent.pkg`.
- The Node Agent package should be run on every Mac that will execute model
  ranks. It installs `/Users/Shared/TokenityCode`,
  `/Users/Shared/TokenityRuntime`, `/Users/Shared/TokenityModels`, and
  `/Library/LaunchDaemons/ai.tokenity.node-agent.plist`.

The package starts Node Agent on port `9100` as a launchd-managed service for
the current console user. The app and Agents coordinate ranks over typed HTTP;
product startup does not use SSH.

For the current known Mac A and Mac B LAN addresses, the postinstall script also installs and starts Thunderbolt keepalive:

- Mac A LAN address detected: configures `en4`, `192.168.0.1`, peer `192.168.0.2`
- Mac B LAN address detected: configures `en5`, `192.168.0.2`, peer `192.168.0.1`

Other Macs still get the backend code, runtime, and Node Agent. They do not get
a Thunderbolt keepalive configuration unless the installer script is extended
for their LAN/RDMA layout.

## Node Agent Package Modes

`TOKENITY_NODE_AGENT_PACKAGE` accepts:

- `auto` (default): include the package when a runtime source/cache exists;
  otherwise build a controller-only DMG.
- `required`: fail unless the Node Agent runtime can be included.
- `skip`: intentionally build a controller-only DMG.

To require a complete cluster DMG:

```bash
TOKENITY_NODE_AGENT_PACKAGE=required \
TOKENITY_RUNTIME_SOURCE=/path/to/TokenityRuntime \
./scripts/package-tokenity-dmg.sh
```

When included, the additional standalone output is:

```text
dist/Tokenity-NodeAgent-Runtime-0.1.0.pkg
```

Runtime cache and temporary build outputs are under `dist/`, which is
git-ignored.

## Model Weights

The default DMG does not include model weights. Compatible MLX model folders
belong under:

```text
/Users/Shared/TokenityModels
```

To build a very large offline installer that includes model weights:

```bash
TOKENITY_NODE_AGENT_PACKAGE=required \
TOKENITY_INCLUDE_MODEL=1 \
./scripts/package-tokenity-dmg.sh
```

Set `TOKENITY_MODEL_SOURCE` when the weights are in another local directory.

## Signing Note

The app bundle is ad-hoc signed so its resources are sealed locally. The
current build machine has no Developer ID identities configured.

The optional Node Agent package is not Developer ID signed because no Developer
ID Installer certificate is configured on this machine. It is suitable as a
local/internal installer, but Gatekeeper signature verification reports:

```text
Status: no signature
```

For public distribution, sign the app and package with the relevant Developer
ID certificates and notarize the final DMG.

## Verification Commands

Mount-check the DMG and verify the drag-install layout:

```bash
hdiutil attach -readonly -nobrowse \
  /Users/zxc/Documents/Tokenity-Stable/dist/Tokenity-0.1.0.dmg
ls -la "/Volumes/Tokenity 0.1.0"
```

When the Node Agent package is present, check its payload:

```bash
pkgutil --payload-files \
  /Users/zxc/Documents/Tokenity-Stable/dist/Tokenity-NodeAgent-Runtime-0.1.0.pkg \
  | egrep 'TokenityCode/tokenity|TokenityRuntime/current|ai.tokenity.node-agent.plist'
```

After installing on a Mac:

```bash
curl --noproxy "*" -sS http://127.0.0.1:9100/v1/node/info
```
