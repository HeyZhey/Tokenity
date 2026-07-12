# Tokenity Installer DMG

The installer build script is:

```bash
cd /Users/zxc/Documents/Tokenity
./scripts/package-tokenity-dmg.sh
```

Default outputs:

```text
/Users/zxc/Documents/Tokenity/dist/Tokenity-0.1.0.pkg
/Users/zxc/Documents/Tokenity/dist/Tokenity-0.1.0.dmg
```

## What The Installer Installs

The DMG contains a macOS installer package. The package installs:

- `/Applications/TokenityControl.app`
- `/Users/Shared/TokenityCode`
- `/Users/Shared/TokenityRuntime`
- `/Users/Shared/TokenityModels`
- `/Library/LaunchDaemons/ai.tokenity.node-agent.plist`

The postinstall script starts NodeAgent on port `9100` as a launchd-managed
service for the current console user. This is the supported multi-Mac runtime;
the app and Agents coordinate ranks over HTTP and do not configure SSH.

For the current known Mac A and Mac B LAN addresses, the postinstall script also installs and starts Thunderbolt keepalive:

- Mac A LAN address detected: configures `en4`, `192.168.0.1`, peer `192.168.0.2`
- Mac B LAN address detected: configures `en5`, `192.168.0.2`, peer `192.168.0.1`

Other Macs still get the UI, backend code, runtime, and NodeAgent. They do not get a Thunderbolt keepalive config unless the installer script is extended for their LAN/RDMA layout.

## Runtime Source

The local development machine currently does not have `/Users/Shared/TokenityRuntime`.

By default, `package-tokenity-dmg.sh` uses the local shared runtime or its local
build cache. Override the local source with:

```bash
TOKENITY_RUNTIME_SOURCE=/path/to/TokenityRuntime ./scripts/package-tokenity-dmg.sh
```

Runtime cache and temporary build outputs are under `/Users/zxc/Documents/Tokenity/dist`, which is git-ignored.

## Model Weights

The default DMG does not include `Qwen3.5-122B-A10B-4bit` weights because the model is about `65GB`.

The installed expected model path is still:

```text
/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit
```

To build a very large offline installer that includes model weights:

```bash
TOKENITY_INCLUDE_MODEL=1 ./scripts/package-tokenity-dmg.sh
```

Set `TOKENITY_MODEL_SOURCE` when the weights are in another local directory.

## Signing Note

The app bundle is ad-hoc signed so its resources are sealed locally.

The installer package is not Developer ID signed because no Developer ID Installer certificate is configured on this machine. It is suitable as a local/internal installer, but Gatekeeper signature verification reports:

```text
Status: no signature
```

For distribution outside the local Macs, sign the package with `productsign` using a Developer ID Installer certificate and notarize the final DMG.

## Verification Commands

Check package payload:

```bash
pkgutil --payload-files /Users/zxc/Documents/Tokenity/dist/Tokenity-0.1.0.pkg | egrep 'TokenityControl.app|TokenityCode/tokenity|TokenityRuntime/current|ai.tokenity.node-agent.plist'
```

Mount-check the DMG:

```bash
hdiutil attach -readonly /Users/zxc/Documents/Tokenity/dist/Tokenity-0.1.0.dmg
```

After installing on a Mac:

```bash
curl --noproxy "*" -sS http://127.0.0.1:9100/v1/node/info
```
