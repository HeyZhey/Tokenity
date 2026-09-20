import plistlib
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_macos_version_check_works_with_system_awk():
    source = (ROOT / "scripts/package-tokenity-dmg.sh").read_text(encoding="utf-8")
    start = source.index("version_at_least() {")
    end = source.index("\n}\n\nif [[", start) + 3
    function = source[start:end]

    subprocess.run(
        [
            "/bin/zsh",
            "-c",
            "\n".join(
                [
                    function,
                    "version_at_least 26.5.1 26.2",
                    "! version_at_least 26.1 26.2",
                    "version_at_least 26.2 26.2",
                ]
            ),
        ],
        check=True,
    )


def test_dmg_exposes_one_installer_with_app_and_runtime_components():
    source = (ROOT / "scripts/package-tokenity-dmg.sh").read_text(encoding="utf-8")

    assert 'TOKENITY_BUNDLE_NAME=Tokenity' in source
    assert 'cp "$INSTALLER_PKG" "$DMG_ROOT/Install Tokenity.pkg"' in source
    assert "productbuild --synthesize" in source
    assert '--component-plist "$APP_COMPONENT_PLIST"' in source
    assert '--package "$PKG_PATH"' in source
    assert 'Applications -> /Applications' not in source
    assert 'Install Tokenity Node Agent.pkg' not in source
    assert 'tokenity-runtime-manifest.py" verify' in source
    assert 'H3_BINARY="$RUNTIME_ROOT/current/bin/mlx-serve"' in source


def test_runtime_upgrade_uses_verified_staging_and_rollback_paths():
    source = (ROOT / "scripts/package-tokenity-dmg.sh").read_text(encoding="utf-8")

    assert 'RUNTIME_STAGE_ROOT="$INSTALL_ROOT/.Runtime.installing"' in source
    assert 'RUNTIME_BACKUP_ROOT="$INSTALL_ROOT/.Runtime.previous"' in source
    assert 'copy_runtime_dir "$RUNTIME_CACHE/" "$PAYLOAD_DIR$RUNTIME_STAGE_ROOT"' in source
    assert '"$RUNTIME_STAGE_ROOT"' in source
    assert '--manifest "$RUNTIME_STAGE_ROOT/runtime-manifest.json"' in source
    assert '/bin/mv "$RUNTIME_ROOT" "$RUNTIME_BACKUP_ROOT"' in source
    assert '/bin/mv "$RUNTIME_STAGE_ROOT" "$RUNTIME_ROOT"' in source
    assert "protect the live Runtime before PackageKit processes its old receipt" in source
    assert "rollback_runtime_install()" in source
    assert 'local exit_code="$?"' in source
    assert 'trap rollback_runtime_install EXIT' in source


def test_app_installer_removes_only_the_known_legacy_bundle_name():
    source = (ROOT / "scripts/package-tokenity-dmg.sh").read_text(encoding="utf-8")

    assert "^/Applications/Tokenity\\.app/Contents/MacOS/TokenityControl" in source
    assert 'LEGACY_APP="/Applications/TokenityControl.app"' in source
    assert "Print :CFBundleIdentifier" in source
    assert '"$legacy_bundle_id" == "ai.tokenity.control"' in source
    assert '/bin/rm -rf "$LEGACY_APP"' in source
    assert '--scripts "$APP_SCRIPTS_DIR"' in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/package-tokenity-dmg.sh",
        "scripts/package-tokenity-agent-update.sh",
    ],
)
def test_installer_migrates_legacy_user_node_agent(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "remove_legacy_user_agents()" in source
    assert source.count("remove_legacy_user_agents") >= 2
    assert "/usr/bin/dscacheutil -q user" in source
    assert "for legacy_label in local.tokenity.node-agent dev.tokenity.dns-sd" in source
    assert 'legacy_plist="$home_directory/Library/LaunchAgents/' in source
    assert '"gui/$legacy_uid/$legacy_label"' in source
    assert '/bin/launchctl bootout "gui/$legacy_uid" "$legacy_plist"' in source
    assert '/bin/rm -f "$legacy_plist"' in source
    assert "ai.tokenity.node-agent" in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/package-tokenity-dmg.sh",
        "scripts/package-tokenity-agent-update.sh",
    ],
)
def test_installer_packages_watchdog_state_and_maintenance_fencing(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert "ai.tokenity.node-agent-watchdog.plist" in source
    assert "tokenity.node_agent.watchdog" in source
    assert "TOKENITY_INSTANCE_STATE_ROOT" in source
    assert "TOKENITY_STATE_ROOT" in source or 'STATE_ROOT="$INSTALL_ROOT/State"' in source
    assert "$STATE_ROOT/instances" in source
    assert '"reason":"' in source
    assert "maintenance.json" in source
    assert '/bin/launchctl bootout system "$WATCHDOG_PLIST"' in source
    assert "http://127.0.0.1:9100" in source
    assert "/v1/node/health" in source
    assert "http://127.0.0.1:9101/v1/watchdog/status" in source
    assert "tokenity-uninstall-node-agent" in source


def test_uninstaller_preserves_models_and_removes_all_system_jobs():
    source = (ROOT / "scripts/uninstall-tokenity-node-agent.sh").read_text(
        encoding="utf-8"
    )

    assert 'WATCHDOG_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent-watchdog.plist"' in source
    assert 'AGENT_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent.plist"' in source
    assert 'TB_PLIST="/Library/LaunchDaemons/ai.tokenity.thunderbolt-keepalive.plist"' in source
    assert '/bin/launchctl bootout system "$WATCHDOG_PLIST"' in source
    assert '/bin/launchctl bootout system "$AGENT_PLIST"' in source
    assert '/bin/launchctl bootout system "$TB_PLIST"' in source
    assert '"$STATE_ROOT/rdma-reset-request"' in source
    assert "TOKENITY_MODEL_ROOT" in source
    assert "MODEL_ROOT" in source
    assert "preserved" in source
    assert '/bin/rm -rf "$MODEL_ROOT"' not in source


def test_node_package_resets_rdma_after_distributed_h3_stop():
    source = (ROOT / "scripts/package-tokenity-dmg.sh").read_text(encoding="utf-8")

    assert "TOKENITY_RDMA_RESET_REQUEST_PATH" in source
    assert 'reset_request="$state_root/rdma-reset-request"' in source
    assert 'reset_link workload-transition' in source
    assert '/bin/rm -f "$reset_request"' in source


def test_agent_update_revision_covers_every_runtime_entrypoint():
    source = (ROOT / "scripts/package-tokenity-agent-update.sh").read_text(
        encoding="utf-8"
    )

    assert '"agent.py"' in source
    assert '"distributed_openai.py"' in source
    assert '"minimax_h3_video.py"' in source


def test_postinstall_uses_embedded_python_without_command_line_tools():
    source = (ROOT / "scripts/package-tokenity-dmg.sh").read_text()
    start = source.index('RUNTIME_EXPECTED_H3_PROTOCOL=')
    end = source.index('} > "$SCRIPTS_DIR/postinstall"', start)
    postinstall = source[start:end]
    assert '\n/usr/bin/python3 "$CODE_ROOT/' not in postinstall
    assert '"$RUNTIME_STAGE_ROOT/current/.venv/bin/python" "$CODE_ROOT/scripts/tokenity-runtime-manifest.py" verify' in postinstall


def test_product_checks_architecture_and_os_before_installing_components():
    source = (ROOT / "scripts/package-tokenity-dmg.sh").read_text()
    assert '--product "$REQUIREMENTS_PLIST"' in source
    assert '<key>arch</key><array><string>arm64</string></array>' in source
    assert '<key>os</key><array><string>$RUNTIME_MINIMUM_MACOS</string></array>' in source


def test_app_package_cannot_relocate_to_a_development_copy(tmp_path):
    source = (ROOT / "scripts/package-tokenity-dmg.sh").read_text()
    start = source.index('APP_COMPONENT_PLIST="$WORK_DIR/AppComponent.plist"')
    end = source.index('\nif [[ "$BUILD_NODE_AGENT_PACKAGE"', start)
    app = tmp_path / "app/Tokenity.app"
    (app / "Contents").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (app / "Contents/Info.plist").write_bytes(plistlib.dumps({
        "CFBundleIdentifier": "ai.tokenity.control",
        "CFBundleName": "Tokenity",
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "4",
        "CFBundleShortVersionString": "0.1.2",
    }))
    subprocess.run([
        "/bin/bash", "-eu", "-c",
        'WORK_DIR="$1"\nAPP_BUNDLE="$1/app/Tokenity.app"\n'
        'APP_SCRIPTS_DIR="$1/scripts"\nAPP_COMPONENT_PKG="$1/app.pkg"\n'
        'VERSION=0.1.2\n' + source[start:end],
        "test-package", str(tmp_path),
    ], check=True, capture_output=True, text=True)
    expanded = tmp_path / "expanded"
    subprocess.run(["/usr/sbin/pkgutil", "--expand", str(tmp_path / "app.pkg"),
                    str(expanded)], check=True, capture_output=True)
    info = ET.parse(expanded / "PackageInfo").getroot()
    assert info.get("install-location") == "/Applications"
    assert info.get("relocatable") == "false"
    assert info.findall("relocate/bundle") == []
    assert info.find("bundle").get("path").removeprefix("./") == "Tokenity.app"
