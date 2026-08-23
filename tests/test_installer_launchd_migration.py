import subprocess
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
    assert '--component "$APP_BUNDLE"' in source
    assert '--package "$PKG_PATH"' in source
    assert 'Applications -> /Applications' not in source
    assert 'Install Tokenity Node Agent.pkg' not in source
    assert 'tokenity-runtime-manifest.py" verify' in source
    assert 'H3_BINARY="$RUNTIME_ROOT/current/bin/mlx-serve"' in source


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
