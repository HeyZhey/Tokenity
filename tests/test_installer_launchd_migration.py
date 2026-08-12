from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


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
    assert 'legacy_plist="$home_directory/Library/LaunchAgents/' in source
    assert '"gui/$legacy_uid/local.tokenity.node-agent"' in source
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


def test_uninstaller_preserves_models_and_removes_both_system_jobs():
    source = (ROOT / "scripts/uninstall-tokenity-node-agent.sh").read_text(
        encoding="utf-8"
    )

    assert 'WATCHDOG_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent-watchdog.plist"' in source
    assert 'AGENT_PLIST="/Library/LaunchDaemons/ai.tokenity.node-agent.plist"' in source
    assert '/bin/launchctl bootout system "$WATCHDOG_PLIST"' in source
    assert '/bin/launchctl bootout system "$AGENT_PLIST"' in source
    assert "TOKENITY_MODEL_ROOT" in source
    assert "MODEL_ROOT" in source
    assert "preserved" in source
    assert '/bin/rm -rf "$MODEL_ROOT"' not in source
