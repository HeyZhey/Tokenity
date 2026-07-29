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
    assert (
        "/Users/*/Library/LaunchAgents/local.tokenity.node-agent.plist(N)"
        in source
    )
    assert '"gui/$legacy_uid/local.tokenity.node-agent"' in source
    assert '/bin/launchctl bootout "gui/$legacy_uid" "$legacy_plist"' in source
    assert '/bin/rm -f "$legacy_plist"' in source
    assert "ai.tokenity.node-agent" in source
