from __future__ import annotations

from tokenity.cli import main


def test_version(capsys):
    assert main(["--version"]) == 0
    assert "tokenity 0.1.0" in capsys.readouterr().out

