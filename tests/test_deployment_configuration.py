from __future__ import annotations

import re
from pathlib import Path

import pytest

from tokenity.paths import (
    code_root,
    data_root,
    h3_binary,
    log_root,
    model_root,
    runtime_python,
    runtime_root,
    state_root,
)


PROJECT_ROOT = Path(__file__).parents[1]
AUDITED_ROOTS = (
    PROJECT_ROOT / "tokenity",
    PROJECT_ROOT / "apps" / "TokenityControl" / "Sources",
    PROJECT_ROOT / "apps" / "TokenityControl" / "Tests",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "packaging",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "tests",
)


def test_all_deployment_paths_support_one_root_override():
    environ = {"TOKENITY_DATA_ROOT": "/deployment/tokenity"}
    assert data_root(environ) == Path("/deployment/tokenity")
    assert model_root(environ) == Path("/deployment/tokenity/Models")
    assert runtime_root(environ) == Path("/deployment/tokenity/Runtime")
    assert code_root(environ) == Path("/deployment/tokenity/Code")
    assert state_root(environ) == Path("/deployment/tokenity/State")
    assert log_root(environ) == Path("/deployment/tokenity/Logs")
    assert runtime_python(environ) == Path(
        "/deployment/tokenity/Runtime/current/.venv/bin/python"
    )
    assert h3_binary(environ) == Path(
        "/deployment/tokenity/Runtime/current/bin/mlx-serve"
    )


def test_individual_deployment_path_overrides_take_precedence():
    environ = {
        "TOKENITY_DATA_ROOT": "/deployment/tokenity",
        "TOKENITY_MODEL_ROOT": "~/models",
        "TOKENITY_RUNTIME_PYTHON": "/runtime/python",
        "TOKENITY_H3_BINARY_PATH": "/native/mlx-serve",
    }
    assert model_root(environ) == Path("~/models").expanduser()
    assert runtime_python(environ) == Path("/runtime/python")
    assert h3_binary(environ) == Path("/native/mlx-serve")


@pytest.mark.parametrize(
    ("resolver", "variable"),
    ((data_root, "TOKENITY_DATA_ROOT"), (model_root, "TOKENITY_MODEL_ROOT")),
)
def test_deployment_path_overrides_reject_working_directory_coupling(
    resolver, variable
):
    with pytest.raises(ValueError, match="must be an absolute path"):
        resolver({variable: "relative/deployment/path"})


def test_executable_sources_contain_no_machine_specific_network_or_user_paths():
    private_ipv4 = re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3})\b"
    )
    user_root = re.escape(str(Path("/") / "Users"))
    absolute_user_path = re.compile(user_root + r"/")
    legacy_names = ("z" + "xc", "for" + "codex", "pro" + "briefing")
    legacy_machine_identity = re.compile(
        rf"\b(?:{'|'.join(map(re.escape, legacy_names))})\b", re.I
    )
    failures: list[str] = []
    candidates = [PROJECT_ROOT / "README.md"]
    for root in AUDITED_ROOTS:
        candidates.extend(root.rglob("*"))
    for path in candidates:
        if not path.is_file() or path.suffix not in {
            ".env",
            ".json",
            ".md",
            ".py",
            ".sh",
            ".swift",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, label in (
            (private_ipv4, "private IP"),
            (absolute_user_path, "absolute user path"),
            (legacy_machine_identity, "legacy machine identity"),
        ):
            if match := pattern.search(text):
                failures.append(
                    f"{path.relative_to(PROJECT_ROOT)}: {label} {match.group(0)!r}"
                )
        legacy_install_root = str(Path("/") / "Users" / "Shared" / "Tokenity")
        if legacy_install_root in text:
            failures.append(
                f"{path.relative_to(PROJECT_ROOT)}: legacy fixed install root"
            )
    assert failures == []
