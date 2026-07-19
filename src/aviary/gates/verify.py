"""Verifier framework. Verifiers are pure functions over the FINAL state of a record
(outcome-gate rule: never judge the path; error-then-recovery is prime data).

Plugins are path-referenced .py files (tasks reference them in YAML), loaded by file
path. Every verifier ships fixtures: clean_pass, clean_fail, recovered_pass.
"""

from __future__ import annotations

import functools
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from aviary.schema.records import ConversationRecord
from aviary.schema.results import VerifierResult

FIXTURE_NAMES = ("clean_pass", "clean_fail", "recovered_pass")
FIXTURE_EXPECT = {"clean_pass": True, "clean_fail": False, "recovered_pass": True}


class VerifierLoadError(RuntimeError):
    pass


def verifier_id(path: Path, repo_root: Path) -> str:
    if not path.is_relative_to(repo_root):
        # Out-of-tree verifier (test stubs): the stem is the stable id.
        return path.stem
    return str(path.relative_to(repo_root)).removeprefix("verifiers/").removesuffix(".py")


@functools.cache
def load_verifier(path: Path):
    """Load a verifier module by file path; it must define verify(rec) -> VerifierResult.

    Memoized on path: verifier files are byte-stable within a run and pure, so a
    module is exec'd once instead of per (record x verifier) — a big win at burn
    scale. Call load_verifier.cache_clear() if a verifier file changes in-process."""
    if not path.exists():
        raise VerifierLoadError(f"verifier not found: {path}")
    spec = importlib.util.spec_from_file_location(f"aviary_verifier_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "verify"):
        raise VerifierLoadError(f"{path} does not define verify(rec)")
    return module


def run_verifier(path: Path, rec: ConversationRecord) -> VerifierResult:
    return load_verifier(path).verify(rec)


@dataclass
class FixtureReport:
    verifier: str
    passed: bool
    failures: list[str]


def check_fixtures(verifier_path: Path, fixtures_root: Path, vid: str) -> FixtureReport:
    """Fixture rule: a verifier without correct fixtures does not merge."""
    fixture_dir = fixtures_root / vid
    failures: list[str] = []
    module = load_verifier(verifier_path)
    for name in FIXTURE_NAMES:
        fixture = fixture_dir / f"{name}.record.json"
        if not fixture.exists():
            failures.append(f"missing fixture {fixture.relative_to(fixtures_root)}")
            continue
        rec = ConversationRecord.model_validate_json(fixture.read_text())
        result = module.verify(rec)
        if result.passed != FIXTURE_EXPECT[name]:
            failures.append(f"{name}: expected passed={FIXTURE_EXPECT[name]}, got {result.passed}")
    return FixtureReport(verifier=vid, passed=not failures, failures=failures)
