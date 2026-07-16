"""Fixture rule, enforced mechanically: every verifier under verifiers/ must ship
clean_pass, clean_fail, and recovered_pass fixtures, and each must produce the
expected outcome. A verifier without correct fixtures does not merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from aviary.gates.verify import check_fixtures, verifier_id

REPO = Path(__file__).resolve().parents[1]
VERIFIERS_ROOT = REPO / "verifiers"
FIXTURES_ROOT = VERIFIERS_ROOT / "fixtures"

ALL_VERIFIERS = sorted(
    p for p in VERIFIERS_ROOT.rglob("*.py") if "fixtures" not in p.parts and p.name != "__init__.py"
)


def test_at_least_one_verifier_exists():
    assert ALL_VERIFIERS


@pytest.mark.parametrize("path", ALL_VERIFIERS, ids=lambda p: verifier_id(p, REPO))
def test_verifier_has_correct_fixtures(path: Path):
    report = check_fixtures(path, FIXTURES_ROOT, verifier_id(path, REPO))
    assert report.passed, f"{report.verifier}: {report.failures}"
