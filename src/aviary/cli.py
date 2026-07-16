"""`aviary` CLI — the justfile recipes delegate here 1:1."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="aviary")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("install", help="verify pinned hermes checkout + symlink datagen dirs")
    sub.add_parser("pilot", help="small calibration run across enabled lanes")
    sub.add_parser("burn", help="full run; refuses without a recent healthy pilot manifest")
    for name in ("gate", "render", "stats"):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        if name == "render":
            p.add_argument(
                "--prior", action="append", default=[], help="prior run_id for corpus dedupe"
            )

    args = parser.parse_args(argv)

    if args.cmd == "install":
        return _install()
    if args.cmd in ("pilot", "burn"):
        from aviary.orchestrate import cmd_generate

        run_id = cmd_generate(args.cmd)
        print(f"run complete: {run_id} — next: just gate {run_id}")
        return 0
    if args.cmd == "gate":
        from aviary.orchestrate import cmd_gate

        cmd_gate(args.run_id)
        return 0
    if args.cmd == "render":
        from aviary.orchestrate import cmd_render
        from aviary.render.split import HoldoutViolation

        try:
            cmd_render(args.run_id, args.prior)
        except HoldoutViolation as e:
            print(f"HOLDOUT VIOLATION — nothing written: {e}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "stats":
        from aviary.orchestrate import cmd_stats

        print(cmd_stats(args.run_id))
        return 0
    return 1


def _install() -> int:
    import subprocess

    from aviary.lanes.a_agentic.hermes_config import verify_config_keys
    from aviary.paths import REPO_ROOT, hermes_dir
    from aviary.teacher.roster import Roster

    roster = Roster.load(REPO_ROOT / "datagen" / "configs" / "teachers.yaml")
    hd = hermes_dir()
    head = subprocess.run(
        ["git", "-C", str(hd), "describe", "--tags", "--always"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if roster.hermes_pin and head != roster.hermes_pin:
        print(
            f"hermes checkout at {head!r} but teachers.yaml pins {roster.hermes_pin!r}",
            file=sys.stderr,
        )
        return 1
    verify_config_keys(hd)

    # repo is source of truth; hermes reads via symlinks, never edit targets in place
    links = {
        hd / "aviary-toolsets": REPO_ROOT / "datagen" / "toolsets",
        hd / "aviary-persona": REPO_ROOT / "datagen" / "persona",
        hd / "aviary-configs": REPO_ROOT / "datagen" / "configs" / "hermes",
    }
    for link, target in links.items():
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            print(f"refusing to replace non-symlink {link}", file=sys.stderr)
            return 1
        link.symlink_to(target)
        print(f"linked {link} -> {target}")
    print(f"hermes {head} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
