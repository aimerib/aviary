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
    for name in ("gate", "render", "stats", "ship"):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        if name == "render":
            p.add_argument(
                "--prior", action="append", default=[], help="prior run_id for corpus dedupe"
            )
        if name == "ship":
            p.add_argument(
                "--repo", default=None, help="HF dataset repo (default: aviary-<run_id>)"
            )
            p.add_argument(
                "--dry-run", action="store_true", help="validate + list files, contact nothing"
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
    if args.cmd == "ship":
        from aviary.ship import ShipError, plan_ship, ship_run

        try:
            if args.dry_run:
                root, files = plan_ship(args.run_id)
                repo = ship_run(args.run_id, args.repo, dry_run=True)
                total = sum(f.stat().st_size for f in files)
                print(f"would ship {len(files)} files ({total / 1e6:.1f} MB) from {root}")
                print(f"  -> PRIVATE hf dataset: {repo}")
                return 0
            repo = ship_run(args.run_id, args.repo)
        except ShipError as e:
            print(f"REFUSED TO SHIP: {e}", file=sys.stderr)
            return 2
        print(f"shipped {args.run_id} -> private hf dataset {repo}")
        return 0
    return 1


def _install() -> int:
    """Verify the pinned hermes checkout and its batch_runner interface. Nothing is
    written into the checkout: batch_runner takes everything via CLI flags, so
    there are no configs to place (datagen/toolsets/ are aviary's own schema
    mirrors for ingest/render, not files hermes reads)."""
    from aviary.lanes.a_agentic.hermes_config import verify_hermes_interface, verify_hermes_pin
    from aviary.paths import REPO_ROOT, hermes_dir
    from aviary.teacher.roster import Roster

    roster = Roster.load(REPO_ROOT / "datagen" / "configs" / "teachers.yaml")
    hd = hermes_dir()
    try:
        head = verify_hermes_pin(hd, roster.hermes_pin)
        verify_hermes_interface(hd)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"hermes {head} ok (batch_runner interface verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
