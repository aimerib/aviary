"""`aviary` CLI — the justfile recipes delegate here 1:1."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


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
            p.add_argument(
                "--card-only",
                action="store_true",
                help="re-push only the dataset card to an already-shipped run",
            )

    pi = sub.add_parser(
        "prepare-interleave", help="render a cleaned external corpus through THE serializer"
    )
    pi.add_argument("name", help="corpus dir under $AVIARY_DATA_DIR/external/")
    pi.add_argument(
        "--with-thoughts", action="store_true", help="emit <think> blocks (default: strip them)"
    )
    pi.add_argument("--against", default=None, help="run_id to report the resulting mix against")

    se = sub.add_parser("ship-external", help="upload a cleaned external corpus (private)")
    se.add_argument("name", help="corpus dir under $AVIARY_DATA_DIR/external/")
    se.add_argument("--repo", default=None, help="HF repo (default: aviary-external-<name>)")
    se.add_argument("--dry-run", action="store_true", help="validate only, contact nothing")

    cr = sub.add_parser("clean-rp-reasoning", help="filter aimeri/rp-reasoning-v2 for review")
    cr.add_argument("--cap", type=int, default=0, help="downsample survivors to N (0 = all)")
    cr.add_argument("--shards", type=int, default=5)
    cr.add_argument("--seed", type=int, default=20260716)
    cr.add_argument("--strict-minor", action="store_true", help="also drop school-age settings")
    cr.add_argument("--drop-claude", action="store_true", help="drop rows naming Claude/Anthropic")

    fb = sub.add_parser("fetch-books", help="download or score public-domain fiction for lane B")
    fb.add_argument("source", choices=["gutenberg", "standardebooks", "local"])
    fb.add_argument(
        "--out", type=Path, help="download dir for gutenberg/standardebooks (OUTSIDE the repo)"
    )
    fb.add_argument("--dir", type=Path, dest="local_dir", help="existing book dir to score (local)")
    fb.add_argument("--limit", type=int, default=100)
    fb.add_argument("--min-interiority", type=float, default=7.0, help="interiority keep threshold")
    fb.add_argument("--score-report", action="store_true", help="score only, download nothing")
    fb.add_argument("--existing", type=Path, default=None, help="lane_b.yaml to dedupe against")
    fb.add_argument("--emit-yaml", action="store_true", help="print appendable book entries")

    ao = sub.add_parser(
        "ingest-ao3", help="filter + fandom-sample AO3 (otwarchive JSONL) into lane B"
    )
    ao.add_argument(
        "--in", dest="ao3_in", type=Path, required=True, help="dir of ao3_works_*.jsonl"
    )
    ao.add_argument("--out", type=Path, required=True, help="corpus output dir (OUTSIDE the repo)")
    ao.add_argument("--limit", type=int, default=3000, help="max works to keep (fandom-diverse)")
    ao.add_argument("--min-words", type=int, default=300, help="drop works shorter than this")
    ao.add_argument(
        "--per-fandom-cap", type=int, default=5, help="max works per fandom, for breadth"
    )
    ao.add_argument("--existing", type=Path, default=None, help="lane_b.yaml to dedupe against")
    ao.add_argument("--emit-yaml", action="store_true", help="print appendable book entries")

    et = sub.add_parser(
        "evolve-tasks", help="Auto Evol-Instruct: grow a task template's instances via a teacher"
    )
    et.add_argument("template", type=Path, help="tasks/<family>/<name>.task.yaml (needs a deriver)")
    et.add_argument("--n", type=int, default=20, help="target number of new instances")
    et.add_argument("--model", default=None, help="roster teacher id (default: lane C character)")

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
            repo = ship_run(args.run_id, args.repo, card_only=args.card_only)
        except ShipError as e:
            print(f"REFUSED TO SHIP: {e}", file=sys.stderr)
            return 2
        what = "dataset card for" if args.card_only else "shipped"
        print(f"{what} {args.run_id} -> private hf dataset {repo}")
        return 0
    if args.cmd == "prepare-interleave":
        import json

        from aviary.external.interleave import prepare

        report = prepare(args.name, with_thoughts=args.with_thoughts, against_run=args.against)
        print(json.dumps(report, indent=2))
        return 0
    if args.cmd == "ship-external":
        from aviary.ship import ShipError, plan_ship_external, ship_external

        try:
            if args.dry_run:
                root, files, report = plan_ship_external(args.name)
                repo = ship_external(args.name, args.repo, dry_run=True)
                total = sum(f.stat().st_size for f in files)
                print(f"would ship {len(files)} files ({total / 1e6:.1f} MB) from {root}")
                print(f"  rows written: {report.get('written', '?')}")
                print(f"  -> PRIVATE hf dataset: {repo}")
                return 0
            repo = ship_external(args.name, args.repo)
        except ShipError as e:
            print(f"REFUSED TO SHIP: {e}", file=sys.stderr)
            return 2
        print(f"shipped external corpus {args.name} -> private hf dataset {repo}")
        return 0
    if args.cmd == "clean-rp-reasoning":
        import json

        from aviary.external.rp_reasoning import clean
        from aviary.paths import external_dir

        out = external_dir("rp-reasoning-v2")
        report = clean(
            out,
            shards=args.shards,
            cap=args.cap,
            seed=args.seed,
            strict_minor=args.strict_minor,
            drop_claude=args.drop_claude,
        )
        print(json.dumps(report, indent=2))
        print(f"\nwritten to {out}")
        return 0
    if args.cmd == "fetch-books":
        from aviary.lanes.b_fiction.fetch import (
            dump_book_yaml,
            existing_keys,
            fetch_gutenberg,
            fetch_standardebooks,
            score_local,
        )

        have = existing_keys(args.existing) if args.existing else set()
        if args.source == "local":
            if not args.local_dir:
                parser.error("fetch-books local requires --dir")
            books = score_local(args.local_dir, min_interiority=args.min_interiority, existing=have)
        elif args.source == "gutenberg":
            if not args.out:
                parser.error("fetch-books gutenberg requires --out")
            books = fetch_gutenberg(
                args.out,
                limit=args.limit,
                min_interiority=args.min_interiority,
                existing=have,
                dry_run=args.score_report,
            )
        else:
            if not args.out:
                parser.error("fetch-books standardebooks requires --out")
            books = fetch_standardebooks(args.out, limit=args.limit)
        if args.emit_yaml and not args.score_report:
            fresh = [b for b in books if b.title.lower() not in have]
            print("\n" + dump_book_yaml(fresh))
        return 0
    if args.cmd == "ingest-ao3":
        from aviary.lanes.b_fiction.ao3 import ingest_ao3
        from aviary.lanes.b_fiction.fetch import dump_book_yaml, existing_keys

        have = existing_keys(args.existing) if args.existing else set()
        books = ingest_ao3(
            args.ao3_in,
            args.out,
            limit=args.limit,
            min_words=args.min_words,
            per_fandom_cap=args.per_fandom_cap,
            existing=have,
        )
        if args.emit_yaml:
            print("\n" + dump_book_yaml(books))
        return 0
    if args.cmd == "evolve-tasks":
        import yaml

        from aviary.lanes.a_agentic.evolve import (
            DERIVERS,
            EVOLVE_SYSTEM,
            evolve_template,
            existing_instance_keys,
        )
        from aviary.orchestrate import make_clients
        from aviary.paths import configs_dir
        from aviary.teacher.client import ChatRequest
        from aviary.teacher.roster import Roster

        template = yaml.safe_load(args.template.read_text())
        tid = template.get("id")
        if tid not in DERIVERS:
            print(f"no evolve deriver for {tid!r}; registered: {sorted(DERIVERS)}", file=sys.stderr)
            return 2
        roster = Roster.load(configs_dir() / "teachers.yaml")
        client, _ = make_clients(roster, "evolve")
        model = args.model or roster.assigned_pool("lane_c", "character")[0].id

        def generate(prompt: str) -> str:
            return client.complete(
                ChatRequest(
                    model=model,
                    system=EVOLVE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=2048,
                    reasoning_off=False,
                ),
                lane="a",
            ).text

        new = evolve_template(
            template, generate, n=args.n, existing=existing_instance_keys(template)
        )
        print(f"# {len(new)} new instances for {tid} (ground truth recomputed)", file=sys.stderr)
        print(
            f"# review the difficulty band, then append each list to {args.template}",
            file=sys.stderr,
        )
        if new:
            cols = {k: [inst[k] for inst in new] for k in new[0]}
            print(yaml.safe_dump(cols, sort_keys=False, allow_unicode=True, width=100))
        return 0
    return 1


def _install() -> int:
    """Verify the pinned hermes checkout and its batch_runner interface. Nothing is
    written into the checkout: batch_runner takes everything via CLI flags, so
    there are no configs to place (datagen/toolsets/ are aviary's own schema
    mirrors for ingest/render, not files hermes reads)."""
    from aviary.lanes.a_agentic.hermes_config import (
        hermes_python,
        verify_hermes_interface,
        verify_hermes_pin,
        verify_hermes_python,
    )
    from aviary.paths import REPO_ROOT, hermes_dir
    from aviary.teacher.roster import Roster

    roster = Roster.load(REPO_ROOT / "datagen" / "configs" / "teachers.yaml")
    hd = hermes_dir()
    python = hermes_python()
    try:
        head = verify_hermes_pin(hd, roster.hermes_pin)
        verify_hermes_interface(hd)
        # The whole point of `just install`: catch a bad hermes interpreter now,
        # not 40 minutes into a run once lanes B and C have been paid for.
        verify_hermes_python(hd, python)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"hermes {head} ok (batch_runner interface verified; interpreter {python})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
