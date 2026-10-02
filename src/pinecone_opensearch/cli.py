"""CLI: translate (M0), replay / report (M2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pinecone_opensearch.client import OpenSearch
from pinecone_opensearch.exceptions import UnsupportedQueryError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pinecone-opensearch",
        description="OpenSearch → Pinecone Documents API translator & shadow eval harness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("translate", help="Dry-run translate a search body to a Pinecone plan")
    t.add_argument("--index", "-i", required=True, help="OpenSearch index (or alias)")
    t.add_argument(
        "--body",
        "-b",
        required=True,
        help="Path to JSON file with the OpenSearch search body, or '-' for stdin",
    )
    t.add_argument("--config", "-c", default="pinecone_opensearch.yaml")
    t.add_argument("--routing", default=None)
    t.add_argument("--pretty", action="store_true")

    r = sub.add_parser("replay", help="Replay a query log through ShadowClient")
    r.add_argument("--log", "-l", required=True, help="Input JSONL or slowlog path")
    r.add_argument("--out", "-o", required=True, help="Shadow comparison JSONL output path")
    r.add_argument("--config", "-c", default="pinecone_opensearch.yaml",
                   help="Config for the Pinecone (secondary) client")
    r.add_argument("--index", "-i", default=None, help="Default index when log rows omit it")
    r.add_argument("--qps", type=float, default=None, help="Target queries per second")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--warm-up", type=int, default=100)
    r.add_argument(
        "--primary",
        default="opensearch",
        help="Primary engine: 'opensearch' (requires --opensearch-hosts) or 'pinecone'",
    )
    r.add_argument("--opensearch-hosts", default=None,
                   help="JSON list of OpenSearch hosts, e.g. '[{\"host\":\"localhost\",\"port\":9200}]'")
    r.add_argument("--pinecone-api-key", default=None)

    rep = sub.add_parser("report", help="Summarize a shadow JSONL log + POC charter")
    rep.add_argument("--log", "-l", required=True, help="Shadow JSONL from replay/ShadowClient")
    rep.add_argument("--qrels", default=None, help="Optional qrels TSV or JSONL")
    rep.add_argument("--out", "-o", default=None, help="Write report JSON to this path")
    rep.add_argument("--include-warmup", action="store_true")
    rep.add_argument("--pretty", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "translate":
        return _cmd_translate(args)
    if args.command == "replay":
        return _cmd_replay(args)
    if args.command == "report":
        return _cmd_report(args)
    print(f"unknown command {args.command}", file=sys.stderr)
    return 2


def _cmd_translate(args: argparse.Namespace) -> int:
    if args.body == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(args.body).read_text(encoding="utf-8")
    body = json.loads(raw)

    client = OpenSearch(config=args.config)
    try:
        plan = client.translate(index=args.index, body=body, routing=args.routing)
    except UnsupportedQueryError as exc:
        print(
            json.dumps({"error": "unsupported_query", "path": exc.path, "message": str(exc)}),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"translate failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(plan, indent=2 if args.pretty else None))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from pinecone_opensearch.replay import iter_query_log, replay
    from pinecone_opensearch.shadow import ShadowClient

    pinecone = OpenSearch(
        pinecone_api_key=args.pinecone_api_key,
        config=args.config,
    )

    if args.primary == "pinecone":
        # Demo / single-engine shadow (both sides pinecone) — useful without OS cluster
        primary = pinecone
        secondary = pinecone
    else:
        if not args.opensearch_hosts:
            print(
                "replay with primary=opensearch requires --opensearch-hosts "
                "(or pass --primary pinecone for a single-engine dry demo)",
                file=sys.stderr,
            )
            return 2
        from opensearchpy import OpenSearch as OsClient

        hosts = json.loads(args.opensearch_hosts)
        primary = OsClient(hosts=hosts)
        secondary = pinecone

    shadow = ShadowClient(
        primary,
        secondary,
        args.out,
        warm_up=args.warm_up,
    )
    summary = replay(
        shadow,
        iter_query_log(args.log),
        default_index=args.index,
        qps=args.qps,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["errors"] == 0 else 1


def _cmd_report(args: argparse.Namespace) -> int:
    from pinecone_opensearch.report import build_report_from_path

    report = build_report_from_path(
        args.log,
        qrels_path=args.qrels,
        warm_up_exclude=not args.include_warmup,
    )
    text = json.dumps(report, indent=2 if args.pretty else None, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    # Exit 0 even on charter fail — report is informational; callers inspect poc_charter.pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
