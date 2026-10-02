"""Synthetic quality-audit bundle factory; never contains formal audit data."""

import json
from collections import Counter
from pathlib import Path

from opensearch_vl_repro.rl.quality_audit import QUALITY_AUDIT_VERSION, QUALITY_ELIGIBILITY_RULE


def make_quality_bundle(directory: Path, candidate_ids: list[str], statuses: list[str],
                        *, main_count: int, note: str = "") -> Path:
    assert len(candidate_ids) == len(statuses)
    directory.mkdir(parents=True)
    rows = [{"candidate_rank": rank, "source_sample_id": identity, "status": status,
             "reason": "exclude_question_ambiguous" if status == "exclude" else None,
             "note": note}
            for rank, (identity, status) in enumerate(zip(candidate_ids, statuses, strict=True), start=1)]
    ok = [{"candidate_rank": row["candidate_rank"], "source_sample_id": row["source_sample_id"]}
          for row in rows if row["status"] == "ok"]
    selected, reserve = ok[:main_count], ok[main_count:]
    counts = Counter(row["status"] for row in rows)
    allowlist = {"schema_version": 1, "audit_version": QUALITY_AUDIT_VERSION,
                 "audited_candidate_rank_start": 1, "audited_candidate_rank_end": len(rows),
                 "audited_count": len(rows), "eligibility_rule": QUALITY_ELIGIBILITY_RULE,
                 "total_ok_count": len(ok), "selected_main_count": main_count,
                 "reserve_ok_count": len(reserve), "selected_main400": selected,
                 "reserve_ok": reserve}
    exclusions = {"schema_version": 1, "audit_version": QUALITY_AUDIT_VERSION,
                  "excluded_count": counts["exclude"],
                  "excluded": [{key: row[key] for key in ("candidate_rank", "source_sample_id", "reason", "note")}
                               for row in rows if row["status"] == "exclude"]}
    manual = {"schema_version": 1, "audit_version": QUALITY_AUDIT_VERSION,
              "manual_review_count": counts["manual_review"],
              "samples": [{key: row[key] for key in ("candidate_rank", "source_sample_id", "note")}
                          for row in rows if row["status"] == "manual_review"]}
    summary = {"schema_version": 1, "audit_version": QUALITY_AUDIT_VERSION,
               "candidate_rank_start": 1, "candidate_rank_end": len(rows),
               "audited_count": len(rows), "ok_count": counts["ok"],
               "excluded_count": counts["exclude"], "manual_review_count": counts["manual_review"],
               "reason_counts": {"exclude_question_ambiguous": counts["exclude"]} if counts["exclude"] else {},
               "eligibility_rule": QUALITY_ELIGIBILITY_RULE,
               "final_main_count": main_count, "reserve_ok_count": len(reserve),
               "selected_main400_first_rank": selected[0]["candidate_rank"],
               "selected_main400_last_rank": selected[-1]["candidate_rank"],
               "selected_main400_membership": [item["source_sample_id"] for item in selected],
               "source_candidate_manifest_sha256_values": ["a" * 64], "complete": True}
    (directory / "rl_quality_audit_v1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for filename, value in (("rl_quality_ok_allowlist_v1.json", allowlist),
                            ("rl_quality_exclusions_v1.json", exclusions),
                            ("rl_quality_manual_review_v1.json", manual),
                            ("rl_quality_audit_summary_v1.json", summary)):
        (directory / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return directory
