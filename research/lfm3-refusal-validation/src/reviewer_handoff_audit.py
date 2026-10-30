"""Read-only private reviewer handoff and batch integrity audit.

This is an administrative custody check, NOT a grader or evidence of
refusal efficacy. Public output contains only aggregate counts and statuses.
No unblinding key, prompt text, model response, grader ID, or row ID is printed.
"""
import argparse
import json
import stat
from collections import Counter
from pathlib import Path

from src.reviewer_batches import ORIGINAL_FIELDS, read_csv, source_hash
from src.reviewer_progress_audit import classify_row
from src.xstest_450_eval import ARM_ORDER, read_verified_dataset

ROWS_PER_REVIEWER=1350
BATCHES=15
ROWS_PER_BATCH=90


def _private_regular(path):
    path=Path(path)
    if path.is_symlink():
        raise ValueError("review file is a symlink")
    st=path.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:
        raise ValueError("review file is not a single-link regular file")
    if st.st_mode & 0o077:
        raise PermissionError("review files must be private to their owner")


def _private_directory(path):
    path=Path(path)
    if path.is_symlink():
        raise ValueError("review directory is a symlink")
    st=path.stat()
    if not stat.S_ISDIR(st.st_mode) or st.st_mode & 0o077:
        raise PermissionError("review directory must be mode 0700 or stricter")


def audit(packet_dir,batches_dir,source=None):
    packet_dir=Path(packet_dir)
    batches_dir=Path(batches_dir)
    _private_directory(packet_dir)
    _private_directory(batches_dir)
    manifest_path=batches_dir/"private_batch_manifest.json"
    _private_regular(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names=[f"batch_{i:03d}.csv" for i in range(1,BATCHES+1)]
    ids=manifest.get("review_ids")
    if (manifest.get("batch_size")!=ROWS_PER_BATCH or
        manifest.get("records_per_reviewer")!=ROWS_PER_REVIEWER or
        manifest.get("part_names")!=expected_names or
        not isinstance(ids,list) or len(ids)!=ROWS_PER_REVIEWER or
        len(set(ids))!=ROWS_PER_REVIEWER):
        raise ValueError("bad manifest batch dimensions or IDs")
    if set(manifest.get("reviewer_source_sha256",{}))!={"1","2"}:
        raise ValueError("missing source sheet fingerprints")
    if {p.name for p in batches_dir.iterdir()}!={
        "private_batch_manifest.json","reviewer_1","reviewer_2"
    }:
        raise ValueError("unexpected review files or folders in shared batch directory")
    keypath=packet_dir/"private_model_key.json"
    _private_regular(keypath)
    if source_hash(keypath)!=manifest.get("private_key_sha256"):
        raise ValueError("private source mapping fingerprint differs")
    mapping=json.loads(keypath.read_text(encoding="utf-8"))
    if set(mapping)!=set(ids):
        raise ValueError("model key does not match blinded IDs")
    cases={r["id"]:r for r in (source if source is not None else read_verified_dataset())}
    if len(cases)!=450:
        raise ValueError("expected pinned 450-prompt dataset")
    pairs=Counter()
    for key,item in mapping.items():
        case=item.get("source_id")
        arm=item.get("arm")
        if case not in cases or arm not in ARM_ORDER:
            raise ValueError("unrecognized model/key mapping")
        pairs[(case,arm)]+=1
    if len(pairs)!=ROWS_PER_REVIEWER or any(n!=1 for n in pairs.values()):
        raise ValueError("not exactly three model responses for each source prompt")

    result=[]
    review_names=[]
    for reviewer in (1,2):
        src=packet_dir/f"blinded_reviewer_{reviewer}.csv"
        _private_regular(src)
        if source_hash(src)!=manifest["reviewer_source_sha256"][str(reviewer)]:
            raise ValueError("immutable source review sheet fingerprint differs")
        original=read_csv(src)
        if [r["review_id"] for r in original]!=ids:
            raise ValueError("original source sheet differs from fixed batch IDs")
        folder=batches_dir/f"reviewer_{reviewer}"
        _private_directory(folder)
        if {p.name for p in folder.iterdir()}!=set(expected_names):
            raise ValueError("missing or additional reviewer batch file")
        counts=Counter()
        grader_ids=set()
        active_batches=0
        next_batch=None
        total=0
        for batch_index,name in enumerate(expected_names,1):
            part=folder/name
            _private_regular(part)
            rows=read_csv(part)
            if len(rows)!=ROWS_PER_BATCH:
                raise ValueError("wrong reviewer batch record count")
            subset=original[(batch_index-1)*ROWS_PER_BATCH:batch_index*ROWS_PER_BATCH]
            used=0
            for baseline,row in zip(subset,rows):
                if any(row[k]!=baseline[k] for k in ORIGINAL_FIELDS):
                    raise ValueError("source prompt or model response changed in reviewer batch")
                if row["reviewer_id"].strip():
                    grader_ids.add(row["reviewer_id"].strip())
                case=mapping[row["review_id"]]["source_id"]
                state=classify_row(row,cases[case]["class"]=="safe")
                counts[state]+=1
                used+=int(bool(row["reviewer_id"].strip()))
                total+=1
            if used:
                active_batches+=1
            if next_batch is None and any(
                classify_row(r,cases[mapping[r["review_id"]]["source_id"]]["class"]=="safe")!="graded"
                for r in rows
            ):
                next_batch=batch_index
        if total!=ROWS_PER_REVIEWER or len(grader_ids)>1:
            raise ValueError("incomplete review or inconsistent human identity")
        review_names.append(next(iter(grader_ids)) if grader_ids else None)
        result.append({
            "batch_count":BATCHES,
            "records":ROWS_PER_REVIEWER,
            "batches_with_human_entries":active_batches,
            "graded":counts["graded"],
            "ungraded":counts["ungraded"],
            "needs_adjudication":counts["requires_adjudication"],
            "next_incomplete_batch":next_batch,
        })
    if review_names[0] is not None and review_names[0]==review_names[1]:
        raise ValueError("same reviewer identity used for both blinded assignments")
    return {
        "kind":"private blinded human-review custody and readiness, no automatic scoring",
        "source_checkpoint_count":3,
        "prompt_count":450,
        "reviewer_1":result[0],
        "reviewer_2":result[1],
        "original_source_columns_intact":True,
        "reviewer_batch_privacy_permissions_verified":True,
        "human_review_completed":all(x["graded"]==ROWS_PER_REVIEWER for x in result),
        "reviewer_consensus_verified":False,
        "model_unblinding_performed":False,
        "public_safety_results_available":False,
        "release_approved":False,
        "human_action_needed":any(x["graded"]!=ROWS_PER_REVIEWER for x in result),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--packet-dir",type=Path,required=True)
    p.add_argument("--batches-dir",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.packet_dir,args.batches_dir),indent=2))


if __name__=="__main__":
    main()
