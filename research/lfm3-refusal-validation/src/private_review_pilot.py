"""Private balanced human-review pilot routed into existing blinded batches.

Selection uses the frozen journal but never publishes model keys, prompt
text, sample IDs, reviewer identity, grades, or repeated response strings.
This tool NEVER generates or infers human ratings.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat

from src.blinded_xstest_review import RESPONSE_CATEGORIES
from src.eval_contracts import refusal_proxy
from src.local_reviewer_terminal import (
    assignment_status, open_batch, review_one,
)
from src.reviewer_batches import source_hash
from src.reviewer_handoff_audit import audit as custody_audit
from src.reviewer_progress_audit import classify_row
from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.xstest_450_eval import ARM_ORDER, read_verified_dataset
from src.xstest_progress_audit import audit as journal_audit, snapshot_complete_lines

PILOT_NAME = "lfm3-model-blind-32-prompt-pilot-v1"
SOURCE_EVIDENCE = Path(__file__).resolve().parents[1] / "evidence/20261030_xstest_response_diversity.json"
BUCKETS = ("safe_mode", "safe_other", "unsafe_mode", "unsafe_other")
N_PER_BUCKET = 8
N_PROMPTS = 32
N_PER_REVIEWER = 96
def _regular_private(path):
    p = Path(path)
    if any(part.is_symlink() for part in (p, *p.parents)):
        raise ValueError("private route or any ancestor cannot be symlinked")
    directory = p.parent.stat()
    if not stat.S_ISDIR(directory.st_mode) or directory.st_mode & 0o077:
        raise PermissionError("private route directory must be owner-only")
    st = p.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_mode & 0o077:
        raise PermissionError("route must be owner-only regular file")
    return p


def deterministic_sample(group, case_ids, k=N_PER_BUCKET):
    """Stable across restarts and independent of model response text order."""
    if len(case_ids) < k or len(case_ids) != len(set(case_ids)):
        raise ValueError("insufficient distinct source prompts")
    def priority(case):
        payload = (PILOT_NAME + ":" + group + ":" + case).encode()
        return hashlib.sha256(payload).hexdigest(), case
    return sorted(case_ids, key=priority)[:k]


def select_cases(records, dataset, reported):
    cases = {x["id"]: x for x in dataset}
    if len(cases) != 450 or len(records) != 1350:
        raise ValueError("incorrect XSTest panel coverage")
    lookup = {}
    for item in records:
        key = (item["id"], item["arm"])
        if key in lookup or item["id"] not in cases or item["arm"] not in ARM_ORDER:
            raise ValueError("duplicate or unexpected model/case")
        if item["class"] != cases[item["id"]]["class"]:
            raise ValueError("class mismatch")
        lookup[key] = item
    if len(lookup) != 1350:
        raise ValueError("incomplete paired panel")
    c = ARM_ORDER[2]
    mode_counts = Counter(lookup[(case,c)]["response"] for case in cases)
    most = mode_counts.most_common(2)
    if len(most) < 2 or most[0][1] != 156 or most[0][1] == most[1][1]:
        raise ValueError("frozen C mode must be unique and occur 156 times")
    mode = most[0][0]
    baseline = reported["checkpoint_data"][c]["overall"]["top_mode"]
    if not baseline or baseline["total_occurrences"] != 156 or not baseline["keyword_positive"]:
        raise ValueError("source publication does not match modal response")
    buckets = {name: [] for name in BUCKETS}
    for case, source in cases.items():
        group = source["class"]
        repeated = lookup[(case,c)]["response"] == mode
        bucket = group + ("_mode" if repeated else "_other")
        buckets[bucket].append(case)
    chosen = {b:deterministic_sample(b, ids) for b,ids in buckets.items()}
    if len(set(sum(chosen.values(), []))) != N_PROMPTS:
        raise ValueError("overlapping or missing pilot prompts")
    return chosen
def create_route(results_dir, packet_dir, batches_dir, out):
    """Generate one private route, not a copy of any answer or new review sheet."""
    if out.exists():
        raise FileExistsError("do not overwrite a reviewer route")
    report = custody_audit(packet_dir, batches_dir)
    if not report["original_source_columns_intact"] or not report["reviewer_batch_privacy_permissions_verified"]:
        raise ValueError("private packet custody not proved")
    progress = journal_audit(results_dir)
    if (progress["journal_integrity"] != "passed"
            or progress["recorded_generations"] != 1350
            or progress["ignored_unterminated_tail"]
            or progress["checkpoint_sha256"] != checked_arm_paths(ARMS)):
        raise ValueError("journal / checkpoint manifest mismatch")
    records, partial, digest = snapshot_complete_lines(
        results_dir / "private_generations.jsonl")
    evidence = json.loads(SOURCE_EVIDENCE.read_text(encoding="utf-8"))
    if (partial or digest != progress["journal_complete_lines_sha256"]
            or evidence["private_journal_sha256"] != digest
            or evidence["checkpoint_sha256"] != progress["checkpoint_sha256"]):
        raise ValueError("duplicate audit/source journal fingerprint mismatch")
    chosen = select_cases(records, read_verified_dataset(), evidence)
    mapping = json.loads((packet_dir/"private_model_key.json").read_text())
    reverse = {}
    for blinded_id, identity in mapping.items():
        key = (identity.get("source_id"), identity.get("arm"))
        if key in reverse:
            raise ValueError("duplicate model key in blinded mapping")
        reverse[key] = blinded_id
    if len(reverse) != 1350:
        raise ValueError("private blinded mapping incomplete")
    selection = []
    for bucket, case_ids in chosen.items():
        for case in case_ids:
            for arm in ARM_ORDER:
                selection.append(reverse[(case,arm)])
    if len(set(selection)) != N_PER_REVIEWER:
        raise ValueError("expected 96 distinct blinded review IDs")
    selection.sort(key=lambda ident: hashlib.sha256(
        (PILOT_NAME+":order:"+ident).encode()).hexdigest())
    manifest = json.loads((batches_dir/"private_batch_manifest.json").read_text())
    ids = manifest["review_ids"]
    if len(ids) != 1350 or len(set(ids)) != 1350:
        raise ValueError("bad review batch manifest")
    position = {identity:(i//90+1,i%90+1) for i,identity in enumerate(ids)}
    route = [{"review_id":key,"batch":position[key][0],"index":position[key][1]}
             for key in selection]
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if out.parent.stat().st_mode & 0o077:
        raise PermissionError("private route directory must be owner-only")
    private_manifest = {
        "study": PILOT_NAME, "n_prompts": N_PROMPTS,
        "entries_per_reviewer": N_PER_REVIEWER,
        "source_journal_sha256": digest,
        "review_batch_manifest_sha256": source_hash(batches_dir/"private_batch_manifest.json"),
        "private_model_key_sha256": source_hash(packet_dir/"private_model_key.json"),
        "route": route,
        "human_labels_present_in_route": False, "model_identity_present_in_route": False,
    }
    fd = os.open(str(out), os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as f:
        json.dump(private_manifest,f,indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return {
        "pilot":PILOT_NAME, "n_source_prompts":N_PROMPTS,
        "n_responses_per_reviewer":N_PER_REVIEWER,
        "two_independent_human_reviewers_required":True,
        "selection_counts_without_model_identifiers":{b:len(v) for b,v in chosen.items()},
        "original_blinded_review_csvs_modified":False,
        "actual_human_grades_added":0, "safety_approval":False,
    }
def load_route(route_path, packet_dir, batches_dir, *, include_manifest=False):
    # Open once: subsequent consumers must use the exact verified snapshot,
    # not reopen a mutable path after source attestation.
    path = _regular_private(route_path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or opened.st_mode & 0o077):
            raise PermissionError("opened private route is not owner-only regular file")
        route = json.load(stream)
    if (route.get("study") != PILOT_NAME
            or route.get("n_prompts") != N_PROMPTS
            or route.get("entries_per_reviewer") != N_PER_REVIEWER
            or route.get("human_labels_present_in_route") is not False
            or route.get("model_identity_present_in_route") is not False):
        raise ValueError("unsupported route manifest")
    if (source_hash(batches_dir/"private_batch_manifest.json")
            != route.get("review_batch_manifest_sha256")
            or source_hash(packet_dir/"private_model_key.json")
            != route.get("private_model_key_sha256")):
        raise ValueError("pilot source manifest fingerprint mismatch")
    manifest = json.loads((batches_dir/"private_batch_manifest.json").read_text())
    identities = manifest["review_ids"]
    entries = route.get("route")
    if not isinstance(entries,list) or len(entries)!=N_PER_REVIEWER:
        raise ValueError("incorrect number of blinded pilot targets")
    seen=set()
    for entry in entries:
        if set(entry) != {"review_id","batch","index"}:
            raise ValueError("route contains unexpected fields")
        blinded, batch, row=entry["review_id"],entry["batch"],entry["index"]
        if (type(batch) is not int or type(row) is not int or
                not 1<=batch<=15 or not 1<=row<=90
                or identities[(batch-1)*90+row-1] != blinded or blinded in seen):
            raise ValueError("tampered route position or duplicate")
        seen.add(blinded)
    return (entries, route) if include_manifest else entries
def pilot_progress(entries, batches_dir, reviewer, packet_dir):
    custody_audit(packet_dir,batches_dir)
    mapping = json.loads((packet_dir/"private_model_key.json").read_text())
    cases = {x["id"]:x for x in read_verified_dataset()}
    cache = {}
    states = Counter()
    for entry in entries:
        batch=entry["batch"]
        if batch not in cache:
            cache[batch] = open_batch(batches_dir,reviewer,batch)[1]
        row = cache[batch][entry["index"]-1]
        if row["review_id"] != entry["review_id"]:
            raise ValueError("reviewer batch identity no longer matches route")
        source_class = cases[mapping[entry["review_id"]]["source_id"]]["class"]
        states[classify_row(row, source_class=="safe")] += 1
    return {
        "assigned_reviewer_number": reviewer,
        "targeted_items":N_PER_REVIEWER,
        "graded":states["graded"],
        "ungraded":states["ungraded"],
        "requires_adjudication":states["requires_adjudication"],
        "pilot_human_complete":states["graded"]==N_PER_REVIEWER,
        "full_benchmark_review_complete":False,
        "safety_claim_permitted":False,
    }
def interactive_review(entries, batches_dir, reviewer, reviewer_id, packet_dir):
    if not os.isatty(0) or not os.isatty(1):
        raise PermissionError("human review requires local interactive terminal")
    if not reviewer_id or not reviewer_id.strip():
        raise ValueError("actual independent human reviewer ID required")
    assignment_status(batches_dir, reviewer, reviewer_id)
    mapping = json.loads((packet_dir/"private_model_key.json").read_text())
    cases = {x["id"]:x for x in read_verified_dataset()}
    for entry in entries:
        batch=entry["batch"]
        path,rows=open_batch(batches_dir,reviewer,batch)
        target=rows[entry["index"]-1]
        cls=cases[mapping[entry["review_id"]]["source_id"]]["class"]
        if classify_row(target,cls=="safe")=="graded":
            continue
        if not review_one(path,rows,entry["index"]-1,reviewer_id):
            break
    return pilot_progress(entries,batches_dir,reviewer,packet_dir)
def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode",required=True)
    c=sub.add_parser("create")
    c.add_argument("--results-dir",type=Path,required=True)
    c.add_argument("--packet-dir",type=Path,required=True)
    c.add_argument("--batches-dir",type=Path,required=True)
    c.add_argument("--out",type=Path,required=True)
    for action in ("status","review"):
        cmd=sub.add_parser(action)
        cmd.add_argument("--results-dir",type=Path,default=Path(
            "/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_20261029_v1"))
        cmd.add_argument("--packet-dir",type=Path,required=True)
        cmd.add_argument("--batches-dir",type=Path,required=True)
        cmd.add_argument("--route",type=Path,required=True)
        cmd.add_argument("--reviewer",type=int,required=True,choices=(1,2))
        if action=="review":
            cmd.add_argument("--reviewer-id",required=True)
    args=p.parse_args()
    os.umask(0o077)
    if args.mode=="create":
        result=create_route(args.results_dir,args.packet_dir,args.batches_dir,args.out)
    else:
        # Validate exact source-derived selection before inspecting or rating.
        # Valid ID/row offsets alone do not prove a pilot remains balanced.
        from src.private_review_pilot_attestation import attest_route_with_entries
        _, entries = attest_route_with_entries(
            args.results_dir, args.packet_dir, args.batches_dir, args.route)
        if args.mode=="status":
            result=pilot_progress(entries,args.batches_dir,args.reviewer,args.packet_dir)
        else:
            result=interactive_review(entries,args.batches_dir,args.reviewer,
                                      args.reviewer_id,args.packet_dir)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
