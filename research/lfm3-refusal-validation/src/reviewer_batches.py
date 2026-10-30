"""Safe local batching and fail-closed reassembly of blinded human XSTest sheets.

No model inference, automatic grading, network transfer, or output publication.
Raw prompts and responses remain in private paths.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

FIELDS=("review_id","prompt","model_response","reviewer_id","response_category",
        "actionable_harmful_content","benign_helpfulness_0_3","uncertainty_notes")
ORIGINAL_FIELDS=("review_id","prompt","model_response")


def read_csv(path):
    with Path(path).open(encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        if tuple(reader.fieldnames or ())!=FIELDS:
            raise ValueError("private reviewer sheet schema mismatch")
        rows=list(reader)
    if len(set(row["review_id"] for row in rows))!=len(rows):
        raise ValueError("duplicate blinded review ID")
    return rows


def write_csv(path,rows):
    path=Path(path)
    with path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n",extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_packet(packet_dir, output_dir, batch_size=90):
    os.umask(0o077)
    packet_dir=Path(packet_dir)
    output_dir=Path(output_dir)
    if not 1<=batch_size<=450:
        raise ValueError("batch size must be 1..450")
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite any existing review batch directory")
    original=[]
    for reviewer in (1,2):
        path=packet_dir/f"blinded_reviewer_{reviewer}.csv"
        rows=read_csv(path)
        if len(rows)!=1350:
            raise ValueError("source reviewer packet is not the full 1350-item evaluation")
        original.append((path,rows))
    if any(x["review_id"]!=y["review_id"] or
           any(x[k]!=y[k] for k in ("prompt","model_response"))
           for x,y in zip(original[0][1],original[1][1])):
        raise ValueError("independent source packets are not identical")
    key=json.loads((packet_dir/"private_model_key.json").read_text())
    if set(key)!={r["review_id"] for r in original[0][1]}:
        raise ValueError("private mapping not identical to reviewer records")
    output_dir.mkdir(parents=True,mode=0o700)
    manifest={
        "study":"three-arm XSTest 450 prompts x 3 model outputs",
        "batch_size":batch_size,
        "records_per_reviewer":1350,
        "reviewer_source_sha256":{str(n):source_hash(path) for n,(path,_) in enumerate(original,1)},
        "private_key_sha256":source_hash(packet_dir/"private_model_key.json"),
        "review_ids":[row["review_id"] for row in original[0][1]],
        "part_names":[],
        "human_reviews_completed":False,
    }
    for reviewer,(_,rows) in enumerate(original,1):
        folder=output_dir/f"reviewer_{reviewer}"
        folder.mkdir(mode=0o700)
        for i,start in enumerate(range(0,len(rows),batch_size),1):
            fname=f"batch_{i:03d}.csv"
            write_csv(folder/fname,rows[start:start+batch_size])
            if reviewer==1:
                manifest["part_names"].append(fname)
    path=output_dir/"private_batch_manifest.json"
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    path.chmod(0o600)
    return {"batches_per_reviewer":len(manifest["part_names"]),
            "rows_per_reviewer":1350,
            "reviewers_required":2,
            "human_reviews_completed":False}


def assemble_packets(packet_dir, batches_dir, output_dir):
    """Accept only independently completed sheets, preserving source text byte-for-byte."""
    from src.reviewer_progress_audit import classify_row
    from src.xstest_450_eval import read_verified_dataset
    os.umask(0o077)
    packet_dir=Path(packet_dir)
    batches_dir=Path(batches_dir)
    output_dir=Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite review work")
    manifest=json.loads((batches_dir/"private_batch_manifest.json").read_text())
    if manifest["records_per_reviewer"]!=1350:
        raise ValueError("invalid source count")
    keypath=packet_dir/"private_model_key.json"
    if source_hash(keypath)!=manifest["private_key_sha256"]:
        raise ValueError("private model mapping changed")
    key=json.loads(keypath.read_text())
    source={r["id"]:r for r in read_verified_dataset()}
    assembled=[]
    names=[]
    for reviewer in (1,2):
        source_path=packet_dir/f"blinded_reviewer_{reviewer}.csv"
        if source_hash(source_path)!=manifest["reviewer_source_sha256"][str(reviewer)]:
            raise ValueError("original blinded sheet changed since batching")
        originals=read_csv(source_path)
        rows=[]
        for name in manifest["part_names"]:
            rows.extend(read_csv(batches_dir/f"reviewer_{reviewer}"/name))
        if len(rows)!=1350:
            raise ValueError("reviewer batch records missing or added")
        if [r["review_id"] for r in rows]!=manifest["review_ids"]:
            raise ValueError("reviewer batch IDs out of order or duplicated")
        ids=set()
        for a,b in zip(originals,rows):
            if any(a[field]!=b[field] for field in ORIGINAL_FIELDS):
                raise ValueError("prompt, response, or blinded ID modified during review")
            reviewer_id=b["reviewer_id"].strip()
            if not reviewer_id:
                raise ValueError("human grader identity missing")
            ids.add(reviewer_id)
            item=key[b["review_id"]]
            is_safe=source[item["source_id"]]["class"]=="safe"
            if classify_row(b,is_safe)!="graded":
                raise ValueError("human annotation missing or uncertain; adjudication required")
        if len(ids)!=1:
            raise ValueError("one consistent independent reviewer per packet")
        names.append(next(iter(ids)))
        assembled.append(rows)
    if names[0]==names[1]:
        raise ValueError("the two reviewers must be independent and distinct")
    output_dir.mkdir(mode=0o700)
    for n,rows in enumerate(assembled,1):
        write_csv(output_dir/f"blinded_reviewer_{n}.csv",rows)
    shutil.copyfile(keypath,output_dir/"private_model_key.json")
    (output_dir/"private_model_key.json").chmod(0o600)
    # The established consensus scorer, not this tool, remains responsible for disagreement rejection.
    return {"rows_per_reviewer":1350,
            "two_distinct_reviewers":True,
            "full_human_consensus_validated":False,
            "requires_existing_consensus_scorer":True}


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode",required=True)
    for action in ("split","assemble"):
        s=sub.add_parser(action)
        s.add_argument("--packet-dir",type=Path,required=True)
        s.add_argument("--batches-dir",type=Path,required=True)
        if action=="assemble":
            s.add_argument("--output-dir",type=Path,required=True)
        else:
            s.add_argument("--batch-size",type=int,default=90)
    args=p.parse_args()
    if args.mode=="split":
        result=split_packet(args.packet_dir,args.batches_dir,args.batch_size)
    else:
        result=assemble_packets(args.packet_dir,args.batches_dir,args.output_dir)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
