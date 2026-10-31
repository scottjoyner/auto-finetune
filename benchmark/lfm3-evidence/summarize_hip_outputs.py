#!/usr/bin/env python3
"""Summarize synthetic HIP no-FA / FA generation comparisons without conflating metrics."""
import json
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/"hip-output-summary.json"


def require(test, message):
    if not test: raise ValueError(message)


def score(path):
    r=json.loads(path.read_text())
    require(r.get("schema")=="lfm3.hip-output-smoke.v1",path.name+": schema")
    require(r.get("model_sha256_expected")=="43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452",path.name+": model identity")
    status=r.get("status")
    require(status in ("completed","failed"),path.name+": status")
    if status!="completed":
        return {"receipt":path.name,"status":status,"reason":r.get("reason","no reason"),"fixtures_completed":len(r.get("cases",[]))}
    cases=r.get("cases")
    require(isinstance(cases,list) and cases and len(cases)<=6,path.name+": cases")
    check_agreement=0
    correct={name:0 for name in ["HIP-noFA","HIP-FA"]}
    contained={name:0 for name in correct}
    divergent=[]
    for case in cases:
        variants=case["variants"]
        require(set(variants)==set(correct),path.name+": missing variant")
        a=variants["HIP-noFA"]["generated"]
        b=variants["HIP-FA"]["generated"]
        require(isinstance(a,str) and isinstance(b,str) and a and b,path.name+": output")
        same=(a==b)
        require(case["exact_output_match"] is same,path.name+": agreement indicator mismatch")
        check_agreement+=same
        if not same:divergent.append(case["fixture"])
        for name in correct:
            v=variants[name]
            require(v["backend_confirmed"] is True,path.name+": unexpected backend")
            expected=case["expected"]
            strict=(v["generated"]==expected)
            contains=(expected in v["generated"])
            require(v["strict_expected"] is strict and v["contains_expected"] is contains,
                    path.name+": indicator mismatch")
            correct[name]+=strict
            contained[name]+=contains
            require(case["chat_template_sha256"]==r["sha256"]["embedded_template"],
                    path.name+": chat-template identity mismatch")
    summary=r.get("summary",{})
    require(summary.get("fixtures")==len(cases),path.name+": fixture count")
    require(summary.get("exact_output_agreement_count")==check_agreement,path.name+": agreement count")
    require(summary.get("strict_correct_count")==correct,path.name+": strict count")
    require(summary.get("contains_expected_count")==contained,path.name+": substring count")
    return {"receipt":path.name,"status":status,
            "fixtures":len(cases),"exact_agreement":check_agreement,
            "strict_final_answer":correct,"contains_expected":contained,
            "divergent_fixtures":divergent,"model_sha256":r["model_sha256_expected"],
            "test_executable_sha256":r["sha256"]["test_executable"],
            "template_sha256":r["sha256"]["embedded_template"],
            "hip_backend_lib_sha256":{
                "HIP-noFA":r["sha256"]["hip_no_fa_backend"],
                "HIP-FA":r["sha256"]["hip_fa_backend"]}}


def main():
    inputs=sorted(ROOT.glob("hip-output-compare-*.json"))
    if not inputs:
        require(not OUT.exists(),"orphan output summary")
        print("No synthetic output comparisons")
        return 0
    results=[score(p) for p in inputs]
    result={"schema":"lfm3.hip-output-summary.v1",
            "warning":"Tiny deterministic synthetic suite; identical outputs do not prove general behavioral equivalence. A substring match is not strict correctness.",
            "results":results}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    for row in results:
        print(row["receipt"],row.get("status"),row.get("exact_agreement"),row.get("fixtures"),
              row.get("divergent_fixtures"))
    print("wrote",OUT)
    return 0


if __name__=="__main__":
    try:sys.exit(main())
    except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
        print("FAIL",exc,file=sys.stderr)
        sys.exit(1)
