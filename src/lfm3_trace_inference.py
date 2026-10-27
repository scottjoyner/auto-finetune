"""Opt-in, loopback-only OpenAI-compatible LFM3 inference from safe evidence.

NO remote model download, background service, weight writes, or raw traces.
"""
from __future__ import annotations
import json
from urllib.parse import urlsplit
from urllib.request import Request,build_opener
from trace_memory_bridge import (
    MemoryBridgeError,_NoRedirect,validate_packet,compose_inference_messages,
)
ALLOWED_MODEL = "LiquidAI/LFM3-1.7B"
def complete_local(
    endpoint: str, prompt: str, evidence: dict, *, enabled: bool = False,
    model: str = ALLOWED_MODEL, max_tokens: int = 128,
) -> dict:
    if not enabled:
        raise MemoryBridgeError("local LFM3 inference requires explicit opt-in")
    parsed=urlsplit(endpoint)
    if (parsed.scheme!="http" or parsed.hostname not in ("127.0.0.1","::1")
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path not in ("","/","/v1","/v1/")):
        raise MemoryBridgeError("model endpoint must be numeric loopback")
    if model!=ALLOWED_MODEL or type(max_tokens) is not int or not 1<=max_tokens<=512:
        raise MemoryBridgeError("unsupported or unpinned model/options")
    validate_packet(evidence)
    messages=compose_inference_messages(prompt,evidence)
    payload=json.dumps({"model":model,"messages":messages,
                        "max_tokens":max_tokens,"temperature":0.0}).encode()
    if len(payload)>16_384:
        raise MemoryBridgeError("model request exceeds bounded payload")
    base=parsed.netloc
    request=Request("http://"+base+"/v1/chat/completions",data=payload,
                    headers={"Content-Type":"application/json",
                             "Accept":"application/json"},method="POST")
    with build_opener(_NoRedirect()).open(request,timeout=12) as reply:
        if reply.status!=200:
            raise MemoryBridgeError("model runtime returned non-200")
        raw=reply.read(32_769)
    if len(raw)>32_768:raise MemoryBridgeError("model response exceeded budget")
    result=json.loads(raw)
    choices=result.get("choices",[]) if isinstance(result,dict) else []
    if not isinstance(choices,list) or len(choices)<1:
        raise MemoryBridgeError("unexpected model response shape")
    answer=choices[0].get("message",{}).get("content")
    if not isinstance(answer,str) or len(answer)>16_384:
        raise MemoryBridgeError("non-text or oversized model output")
    return {"model":model,"answer":answer,
            "retrieved_metadata_rows":len(evidence["rows"]),
            "model_weights_modified":False,"production_routed":False}