"""Actual PyTorch module hook test; no LFM checkpoint weights used."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from lfm3_head_gate_hook import (
    GateHookRefused,strict_json,validate,synthetic_head_gate_hooks,
    ATTENTION_LAYERS,CONFIG_SHA256
)
try:import torch
except ImportError:torch=None

def make_plan(bin_path):
    values=[((j%13)-6)*0.02 for j in range(384)]
    output=subprocess.run([str(bin_path),"--synthetic-only","0.8"],
                          input=json.dumps(values).encode(),capture_output=True,
                          timeout=8,check=False)
    if output.returncode:
        raise RuntimeError("Rust gate plan binary failed")
    return strict_json(output.stdout)

def load_config():
    path=os.environ.get("LFM3_CONFIG_JSON")
    if not path:return None
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CONFIG_SHA256:
        raise RuntimeError("model config outside pinned official revision")
    return raw

class PlanSchemaTest(unittest.TestCase):
    def fixture(self):
        return {
            "schema":"sg-lfm3-gate-plan.v1",
            "model_id":"LiquidAI/LFM3-1.7B",
            "model_config_sha256":CONFIG_SHA256,
            "source_kind":"synthetic-lab-v1",
            "source_vector_sha256":"a"*64,
            "research_only":True,"model_weights_modified":False,
            "attention_backend_modified":False,
            "layers":[{
                "layer_index":idx,"layer_type":kind,
                "head_logit_deltas":[0.0]*16
            } for idx,kind in ATTENTION_LAYERS],
        }
    def test_strict_synthetic_only_plan_and_field_rejection(self):
        obj=self.fixture()
        self.assertIs(validate(obj),obj)
        obj["source_kind"]="real-unsigned-traces"
        with self.assertRaises(GateHookRefused):validate(obj)
        obj=self.fixture();obj["layers"][2]["head_logit_deltas"]=[0.0]*15
        with self.assertRaises(GateHookRefused):validate(obj)
    def test_duplicate_json_and_unsafe_delta_rejected(self):
        with self.assertRaises(GateHookRefused):
            strict_json(b'{"schema":"a","schema":"b"}')
        obj=self.fixture();obj["layers"][0]["head_logit_deltas"][0]=float("nan")
        with self.assertRaises(GateHookRefused):validate(obj)

@unittest.skipUnless(torch is not None,"PyTorch not available on this host")
class GateHookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary=os.environ.get("SG_LFM3_GATE_PLAN_BIN")
        cls.config_bytes=load_config()
        source_path=os.environ.get("LFM3_MODELING_PY")
        cls.modeling_bytes=Path(source_path).read_bytes() if source_path else None
        if not cls.binary or not cls.config_bytes or not cls.modeling_bytes:
            raise unittest.SkipTest("set exact Rust binary path and pinned config file")

    def fake_model(self):
        from types import SimpleNamespace
        layer_types=json.loads(self.config_bytes)["layer_types"]
        class AttentionBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.is_attention_layer=True
                self.self_attn=torch.nn.Module()
                self.self_attn.gate_proj=torch.nn.Linear(2048,16,bias=False)
        class ConvBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.is_attention_layer=False
        model=torch.nn.Module()
        model.model=torch.nn.Module()
        model.model.layers=torch.nn.ModuleList([
            AttentionBlock() if l!="conv" else ConvBlock() for l in layer_types
        ])
        model.config=SimpleNamespace(model_type="lfm3",num_hidden_layers=24,
            num_attention_heads=16,num_key_value_heads=4,
            head_dim=128,hidden_size=2048,sliding_window=256,
            layer_types=layer_types)
        return model

    def test_reversible_head_gating_and_no_model_weight_changes(self):
        torch.manual_seed(19)
        model=self.fake_model()
        plan=make_plan(self.binary)
        self.assertEqual(len(plan["layers"]),11)
        validate(plan)
        states=torch.randn(1,2,2048)
        names=[i for i,_ in ATTENTION_LAYERS]
        before=[model.model.layers[i].self_attn.gate_proj.weight.detach().clone()
                for i in names]
        baseline=[model.model.layers[i].self_attn.gate_proj(states).detach().clone()
                  for i in names]
        with synthetic_head_gate_hooks(model,plan,self.config_bytes,self.modeling_bytes,enabled=True) as info:
            self.assertEqual(info["hooked_attention_layers"],11)
            changed=[model.model.layers[i].self_attn.gate_proj(states) for i in names]
            self.assertTrue(any(not torch.equal(a,b)
                                for a,b in zip(baseline,changed)))
            for row,a,b in zip(plan["layers"],baseline,changed):
                target=torch.tensor(row["head_logit_deltas"],dtype=b.dtype)
                self.assertTrue(torch.allclose(b-a,target.view(1,1,16),atol=2e-6))
        after=[model.model.layers[i].self_attn.gate_proj(states) for i in names]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(baseline,after)))
        new_weights=[model.model.layers[i].self_attn.gate_proj.weight for i in names]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,new_weights)))

    def test_refusals_and_no_implicit_runtime_hook(self):
        model=self.fake_model();plan=make_plan(self.binary)
        with self.assertRaises(GateHookRefused):
            with synthetic_head_gate_hooks(model,plan,self.config_bytes,self.modeling_bytes):pass
        bad=dict(plan);bad["research_only"]=False
        with self.assertRaises(GateHookRefused):
            with synthetic_head_gate_hooks(model,bad,self.config_bytes,self.modeling_bytes,enabled=True):pass
        bad=json.loads(json.dumps(plan))
        bad["layers"][0]["head_logit_deltas"][0]=1e6
        with self.assertRaises(GateHookRefused):
            with synthetic_head_gate_hooks(model,bad,self.config_bytes,self.modeling_bytes,enabled=True):pass
        with self.assertRaises(GateHookRefused):
            with synthetic_head_gate_hooks(model,plan,b"wrong",self.modeling_bytes,enabled=True):pass
        self.assertFalse(any(bool(model.model.layers[i].self_attn.gate_proj._forward_hooks)
                             for i,_ in ATTENTION_LAYERS))

    def test_zero_delta_baseline_identical_and_hook_cleanup_on_error(self):
        model=self.fake_model()
        import json as js
        vector=[((j%7)-3)*0.04 for j in range(384)]
        run=subprocess.run([str(self.binary),"--synthetic-only","0"],
            input=js.dumps(vector).encode(),capture_output=True,check=True,timeout=8)
        plan=strict_json(run.stdout)
        input_states=torch.randn(1,1,2048)
        original=model.model.layers[2].self_attn.gate_proj(input_states)
        with synthetic_head_gate_hooks(model,plan,self.config_bytes,self.modeling_bytes,enabled=True):
            self.assertTrue(torch.equal(
                model.model.layers[2].self_attn.gate_proj(input_states),original))
        bad=make_plan(self.binary)
        # Force a failure AFTER the first valid hook was registered.
        model.model.layers[4].self_attn.gate_proj=torch.nn.Linear(64,16)
        with self.assertRaises(GateHookRefused):
            with synthetic_head_gate_hooks(model,bad,self.config_bytes,self.modeling_bytes,enabled=True):pass
        self.assertFalse(model.model.layers[2].self_attn.gate_proj._forward_hooks)

if __name__=="__main__":unittest.main()
