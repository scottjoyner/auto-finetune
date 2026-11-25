"""Exercise VRAM-based recommendations without querying real devices."""
import json
from types import SimpleNamespace

import pytest
from src import auto_scaler as scaler


GIB=1024**3


def test_rocm_json_is_preferred_over_nvidia(monkeypatch):
    seen=[]
    def fake_run(args,**kwargs):
        seen.append(args[0])
        if args[0] != "rocm-smi":
            raise AssertionError("should not access secondary device backend")
        return SimpleNamespace(returncode=0,stdout=json.dumps({
            "card0":{"VRAM Total (B)":16*GIB,"VRAM Used (B)":6*GIB}
        }))
    monkeypatch.setattr(scaler.subprocess,"run",fake_run)
    total,free=scaler.detect_vram()
    assert (total,free)==(16,10)
    assert seen==["rocm-smi"]


def test_cuda_csv_fallback_when_rocm_unavailable(monkeypatch):
    def fake_run(args,**kwargs):
        if args[0]=="rocm-smi":
            raise OSError("rocm unavailable")
        return SimpleNamespace(returncode=0,stdout="24576, 16384\n")
    monkeypatch.setattr(scaler.subprocess,"run",fake_run)
    assert scaler.detect_vram()==(24,16)


def test_fallback_when_all_device_tools_unavailable(monkeypatch):
    monkeypatch.setattr(scaler.subprocess,"run",
                        lambda *_a,**_k: (_ for _ in ()).throw(FileNotFoundError("no tools")))
    fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
    monkeypatch.setitem(__import__("sys").modules,"torch",fake_torch)
    assert scaler.detect_vram()==(15.0,8.0)


def test_torch_fallback_without_shell_tools(monkeypatch):
    monkeypatch.setattr(scaler.subprocess,"run",
                        lambda *_a,**_k: (_ for _ in ()).throw(OSError("no cli")))
    cuda=SimpleNamespace(
        is_available=lambda:True,
        get_device_properties=lambda i:SimpleNamespace(total_memory=32*GIB),
    )
    monkeypatch.setitem(__import__("sys").modules,"torch",SimpleNamespace(cuda=cuda))
    assert scaler.detect_vram()==(32,pytest.approx(25.6))


@pytest.mark.parametrize("available,expected_batch,expected_accum,expected_seq",[
    (12,4,2,128),
    (7,2,4,128),
    (5.5,1,8,64),
    (2.5,1,16,32),
])
def test_throughput_mode_selects_memory_band(available,expected_batch,expected_accum,expected_seq):
    config=scaler.recommend(available_vram_gb=available,total_vram_gb=16,
                            model_params_b=.5,prefer_quality=False)
    assert config.per_device_batch_size==expected_batch
    assert config.gradient_accumulation_steps==expected_accum
    assert config.max_seq_length==expected_seq
    assert config.throughput_estimate>0


@pytest.mark.parametrize("available,expected_seq",[
    (12,512),
    (7,256),
    (5.5,128),
    (2.5,64),
])
def test_quality_mode_selects_longer_sequences(available,expected_seq):
    config=scaler.recommend(available_vram_gb=available,total_vram_gb=16,
                            model_params_b=.5,prefer_quality=True)
    assert config.max_seq_length==expected_seq
    assert config.per_device_batch_size==1


def test_lfm3_branch_and_large_model_quantization():
    lfm=scaler.recommend(available_vram_gb=15,total_vram_gb=16,
                        model_params_b=1.7,model_name="LiquidAI/LFM3-1.7B")
    assert lfm.max_seq_length==4096
    assert lfm.gradient_accumulation_steps==16
    assert lfm.load_in_4bit is False
    large=scaler.recommend(available_vram_gb=24,total_vram_gb=32,model_params_b=4.0)
    assert large.load_in_4bit is True


def test_detect_called_only_when_parameter_missing(monkeypatch):
    count=[]
    def detect():
        count.append("detect")
        return 16.,10.
    monkeypatch.setattr(scaler,"detect_vram",detect)
    assert scaler.recommend(available_vram_gb=12,total_vram_gb=16).available_vram_gb==12
    assert count==[]
    result=scaler.recommend(available_vram_gb=None,total_vram_gb=16)
    assert result.available_vram_gb==10
    assert count==["detect"]


def test_cli_json_and_env_output_without_device_probe(monkeypatch,capsys):
    monkeypatch.setattr(scaler,"detect_vram",lambda:(16.,12.))
    assert scaler.main(["--model-size",".5","--json"])==0
    data=json.loads(capsys.readouterr().out)
    assert data["available_vram_gb"]==12
    assert scaler.main(["--model-size",".5","--prefer-quality"])==0
    text=capsys.readouterr().out
    assert "TRAIN_MAX_SEQ_LENGTH=" in text
    assert "TRAIN_LOAD_4BIT=" in text
