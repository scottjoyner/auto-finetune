"""Repo-local Triton kernels for the Agnes delta-rule path.

This package shares a name with the HuggingFace ``kernels`` library, and the
repo root is on sys.path whenever training runs from here. transformers>=5
probes for ``kernels`` during import (``is_kernels_available``), so an
ImportError raised here takes down ``import transformers`` entirely -- which is
how this module became load-bearing for a model that has nothing to do with it.

The submodules are not all equally healthy: ``agnes_patch`` imports a
``kvcache`` module that does not exist in this tree, and defines
``apply_triton_patch``/``integrate_kv_cache`` rather than the ``apply_patch``/
``revert_patch`` names previously re-exported here. So the imports are guarded
and only the names that actually resolve are exported, instead of fabricating
aliases that would fail later at call time.

The Triton entry points are plain modules with no such dependencies, so they
stay unconditional.
"""
import logging

from kernels.delta_rule_triton import (
    causal_conv1d_triton,
    delta_rule_chunked_triton,
    delta_rule_stepwise_triton,
)

__all__ = [
    "causal_conv1d_triton",
    "delta_rule_chunked_triton",
    "delta_rule_stepwise_triton",
]

for _name, _module in (("apply_triton_patch", "agnes_patch"),
                      ("integrate_kv_cache", "agnes_patch"),
                      ("benchmark_kv_cache", "benchmark_kv_cache")):
    try:
        _obj = getattr(__import__(f"kernels.{_module}", fromlist=[_name]), _name)
    except Exception as exc:  # optional, and known-broken for agnes_patch
        logging.getLogger(__name__).debug(
            "kernels: %s.%s unavailable (%s)", _module, _name, exc)
    else:
        globals()[_name] = _obj
        __all__.append(_name)