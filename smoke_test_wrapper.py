# LOCAL SMOKE TEST ONLY — not uploaded to remote, not imported by test.py.
#
# torch==1.11.0 (local `forecast_mae` env) strictly rejects plain Tensor
# values in nn.ParameterDict.__setitem__ (see torch/nn/modules/module.py
# register_parameter type check). test.py's actor_embeds construction
# (line 154/180/184/203) passes plain Tensors sliced from a .clone().detach()
# Parameter, which is fine on newer torch (auto-wraps) but crashes here.
# This is an environment artifact, not a bug in test.py — patch it only
# for local runs so test.py itself stays byte-identical to what's on remote.
import torch

_orig_setitem = torch.nn.ParameterDict.__setitem__


def _patched_setitem(self, key, value):
    if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
        value = torch.nn.Parameter(value)
    return _orig_setitem(self, key, value)


torch.nn.ParameterDict.__setitem__ = _patched_setitem

if __name__ == "__main__":
    import runpy
    # run as __main__ (not `import test`) so hydra's config_path, which is
    # resolved relative to test.py's own __file__, works the same as a
    # direct `python test.py` invocation.
    runpy.run_path("test.py", run_name="__main__")
