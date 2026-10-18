"""Observe actual V2 loss gradients during explicitly enabled preflight only.

The preflight wrapper gates install() with CMERC_GRADIENT_PROBE. Importing this
module does nothing. No forward, optimizer step, or persistent model edit occurs.
"""

import inspect
import json
import math
import os
from pathlib import Path
import random
import sys


def install(output_path):
    """Observe the first two CMERC training backwards, then restore backward.

    Call only from the CMERC_GRADIENT_PROBE preflight branch. An optional
    CMERC_GRADIENT_BATCHES integer changes the bounded sample count (1..8).
    Returns an idempotent uninstall callback; it can also end observation early.
    """
    import torch

    if getattr(torch.Tensor.backward, "_cmerc_gradient_observer", False):
        raise RuntimeError("a CMERC gradient observer is already installed")
    limit = int(os.environ.get("CMERC_GRADIENT_BATCHES", "2"))
    if not 1 <= limit <= 8:
        raise ValueError("CMERC_GRADIENT_BATCHES must be between 1 and 8")
    output = Path(output_path)
    original_backward = torch.Tensor.backward
    sampled = 0
    busy = False

    def emit(record):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def uninstall():
        if torch.Tensor.backward is observed_backward:
            torch.Tensor.backward = original_backward

    def training_frame():
        current = inspect.currentframe()
        try:
            candidate = current.f_back
            for _ in range(12):
                if candidate is None:
                    break
                if (candidate.f_code.co_name == "train_or_eval_graph_model"
                        and Path(candidate.f_code.co_filename).name == "train_cm.py"
                        and candidate.f_locals.get("train") is True):
                    return candidate
                candidate = candidate.f_back
            return None
        finally:
            del current

    def component(loss, hidden, enabled, coefficient):
        is_tensor = torch.is_tensor(loss)
        record = {
            "enabled": bool(enabled),
            "coefficient": float(coefficient),
            "requires_grad": bool(is_tensor and loss.requires_grad),
            "tensor_shape": list(loss.shape) if is_tensor else None,
            "hidden_requires_grad": bool(hidden.requires_grad),
            "gradient_shape": None,
            "norm": None,
            "finite": None,
            "nonzero": False,
        }
        if not enabled:
            record["status"] = "disabled"
        elif not is_tensor or not loss.requires_grad:
            record["status"] = "constant_or_conditional_zero"
        elif not hidden.requires_grad:
            record["status"] = "hidden_not_differentiable"
        else:
            try:
                gradient = torch.autograd.grad(
                    loss, hidden, retain_graph=True, create_graph=False,
                    allow_unused=True,
                )[0]
                if gradient is None:
                    record["status"] = "unused_hidden"
                else:
                    detached = gradient.detach()
                    norm = float(detached.float().norm().item())
                    finite = bool(torch.isfinite(detached).all().item())
                    record.update({
                        "status": "observed",
                        "gradient_shape": list(detached.shape),
                        "norm": norm if math.isfinite(norm) else None,
                        "finite": finite and math.isfinite(norm),
                        "nonzero": finite and math.isfinite(norm) and norm > 0,
                    })
            except Exception as error:
                record.update(status="observer_error", error=str(error))
        return record

    def observe(frame, sample_index):
        local = frame.f_locals
        config = frame.f_globals.get("args")
        hidden = local.get("hidden")
        if not torch.is_tensor(hidden):
            raise RuntimeError("actual training frame has no tensor hidden")
        numpy = sys.modules.get("numpy")
        saved_python = random.getstate()
        saved_numpy = numpy.random.get_state() if numpy is not None else None
        saved_torch = torch.get_rng_state()
        saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        record = {
            "schema": "cmerc_actual_loss_gradients_v1",
            "pid": os.getpid(),
            "source": str(frame.f_code.co_filename),
            "sample_index": sample_index,
            "epoch": local.get("epoch"),
            "seed": getattr(config, "seed", None),
            "rng_policy": getattr(config, "rng_policy", None),
            "batch_policy": getattr(config, "batch_policy", None),
            "hidden_shape": list(hidden.shape),
            "hidden_requires_grad": bool(hidden.requires_grad),
            "components": {},
        }
        specifications = (
            ("classification", "classification_loss", True, 1.0),
            ("cc", "rank_loss", getattr(config, "calibrate", False),
             getattr(config, "rank_coff", 0.0)),
            ("mscl", "Mscl_tav", getattr(config, "contrastlearning", False),
             getattr(config, "mscl_coff", 0.0)),
            ("cscl", "Cscl_c", getattr(config, "contrastlearning", False),
             getattr(config, "cscl_coff", 0.0)),
        )
        try:
            for name, variable, enabled, coefficient in specifications:
                record["components"][name] = component(
                    local.get(variable), hidden, enabled, coefficient,
                )
        finally:
            random.setstate(saved_python)
            if saved_numpy is not None:
                numpy.random.set_state(saved_numpy)
            torch.set_rng_state(saved_torch)
            if saved_cuda is not None:
                torch.cuda.set_rng_state_all(saved_cuda)
        record["rng_preserved"] = True
        emit(record)

    def observed_backward(tensor, *positional, **keywords):
        nonlocal sampled, busy
        frame = None
        if not busy and sampled < limit:
            frame = training_frame()
        if frame is not None:
            try:
                config = frame.f_globals.get("args")
                is_cmerc = (getattr(config, "calibrate", False)
                            or getattr(config, "contrastlearning", False))
                if is_cmerc:
                    sampled += 1
                    busy = True
                    try:
                        observe(frame, sampled)
                    except Exception as error:
                        emit({"schema": "cmerc_actual_loss_gradients_v1",
                              "sample_index": sampled, "observer_error": str(error)})
                    finally:
                        busy = False
                    if sampled >= limit:
                        uninstall()
            finally:
                del frame
        return original_backward(tensor, *positional, **keywords)

    observed_backward._cmerc_gradient_observer = True
    torch.Tensor.backward = observed_backward
    return uninstall
