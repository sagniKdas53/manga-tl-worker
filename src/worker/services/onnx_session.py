"""One place that decides how the worker's local ONNX models (YOLO, CTD, AOT) are run.

Measured 2026-09-26 on the pipeline's own model files, each setting in a fresh process:

- Stock ONNX Runtime took 34 s for a 580x700 CTD crop (about 70 us per fed pixel, the same on
  chrome-box). The time is denormal floats: the graph's ``ConvTranspose`` layers underflow and the
  CPU crawls through the subnormal arithmetic. Flushing denormals to zero
  (``session.set_denormal_as_zero``) brings the same crop to 0.6-0.8 s on stock ORT 1.30 and
  a 1024x1024 crop to 1.5-1.9 s, with identical masks.
- Intel's OpenVINO execution provider (the ``onnxruntime-openvino`` build, same ``onnxruntime``
  module, x86-64 only) adds about 1.5x on CTD (0.42 s / 1.16 s) and 1.3x on AOT, with
  identical masks, AOT fills within one grey level, and the same YOLO balloons. It pays no
  recompile for a new input shape.

So every session flushes denormals, and x86-64 uses OpenVINO when the build has it. Elsewhere
(arm64, developer machines) the stock provider remains, so this falls back rather than fails.
``ONNX_EXECUTION_PROVIDER=cpu`` restores the stock provider without a rebuild.

Measure in fresh processes: a session without the flag, run earlier in the same process, left a
flagged session at half speed instead of full speed (seen 2026-09-26).
"""

import logging

from worker.config import ONNX_EXECUTION_PROVIDER

logger = logging.getLogger(__name__)

OPENVINO = "OpenVINOExecutionProvider"
CPU = "CPUExecutionProvider"


def execution_providers(ort) -> list:
    """The provider list for ``ort.InferenceSession``: OpenVINO on CPU when wanted and present."""
    if ONNX_EXECUTION_PROVIDER == "cpu":
        return [CPU]
    if OPENVINO in ort.get_available_providers():
        return [(OPENVINO, {"device_type": "CPU"}), CPU]
    if ONNX_EXECUTION_PROVIDER == "openvino":
        logger.warning("[ONNX] ONNX_EXECUTION_PROVIDER=openvino but this onnxruntime build has no OpenVINO provider")
    return [CPU]


def create_session(ort, model_path: str, label: str):
    """Build an inference session for ``model_path``, falling back to the stock CPU provider.

    Denormals are flushed to zero in every session: on the stock provider that alone takes CTD from
    about 70 to about 1.5 us per fed pixel, and outputs were bit-identical on CTD and AOT.
    """
    options = ort.SessionOptions()
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    providers = execution_providers(ort)
    try:
        session = ort.InferenceSession(model_path, options, providers=providers)
    except Exception as e:
        if providers == [CPU]:
            raise
        logger.warning(f"[{label}] OpenVINO session failed ({e}); falling back to the CPU provider")
        session = ort.InferenceSession(model_path, options, providers=[CPU])
    try:
        active = session.get_providers()[0]
    except Exception:
        active = "unknown"
    logger.info(f"[{label}] ONNX Runtime session on {active}")
    return session
