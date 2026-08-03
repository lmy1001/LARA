import enum
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


def _load_libero_policy():
    """Load the transform with lightweight stubs so this test has no OpenPI runtime dependency."""

    class DataTransformFn:
        pass

    class ModelType(enum.Enum):
        PI0 = "pi0"
        PI0_FAST = "pi0_fast"
        PI05 = "pi05"

    stubs = {
        "openpi": types.ModuleType("openpi"),
        "openpi.transforms": types.ModuleType("openpi.transforms"),
        "openpi.models": types.ModuleType("openpi.models"),
        "openpi.models.model": types.ModuleType("openpi.models.model"),
    }
    stubs["openpi.transforms"].DataTransformFn = DataTransformFn
    stubs["openpi.models.model"].ModelType = ModelType
    stubs["openpi"].transforms = stubs["openpi.transforms"]

    original_modules = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)

    module_name = "_release_test_libero_policy"
    policy_path = Path(__file__).resolve().parents[1] / "src/openpi/policies/libero_policy.py"
    spec = importlib.util.spec_from_file_location(module_name, policy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
    return module, ModelType


_libero_policy, ModelType = _load_libero_policy()
LiberoInputs = _libero_policy.LiberoInputs


def _sample_data(base_image: np.ndarray, wrist_image: np.ndarray) -> dict:
    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "pick up the object",
    }


def test_single_frame_inference_keeps_full_image() -> None:
    base_image = np.zeros((12, 16, 3), dtype=np.uint8)
    wrist_image = np.zeros((12, 16, 3), dtype=np.uint8)

    output = LiberoInputs(model_type=ModelType.PI05)(_sample_data(base_image, wrist_image))

    assert output["image"]["base_0_rgb"].shape == (12, 16, 3)
    assert output["image"]["right_wrist_0_rgb"].shape == (12, 16, 3)
    assert "base_0_rgb_future" not in output["image"]


def test_temporal_training_input_splits_current_and_future() -> None:
    base_image = np.stack(
        [
            np.full((3, 12, 16), 10, dtype=np.uint8),
            np.full((3, 12, 16), 20, dtype=np.uint8),
        ]
    )
    wrist_image = np.zeros((3, 12, 16), dtype=np.uint8)

    output = LiberoInputs(
        model_type=ModelType.PI05,
        include_future_frame=True,
    )(_sample_data(base_image, wrist_image))

    assert np.all(output["image"]["base_0_rgb"] == 10)
    assert np.all(output["image"]["base_0_rgb_future"] == 20)
    assert output["image"]["base_0_rgb"].shape == (12, 16, 3)
    assert output["image"]["base_0_rgb_future"].shape == (12, 16, 3)
