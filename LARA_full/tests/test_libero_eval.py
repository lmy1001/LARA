import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lara_full.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_eval_example():
    pytest.importorskip("msgpack")
    pytest.importorskip("zmq")
    path = REPO_ROOT / "examples/libero/run_libero_eval.py"
    spec = importlib.util.spec_from_file_location("lara_libero_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_libero_embodiment_matches_released_projector():
    assert EMBODIMENT_TAG_MAPPING[EmbodimentTag.Libero.value] == 43


def test_libero_config_keeps_two_views_and_mean_std_contract():
    config_text = (REPO_ROOT / "lara_full/experiment/data_config.py").read_text(
        encoding="utf-8"
    )
    assert 'video_keys = ["video.image", "video.wrist_image"]' in config_text
    assert '"state.position": "mean_std"' in config_text
    assert '"action.position": "mean_std"' in config_text


def test_stage3_checkpoint_validation_and_readme_entrypoints():
    server_text = (REPO_ROOT / "scripts/inference_service.py").read_text(
        encoding="utf-8"
    )
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert 'config.get("model_type") != "gr00t_n1_5"' in server_text
    assert 'metadata_path = checkpoint / "experiment_cfg" / "metadata.json"' in server_text
    assert 'if "libero" not in metadata' in server_text
    assert "scripts/inference_service.py" in readme_text
    assert "examples/libero/run_libero_eval.py" in readme_text
    assert '${LARA_POSTTRAIN_OUTPUT_DIR}/checkpoint-<STEP>' in readme_text


def test_msgpack_serializer_round_trip():
    pytest.importorskip("msgpack")
    pytest.importorskip("zmq")
    from lara_full.eval.service import MsgpackSerializer

    source = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
        "scalar": np.float32(1.25),
    }
    restored = MsgpackSerializer.from_bytes(MsgpackSerializer.to_bytes(source))
    np.testing.assert_array_equal(restored["image"], source["image"])
    assert restored["scalar"] == pytest.approx(1.25)


def test_libero_action_uses_second_chunk_element():
    module = _load_eval_example()
    action_chunk = {
        "action.position": np.array([[9, 9, 9], [1, 2, 3]], dtype=np.float32),
        "action.orientation": np.array([[9, 9, 9], [4, 5, 6]], dtype=np.float32),
        "action.gripper": np.array([[0.0], [0.25]], dtype=np.float32),
    }
    action = module.to_libero_action(action_chunk)
    np.testing.assert_array_equal(action, np.array([1, 2, 3, 4, 5, 6, 1]))
