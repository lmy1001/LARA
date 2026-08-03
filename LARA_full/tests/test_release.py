import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PRIVATE_MARKERS = (
    "/" + "share/generalvision",
    "/" + "mnt/data",
    "/" + "vepfs-",
    "liu" + "mengya",
)

DEBUG_MARKERS = (
    "break" + "point(",
    "p" + "db." + "set_" + "trace(",
    "ip" + "db",
    "debug" + "py",
)

OBSOLETE_VISION_MARKERS = (
    "depth_" + "anything",
    "video_" + "depth",
    "optical_" + "flow",
    "optical " + "flow",
    "viz_" + "flow",
    "flow_" + "depth",
    "depth_" + "flow",
    "la" + "pa",
)


def test_yaml_files_parse():
    for path in REPO_ROOT.rglob("*.yaml"):
        with path.open(encoding="utf-8") as handle:
            yaml.safe_load(handle)


def test_release_tree_has_no_private_paths():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".yaml", ".yml", ".md", ".sh", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in PRIVATE_MARKERS), path


def test_python_tree_has_no_release_cleanup_residue():
    commented_debug = re.compile(
        r"^\s*#\s*(?:breakpoint|print|raise\s+ImportError)\s*\(",
        re.MULTILINE,
    )
    hash_banner = re.compile(r"^\s*#{2,}", re.MULTILINE)

    for path in REPO_ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert not any(marker.lower() in lowered for marker in DEBUG_MARKERS), path
        assert not any(
            marker.lower() in lowered for marker in OBSOLETE_VISION_MARKERS
        ), path
        assert commented_debug.search(text) is None, path
        assert hash_banner.search(text) is None, path


def test_tokenizer_data_uses_tokenizer_transform():
    config_path = REPO_ROOT / "moto/configs/data/tokenizer_pretrain_data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    targets = {
        section["data_config"]["_target_"]
        for section in config.values()
    }
    assert targets == {"moto.experiment.data_config.MotoTokenizerDataConfig"}


def test_tokenizer_launcher_uses_resolvable_hydra_config():
    launcher_text = (REPO_ROOT / "run_tokenizer.sh").read_text(encoding="utf-8")
    entrypoint_text = (REPO_ROOT / "scripts/pretrain_moto_tokenizer.py").read_text(
        encoding="utf-8"
    )

    assert "scripts/pretrain_moto_tokenizer.py --config-name train_tokenizer" in launcher_text
    assert '"$@"' in launcher_text
    assert 'config_path="../moto/configs"' in entrypoint_text


def test_data_configs_and_launchers_do_not_require_data_root_environment_variables():
    release_paths = (
        REPO_ROOT / "moto/configs/data/tokenizer_pretrain_data.yaml",
        REPO_ROOT / "lara_full/config/data/mani_test_data_all.yaml",
        REPO_ROOT / "lara_full/config/data/libero_10.yaml",
        REPO_ROOT / "run_tokenizer.sh",
        REPO_ROOT / "run_lara_full_pretrain.sh",
        REPO_ROOT / "run_libero_10.sh",
    )
    data_root_names = (
        "LARA_DATA_ROOT",
        "LARA_OXE_DATA_ROOT",
        "LARA_AGIBOT_DATA_ROOT",
        "LARA_G1_DATA_ROOT",
    )

    for path in release_paths:
        text = path.read_text(encoding="utf-8")
        assert not any(name in text for name in data_root_names), path


def test_lara_pretrain_launcher_uses_canonical_entrypoint():
    launcher_text = (REPO_ROOT / "run_lara_full_pretrain.sh").read_text(
        encoding="utf-8"
    )

    assert "scripts/gr00t_finetune.py --config-name mani_test_all" in launcher_text
    assert '"$@"' in launcher_text
    for required_name in (
        "LARA_BASE_MODEL_PATH",
        "LARA_TOKENIZER_PATH",
        "LARA_VIT_MAE_PATH",
    ):
        assert required_name in launcher_text


def test_libero_stage3_continues_from_stage2_checkpoint():
    launcher_text = (REPO_ROOT / "run_libero_10.sh").read_text(encoding="utf-8")
    experiment = yaml.safe_load(
        (REPO_ROOT / "lara_full/config/lara_posttrain_libero_10.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert (
        "scripts/gr00t_finetune.py --config-name lara_posttrain_libero_10"
        in launcher_text
    )
    assert '"$@"' in launcher_text
    assert "LARA_STAGE2_CHECKPOINT" in launcher_text
    assert "/latent_motion_tokenizer" in launcher_text
    assert experiment["base_model_path"] == "${oc.env:LARA_STAGE2_CHECKPOINT}"
    assert experiment["reinitialize_action_head"] is False
    assert experiment["resume"] is False
    assert experiment["tune_llm"] is False
    assert experiment["tune_visual"] is False
    assert experiment["tune_projector"] is True
    assert experiment["tune_diffusion_model"] is True
    assert experiment["tune_latent_motion_tokenizer"] is True
    assert experiment["latent_loss_weight"] == pytest.approx(0.01)
    assert experiment["tokenizer_vae_loss_weight"] == pytest.approx(0.01)
    assert "n1.6" not in launcher_text.lower()


def test_libero_stage3_data_contract():
    data = yaml.safe_load(
        (REPO_ROOT / "lara_full/config/data/libero_10.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert list(data) == ["data_1"]
    config = data["data_1"]["data_config"]
    assert config["_target_"] == "lara_full.experiment.data_config.LiberoDataConfig"
    assert config["video_keys"] == ["video.image", "video.wrist_image"]
    assert config["state_keys"] == [
        "state.position",
        "state.orientation",
        "state.gripper",
    ]
    assert config["action_keys"] == [
        "action.position",
        "action.orientation",
        "action.gripper",
    ]
    assert config["observation_indices"] == [0, 15]
    assert config["action_indices"] == list(range(16))
    assert config["embodiment_tag"] == "libero"


def test_multiview_training_routes_primary_camera_to_tokenizer():
    transform_text = (
        REPO_ROOT / "lara_full/model/transforms.py"
    ).read_text(encoding="utf-8")

    assert 'batch_data = {"images": images[:, :1], "language": language}' in transform_text
    assert "images = images[:1].astype(np.float32) / 255.0" in transform_text


def test_lara_supervision_uses_tokenizer_embed():
    action_head_text = (
        REPO_ROOT / "lara_full/model/action_head/flow_matching_action_head.py"
    ).read_text(encoding="utf-8")

    assert "target_motion_embed = tokenizer_outputs.embed.reshape(" in action_head_text
    assert "tokenizer_outputs.quant" not in action_head_text


def test_lara_checkpoint_builds_latent_head_before_loading_weights():
    model_text = (REPO_ROOT / "lara_full/model/gr00t_n1.py").read_text(
        encoding="utf-8"
    )

    config_update = model_text.index("model_config.action_head_cfg = action_head_cfg")
    weight_loading = model_text.index("pretrained_model = super().from_pretrained(")
    assert config_update < weight_loading
    assert 'kwargs["config"] = model_config' in model_text


def test_tokenizer_collator_contract():
    pytest.importorskip("torch")
    from moto.utils.collators import collate_moto_tokenizer

    sample = {"video": np.zeros((2, 1, 8, 8, 3), dtype=np.uint8)}
    batch = collate_moto_tokenizer([sample, sample])
    assert batch["rgb_initial"].shape == (2, 1, 3, 8, 8)
    assert batch["rgb_future"].shape == (2, 1, 3, 8, 8)
