import os
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """加载配置文件，支持环境变量覆盖。"""
    if config_path is None:
        # 优先查找 config.local.yaml，其次 config.yaml
        base = Path(__file__).parent.parent
        local = base / "config.local.yaml"
        config_path = local if local.exists() else base / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 将相对路径转为绝对路径（相对于配置文件所在目录）
    base_dir = config_path.parent
    if cfg.get("notes_dir"):
        notes_path = Path(cfg["notes_dir"])
        if not notes_path.is_absolute():
            cfg["notes_dir"] = str((base_dir / notes_path).resolve())
    if cfg.get("index_dir"):
        index_path = Path(cfg["index_dir"])
        if not index_path.is_absolute():
            cfg["index_dir"] = str((base_dir / index_path).resolve())

    # 环境变量覆盖
    env_api_key = os.environ.get("LLM_API_KEY")
    if env_api_key:
        cfg.setdefault("llm", {})["api_key"] = env_api_key

    return cfg
