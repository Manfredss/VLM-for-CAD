#!/usr/bin/env python3
"""Offline preflight checks for Qwen3.6/3.5 CAD fine-tuning.

This command only inspects the local Python environment and local config/model
files.  It never downloads a model and never installs or upgrades packages.
For Qwen3.6 it enforces the currently validated stack: ms-swift >= 4.1.3,
transformers >= 5.0.0.dev0, qwen-vl-utils >= 0.0.14, and decord.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    from .cad_schema import DEFAULT_MODEL, FALLBACK_MODEL
except ImportError:
    from cad_schema import DEFAULT_MODEL, FALLBACK_MODEL  # type: ignore


def _version(value: str) -> Any:
    try:
        from packaging.version import Version

        return Version(value)
    except (ImportError, ValueError):
        numbers = tuple(int(token) for token in re.findall(r"\d+", value)[:3])
        numbers = numbers + (0,) * (3 - len(numbers))
        stage = (
            -1 if re.search(r"(?:dev|a|alpha|b|beta|rc)", value, re.IGNORECASE) else 0
        )
        return (*numbers, stage)


def _distribution_version(names: list[str]) -> Optional[str]:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _dependency(
    name: str, distributions: list[str], module: str, minimum: Optional[str] = None
) -> dict[str, Any]:
    version = _distribution_version(distributions)
    module_found = importlib.util.find_spec(module) is not None
    installed = version is not None or module_found
    compatible = installed and (
        minimum is None
        or (version is not None and _version(version) >= _version(minimum))
    )
    if not installed:
        message = "not installed"
    elif minimum and version is None:
        message = (
            f"module found, but distribution version is unknown (need >= {minimum})"
        )
    elif minimum and not compatible:
        message = f"found {version}; need >= {minimum}"
    else:
        message = f"found {version or 'unknown version'}"
    return {
        "name": name,
        "module": module,
        "installed": installed,
        "version": version,
        "minimum": minimum,
        "compatible": bool(compatible),
        "message": message,
    }


def check_dependencies(model: str) -> list[dict[str, Any]]:
    qwen36 = "qwen3.6" in model.casefold() or "qwen3_6" in model.casefold()
    requirements = [
        ("ms-swift", ["ms-swift"], "swift", "4.1.3" if qwen36 else None),
        (
            "transformers",
            ["transformers"],
            "transformers",
            "5.0.0.dev0" if qwen36 else None,
        ),
        (
            "qwen-vl-utils",
            ["qwen-vl-utils", "qwen_vl_utils"],
            "qwen_vl_utils",
            "0.0.14" if qwen36 else None,
        ),
        ("decord", ["decord"], "decord", None),
        ("torch", ["torch"], "torch", None),
        ("Pillow", ["Pillow"], "PIL", None),
        ("peft", ["peft"], "peft", None),
        ("deepspeed", ["deepspeed"], "deepspeed", None),
    ]
    return [_dependency(*requirement) for requirement in requirements]


def _simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        match = re.match(r"([A-Za-z_][\w.-]*)\s*:\s*(.+)$", stripped)
        if match:
            result[match.group(1)] = match.group(2).strip().strip("'\"")
    return result


def load_config(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, str(exc)
    if path.suffix.casefold() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"JSON line {exc.lineno}: {exc.msg}"
        return (
            (dict(payload), None)
            if isinstance(payload, Mapping)
            else (None, "top-level config must be an object")
        )
    if path.suffix.casefold() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text)
            return (
                (dict(payload), None)
                if isinstance(payload, Mapping)
                else (None, "top-level config must be a mapping")
            )
        except ImportError:
            return _simple_yaml(text), None
        except Exception as exc:  # PyYAML exposes several parser exception types.
            return None, str(exc)
    if path.suffix.casefold() in {".sh", ".bash", ".zsh"}:
        result: dict[str, Any] = {}
        model_match = re.search(
            r"(?:--model(?:_id_or_path)?\s+|\bMODEL(?:_ID)?=)(['\"]?)([^\s'\"]+)\1",
            text,
        )
        if model_match:
            result["model"] = model_match.group(2)
        return result, None
    return None, f"unsupported config extension: {path.suffix or '<none>'}"


def _find_values(value: Any, names: set[str]) -> list[Any]:
    found = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in names:
                found.append(nested)
            found.extend(_find_values(nested, names))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_find_values(nested, names))
    return found


def check_model_and_config(model: str, config_path: Optional[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested_model": model,
        "is_qwen36": "qwen3.6" in model.casefold() or "qwen3_6" in model.casefold(),
        "is_qwen35": "qwen3.5" in model.casefold() or "qwen3_5" in model.casefold(),
        "local_model_checked": False,
        "warnings": [],
        "errors": [],
    }
    local = Path(model).expanduser()
    if local.exists():
        model_config_path = local / "config.json" if local.is_dir() else local
        model_config, error = load_config(model_config_path)
        if error:
            result["errors"].append(f"model config: {error}")
        else:
            result["local_model_checked"] = True
            result["model_config_path"] = str(model_config_path)
            result["model_type"] = (
                model_config.get("model_type") if model_config else None
            )
            architectures = (
                model_config.get("architectures", []) if model_config else []
            )
            result["architectures"] = architectures
            identity = f"{result.get('model_type')} {architectures}".casefold()
            if "qwen" not in identity:
                result["warnings"].append(
                    "local config does not identify a Qwen architecture"
                )
    else:
        result["remote_model_note"] = (
            "Model is a Hub ID; config was not fetched (offline preflight)."
        )

    if not result["is_qwen36"] and not result["is_qwen35"]:
        result["warnings"].append(
            f"model name is neither recognized Qwen3.6 nor Qwen3.5 fallback ({FALLBACK_MODEL})"
        )

    if config_path:
        if not config_path.exists():
            result["errors"].append(f"training config not found: {config_path}")
        else:
            config, error = load_config(config_path)
            if error:
                result["errors"].append(f"training config: {error}")
            else:
                result["training_config_path"] = str(config_path)
                result["training_config_loaded"] = True
                configured_models = _find_values(
                    config,
                    {"model", "model_id", "model_id_or_path", "model_name_or_path"},
                )
                result["configured_models"] = [
                    str(value) for value in configured_models
                ]
                if configured_models and all(
                    str(value) != model for value in configured_models
                ):
                    result["warnings"].append(
                        f"CLI model {model!r} differs from config model(s) {configured_models!r}"
                    )
                train_types = _find_values(
                    config, {"train_type", "tuner_type", "finetuning_type"}
                )
                if train_types and not any(
                    "lora" in str(value).casefold() for value in train_types
                ):
                    result["warnings"].append(
                        f"27B limited-data workflow expects LoRA; config reports {train_types!r}"
                    )
    return result


def check_hardware(skip: bool = False, expected_gpus: int = 3) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked": False,
        "cuda_available": False,
        "devices": [],
        "expected_gpus": expected_gpus,
        "gpu_count_sufficient": False,
        "bf16_supported": False,
    }
    if skip:
        result["note"] = "hardware check skipped"
        return result
    try:
        import torch
    except Exception as exc:
        result["error"] = f"cannot import torch: {exc}"
        return result
    result["checked"] = True
    result["torch_version"] = getattr(torch, "__version__", None)
    result["cuda_available"] = bool(torch.cuda.is_available())
    if result["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / (1024**3), 2),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        result["gpu_count_sufficient"] = len(result["devices"]) >= expected_gpus
        try:
            result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except (AttributeError, TypeError):
            result["bf16_supported"] = bool(result["devices"]) and all(
                device["capability"][0] >= 8 for device in result["devices"]
            )
    else:
        result["note"] = (
            "CUDA is unavailable; static/data audit still works, but 27B training does not."
        )
    result["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hub ID or local model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--config", type=Path, help="Optional ms-swift YAML/JSON or train shell script."
    )
    parser.add_argument("--json-out", "--output", dest="json_out", type=Path)
    parser.add_argument("--skip-hardware", action="store_true")
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=3,
        help="Minimum visible CUDA devices for the default 27B ZeRO-3 launch",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero for missing/incompatible requirements.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_gpus < 1:
        raise SystemExit("--expected-gpus must be at least 1")
    dependencies = check_dependencies(args.model)
    model_config = check_model_and_config(args.model, args.config)
    hardware = check_hardware(args.skip_hardware, args.expected_gpus)
    required_failures = [
        dependency for dependency in dependencies if not dependency["compatible"]
    ]
    errors = [dependency["message"] for dependency in required_failures] + list(
        model_config["errors"]
    )
    warnings = list(model_config["warnings"])
    if not args.skip_hardware:
        if not hardware.get("cuda_available"):
            errors.append("CUDA unavailable for 27B training")
        else:
            if not hardware.get("gpu_count_sufficient"):
                errors.append(
                    "insufficient visible GPUs: "
                    f"found {len(hardware.get('devices', []))}, need {args.expected_gpus}"
                )
            if not hardware.get("bf16_supported"):
                errors.append("visible CUDA hardware does not support bfloat16")
    report = {
        "status": "fail" if errors else ("warning" if warnings else "pass"),
        "offline": True,
        "auto_install_performed": False,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "model": model_config,
        "dependencies": dependencies,
        "hardware": hardware,
        "errors": errors,
        "warnings": warnings,
        "qwen36_stack_note": "Validated baseline: ms-swift 4.1.3, transformers>=5.0.0.dev0, qwen-vl-utils>=0.0.14, decord. No package is installed by this script.",
    }

    print(f"Preflight: {report['status'].upper()}  model={args.model}")
    for dependency in dependencies:
        marker = "OK" if dependency["compatible"] else "FAIL"
        requirement = (
            f">={dependency['minimum']}" if dependency["minimum"] else "installed"
        )
        print(
            f"  [{marker:4s}] {dependency['name']:15s} {dependency['version'] or '-':14s} required={requirement}"
        )
    if hardware.get("checked"):
        print(
            f"  CUDA: {hardware.get('cuda_available')}  "
            f"devices={len(hardware.get('devices', []))}/{args.expected_gpus}  "
            f"bf16={hardware.get('bf16_supported')}"
        )
    for error in errors:
        print(f"  ERROR: {error}")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    print("  No downloads or installations were performed.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Report: {args.json_out}")
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
