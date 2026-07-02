from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSnapshot:
    python: str
    platform: str
    torch: str | None
    cuda_available: bool | None
    cuda_device_name: str | None


def collect_runtime_snapshot() -> RuntimeSnapshot:
    import platform
    import sys

    try:
        import torch
    except ImportError:
        return RuntimeSnapshot(
            python=sys.version,
            platform=platform.platform(),
            torch=None,
            cuda_available=None,
            cuda_device_name=None,
        )

    cuda_available = torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return RuntimeSnapshot(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
    )
