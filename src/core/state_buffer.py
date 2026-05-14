from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from validation.verification.energy_conservation import GAMMA, SMALL, _cc_B


def _copy_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in mapping.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


@dataclass(frozen=True)
class StateSnapshot:
    """MHD 側から公開された最新スナップショット。"""

    mhd_step: int
    simulation_time: float
    density: np.ndarray
    pressure: np.ndarray
    velocity: dict[str, np.ndarray]
    magnetic_field: dict[str, np.ndarray]
    conserved_state: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_advection_inputs(self, *, copy: bool = True) -> dict[str, Any]:
        """
        Advection 層が直接読む入力を返す。

        低頻度 MHD が更新された時点で、速度場・磁場・密度・圧力をこの形式で確定し、
        高頻度 Advection 層は次の MHD 更新まで同じスナップショットを読み続ける。
        """

        def maybe_copy(value: Any) -> Any:
            return value.copy() if copy and isinstance(value, np.ndarray) else value

        return {
            "source_mhd_step": self.mhd_step,
            "simulation_time": self.simulation_time,
            "density": maybe_copy(self.density),
            "pressure": maybe_copy(self.pressure),
            "velocity": {name: maybe_copy(arr) for name, arr in self.velocity.items()},
            "magnetic_field": {
                name: maybe_copy(arr) for name, arr in self.magnetic_field.items()
            },
            "metadata": dict(self.metadata),
        }


class StateBuffer:
    """低頻度 MHD と高頻度 Advection の間で共有する状態バッファ。"""

    def __init__(self) -> None:
        self._latest_snapshot: StateSnapshot | None = None

    @property
    def latest_snapshot(self) -> StateSnapshot | None:
        return self._latest_snapshot

    def publish_mhd_state(
        self,
        conserved_state: Mapping[str, Any],
        *,
        mhd_step: int,
        simulation_time: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> StateSnapshot:
        """
        保存変数から Advection 用の原始量を作り、最新スナップショットとして公開する。
        """

        rho = np.asarray(conserved_state["rho"], dtype=np.float64)
        mx = np.asarray(conserved_state["mx"], dtype=np.float64)
        my = np.asarray(conserved_state["my"], dtype=np.float64)
        mz = np.asarray(conserved_state["mz"], dtype=np.float64)
        energy = np.asarray(conserved_state["E"], dtype=np.float64)
        Bx_f = np.asarray(conserved_state["Bx_f"], dtype=np.float64)
        By_f = np.asarray(conserved_state["By_f"], dtype=np.float64)
        Bz_f = np.asarray(conserved_state["Bz_f"], dtype=np.float64)

        Bx, By, Bz = _cc_B(Bx_f, By_f, Bz_f)
        inv_rho = 1.0 / np.maximum(rho, SMALL)
        vx = mx * inv_rho
        vy = my * inv_rho
        vz = mz * inv_rho
        kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)
        magnetic = 0.5 * (Bx**2 + By**2 + Bz**2)
        pressure = np.maximum((GAMMA - 1.0) * (energy - kinetic - magnetic), SMALL)

        snapshot = StateSnapshot(
            mhd_step=mhd_step,
            simulation_time=simulation_time,
            density=rho.copy(),
            pressure=pressure.copy(),
            velocity={"vx": vx.copy(), "vy": vy.copy(), "vz": vz.copy()},
            magnetic_field={"Bx": Bx.copy(), "By": By.copy(), "Bz": Bz.copy()},
            conserved_state=_copy_mapping(conserved_state),
            metadata=dict(metadata or {}),
        )
        self._latest_snapshot = snapshot
        return snapshot

    def get_advection_inputs(self, *, copy: bool = True) -> dict[str, Any]:
        if self._latest_snapshot is None:
            raise RuntimeError("StateBuffer is empty. Publish an MHD state first.")
        return self._latest_snapshot.to_advection_inputs(copy=copy)
