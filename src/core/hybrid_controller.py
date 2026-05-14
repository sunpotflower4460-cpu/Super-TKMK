from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from validation.verification.energy_conservation import (
    _cfl_dt,
    make_torus_ic_python,
    step as advance_ct_hlle_rk2,
)

from src.core.state_buffer import StateBuffer, StateSnapshot


class MHDSolverAdapter(Protocol):
    """HybridController が期待する低頻度 MHD ソルバの最小インターフェース。"""

    def make_initial_state(self) -> dict[str, Any]:
        ...

    def compute_dt(self, state: dict[str, Any]) -> float:
        ...

    def advance(self, state: dict[str, Any], dt: float) -> dict[str, Any]:
        ...


@dataclass
class ToroidalMHDSolverAdapter:
    """
    toroidal_mhd_test.py が使っている Python MHD ソルバを再利用するアダプタ。

    実体は validation.verification.energy_conservation.py の
    CT + HLLE + SSP-RK2 実装で、toroidal_mhd_test.py も同じソルバ関数を呼ぶ。
    """

    nx: int = 16
    ny: int = 16
    nz: int = 16
    lx: float = 2.0
    ly: float = 2.0
    lz: float = 2.0
    cfl: float = 0.2

    @property
    def spacing(self) -> tuple[float, float, float]:
        return self.lx / self.nx, self.ly / self.ny, self.lz / self.nz

    def make_initial_state(self) -> dict[str, Any]:
        return make_torus_ic_python(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            lx=self.lx,
            ly=self.ly,
            lz=self.lz,
        )

    def compute_dt(self, state: dict[str, Any]) -> float:
        dx, dy, dz = self.spacing
        return _cfl_dt(state, self.cfl, dx, dy, dz)

    def advance(self, state: dict[str, Any], dt: float) -> dict[str, Any]:
        dx, dy, dz = self.spacing
        return advance_ct_hlle_rk2(state, dt, dx, dy, dz)


class HybridController:
    """
    低頻度 MHD と高頻度 Advection の同期を担うコントローラ。

    低頻度 MHD が更新された瞬間に StateBuffer へ最新の velocity / magnetic field /
    density / pressure を書き込み、高頻度 Advection 層はそのスナップショットを
    次の MHD 更新まで読み続ける。
    """

    def __init__(
        self,
        solver: MHDSolverAdapter,
        *,
        mhd_update_interval: int = 50,
        state_buffer: StateBuffer | None = None,
    ) -> None:
        if mhd_update_interval <= 0:
            raise ValueError("mhd_update_interval must be a positive integer.")

        self.solver = solver
        self.mhd_update_interval = mhd_update_interval
        self.state_buffer = state_buffer or StateBuffer()

        self.controller_step = 0
        self.mhd_time = 0.0
        self.advection_time = 0.0
        self.last_mhd_dt = 0.0
        self._state: dict[str, Any] | None = None

    @property
    def state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("HybridController is not initialized.")
        return self._state

    def initialize(self) -> StateSnapshot:
        self._state = self.solver.make_initial_state()
        self.last_mhd_dt = self.solver.compute_dt(self._state)
        return self.state_buffer.publish_mhd_state(
            self._state,
            mhd_step=0,
            simulation_time=self.mhd_time,
            metadata={
                "mhd_update_interval": self.mhd_update_interval,
                "producer": "validation/verification/energy_conservation.py",
                "handoff": (
                    "HybridController publishes the latest velocity and magnetic field "
                    "to StateBuffer only when the low-rate MHD step finishes."
                ),
            },
        )

    def advance_one_step(self) -> dict[str, Any]:
        if self._state is None:
            self.initialize()

        self.controller_step += 1
        substep_dt = self.last_mhd_dt / float(self.mhd_update_interval)
        self.advection_time += substep_dt
        mhd_updated = False

        if self.controller_step % self.mhd_update_interval == 0:
            # 低頻度 MHD が更新されたら、その場で StateBuffer を上書きする。
            # Advection 層はここで公開された snapshot を次の更新まで再利用する。
            self._state = self.solver.advance(self.state, self.last_mhd_dt)
            self.mhd_time += self.last_mhd_dt
            self.state_buffer.publish_mhd_state(
                self._state,
                mhd_step=self.controller_step,
                simulation_time=self.mhd_time,
                metadata={
                    "mhd_update_interval": self.mhd_update_interval,
                    "producer": "validation/verification/energy_conservation.py",
                    "handoff": (
                        "Advection reads StateBuffer.velocity and "
                        "StateBuffer.magnetic_field until the next low-rate publish."
                    ),
                },
            )
            self.last_mhd_dt = self.solver.compute_dt(self._state)
            mhd_updated = True

        advection_inputs = self.state_buffer.get_advection_inputs(copy=True)
        advection_inputs["advection_time"] = self.advection_time
        advection_inputs["mhd_updated"] = mhd_updated
        advection_inputs["controller_step"] = self.controller_step
        return advection_inputs

    def run(self, total_steps: int) -> list[dict[str, Any]]:
        if total_steps < 0:
            raise ValueError("total_steps must be non-negative.")
        if self._state is None:
            self.initialize()
        return [self.advance_one_step() for _ in range(total_steps)]
