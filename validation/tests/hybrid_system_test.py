#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.hybrid_controller import HybridController, ToroidalMHDSolverAdapter


def run_hybrid_system_smoke_test(
    *,
    total_steps: int = 6,
    mhd_update_interval: int = 3,
) -> dict[str, object]:
    solver = ToroidalMHDSolverAdapter(nx=8, ny=8, nz=8, cfl=0.2)
    controller = HybridController(
        solver,
        mhd_update_interval=mhd_update_interval,
    )

    initial_snapshot = controller.initialize()
    initial_vx = initial_snapshot.velocity["vx"].copy()

    history = controller.run(total_steps)
    source_steps = [entry["source_mhd_step"] for entry in history]
    expected_source_steps = [
        step - (step % mhd_update_interval) for step in range(1, total_steps + 1)
    ]

    assert source_steps == expected_source_steps
    assert [entry["mhd_updated"] for entry in history] == [
        (step % mhd_update_interval == 0) for step in range(1, total_steps + 1)
    ]

    latest_snapshot = controller.state_buffer.latest_snapshot
    assert latest_snapshot is not None
    assert latest_snapshot.mhd_step == expected_source_steps[-1]
    assert np.isfinite(latest_snapshot.density).all()
    assert np.isfinite(latest_snapshot.pressure).all()
    assert not np.allclose(latest_snapshot.velocity["vx"], initial_vx)

    return {
        "history": history,
        "latest_snapshot_step": latest_snapshot.mhd_step,
        "final_advection_time": history[-1]["advection_time"],
        "final_mhd_time": latest_snapshot.simulation_time,
    }


def test_hybrid_controller_low_rate_updates() -> None:
    result = run_hybrid_system_smoke_test()
    assert result["latest_snapshot_step"] == 6


def main() -> int:
    result = run_hybrid_system_smoke_test()
    print("Hybrid system smoke test")
    print("----------------------------------------")
    for entry in result["history"]:
        print(
            "step={step:02d} source_mhd_step={source:02d} "
            "mhd_updated={updated} advection_time={time:.6e}".format(
                step=entry["controller_step"],
                source=entry["source_mhd_step"],
                updated=entry["mhd_updated"],
                time=entry["advection_time"],
            )
        )
    print("----------------------------------------")
    print(
        "latest_snapshot_step={step} final_mhd_time={mhd_time:.6e} "
        "final_advection_time={adv_time:.6e}".format(
            step=result["latest_snapshot_step"],
            mhd_time=result["final_mhd_time"],
            adv_time=result["final_advection_time"],
        )
    )
    print("Hybrid system smoke test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
