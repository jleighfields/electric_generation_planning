"""Unit + integration tests for the LP model and results store.

Run fast tests only:   uv run pytest -m "not slow and not e2e"
Run everything:        uv run pytest
"""

import pandas as pd
import pytest

from src.db import ResultsDB, make_fake_results
from src.LP import run_lp
from src.parameters import get_base_inputs

# --------------------------------------------------------------------------- #
# fast unit tests
# --------------------------------------------------------------------------- #


def test_get_base_inputs_has_expected_keys():
    inputs = get_base_inputs()
    expected = {
        "peak_load",
        "min_obj",
        "max_batt_mw",
        "min_batt_mw",
        "max_gas_mw",
        "min_gas_mw",
        "max_wind_mw",
        "min_wind_mw",
        "max_solar_mw",
        "min_solar_mw",
        "restrict_gas",
        "min_charge_level",
        "init_ch_level",
        "batt_hours",
        "batt_eff",
        "use_outside_energy",
        "outside_energy_cost",
        "gas_mw_cost",
        "gas_mwh_cost",
        "batt_mw_cost",
        "wind_mw_cost",
        "wind_mwh_cost",
        "solar_mw_cost",
        "solar_mwh_cost",
        "re_outage_start",
        "re_outage_days",
        "co2_cost",
    }
    assert expected.issubset(inputs.keys())


def test_results_db_crud():
    db = ResultsDB()
    assert len(db.get_runs()) == 0

    db.add_run(make_fake_results("run_a"))
    db.add_run(make_fake_results("run_b"))
    assert set(db.get_runs()) == {"run_a", "run_b"}

    # adding a run with an existing name overwrites rather than duplicates
    db.add_run(make_fake_results("run_a"))
    assert sorted(db.get_runs()) == ["run_a", "run_b"]

    db.delete_run("run_a")
    assert list(db.get_runs()) == ["run_b"]

    db.clear_db()
    assert len(db.get_runs()) == 0


def test_zip_results_writes_archive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = ResultsDB()
    db.add_run(make_fake_results("run_a"))
    db.zip_results()
    assert (tmp_path / "results.zip").is_file()


# --------------------------------------------------------------------------- #
# integration test — full LP solve (~15s)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def solved():
    return run_lp(run_name="pytest")


@pytest.mark.slow
def test_run_lp_shape_and_metrics(solved):
    assert set(solved) == {"run_name", "inputs", "obj_val", "cap_mw", "metrics", "final_df"}
    assert isinstance(solved["final_df"], pd.DataFrame)
    assert solved["final_df"].shape[0] == 8760
    for key in ("total_cost_mill", "total_co2_cost_mill", "gas_percent", "re_percent"):
        assert key in solved["metrics"]


@pytest.mark.slow
def test_run_lp_serves_load(solved):
    # every hour must be served: net_load (generation - load) is non-negative
    assert solved["final_df"]["net_load"].min() >= -1e-3


@pytest.mark.slow
def test_run_lp_no_simultaneous_charge_and_discharge(solved):
    df = solved["final_df"]
    both = ((df["batt_charge"] > 1e-6) & (df["batt_discharge"] > 1e-6)).sum()
    assert int(both) == 0


@pytest.mark.slow
def test_run_lp_capacities_within_bounds(solved):
    inputs = solved["inputs"]
    cap = solved["cap_mw"]
    for res in ("batt", "gas", "wind", "solar"):
        lo, hi = inputs[f"min_{res}_mw"], inputs[f"max_{res}_mw"]
        assert lo - 1e-6 <= cap[f"{res}_mw"] <= hi + 1e-6


@pytest.mark.slow
def test_run_lp_respects_gas_restriction(solved):
    # default restrict_gas is 20% of load; gas generation must not exceed it
    assert solved["metrics"]["gas_percent"] <= 20.0 + 1e-3
