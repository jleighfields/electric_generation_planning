"""Single home for the generation model's parameter values.

Holds the headless default input set (``get_base_inputs``) plus the emission
factors and carbon-inclusive $/MW and $/MWh cost formulas (``cost_inputs``).
``cost_inputs`` is shared by both the defaults here and the Shiny UI
(``app.build_inputs``): each caller supplies its own tunable knob values
(carbon price, per-resource costs, lifecycle toggle) and the formulas that
combine them live here, so the two paths cannot drift apart.
"""

import datetime

# Assumed asset life (years) for the per-MW construction/O&M/decommission
# emission factors. CO2 values are from the CSU study.
ASSET_LIFE_YEARS = 30
BATT_LIFE_YEARS = 15

# Gas plant operating constants.
HEAT_RATE = 8883  # btu/kwh
VOM = 7.16  # $/mwh

# Fixed per-capacity O&M for renewables ($/kW-mo).
WIND_KW_MO = 1.13
SOLAR_KW_MO = 1.13


def cost_inputs(
    co2_cost: float,
    life_cycle_co2: bool,
    wind_cost: float,
    solar_cost: float,
    batt_cost: float,
    gas_mw_cost: float,
    gas_fuel_cost: float,
) -> dict:
    """Compute emission factors and carbon-inclusive resource costs.

    Args:
        co2_cost: Carbon price ($/ton).
        life_cycle_co2: Include lifecycle (per-MW) CO2 for wind, solar, and
            battery. Gas CO2 is always counted.
        wind_cost: Wind energy cost ($/MWh).
        solar_cost: Solar energy cost ($/MWh).
        batt_cost: Battery capacity cost ($/kW-mo).
        gas_mw_cost: Gas capacity cost ($/kW-mo).
        gas_fuel_cost: Gas fuel cost ($/MMBTu).

    Returns:
        A dict of the ``co2_*`` emission factors, ``co2_cost``, and the fixed
        ($/MW-yr) and variable ($/MWh) cost inputs that run_lp expects.
    """
    # emission factors: tons CO2 per MWh generated and per MW installed
    gas_co2_ton_per_mwh = (411 + 854) / 2000
    gas_co2_ton_per_mw = (5981 + 1000 + 35566 + 8210 + 10165 + 1425) / (6 * 18) / ASSET_LIFE_YEARS

    wind_co2_ton_per_mwh = life_cycle_co2 * (0.2 / 2000)
    wind_co2_ton_per_mw = life_cycle_co2 * ((754 + 10 - 241) / ASSET_LIFE_YEARS)

    solar_co2_ton_per_mwh = life_cycle_co2 * (2.1 / 2000)
    solar_co2_ton_per_mw = life_cycle_co2 * ((1202 + 250 - 46) / ASSET_LIFE_YEARS)

    # battery CO2 is given in lbs
    batt_co2_ton_per_mw = life_cycle_co2 * ((1940400 - 83481 + 4903) / 2000 / BATT_LIFE_YEARS)

    # carbon cost per unit ($/MWh and $/MW), folded into each resource's cost
    cc_gas_mwh = co2_cost * gas_co2_ton_per_mwh
    cc_gas_mw = co2_cost * gas_co2_ton_per_mw
    cc_wind_mwh = co2_cost * wind_co2_ton_per_mwh
    cc_wind_mw = co2_cost * wind_co2_ton_per_mw
    cc_solar_mwh = co2_cost * solar_co2_ton_per_mwh
    cc_solar_mw = co2_cost * solar_co2_ton_per_mw
    cc_batt_mw = co2_cost * batt_co2_ton_per_mw

    return {
        "co2_cost": co2_cost,
        "gas_co2_ton_per_mwh": gas_co2_ton_per_mwh,
        "gas_co2_ton_per_mw": gas_co2_ton_per_mw,
        "wind_co2_ton_per_mwh": wind_co2_ton_per_mwh,
        "wind_co2_ton_per_mw": wind_co2_ton_per_mw,
        "solar_co2_ton_per_mwh": solar_co2_ton_per_mwh,
        "solar_co2_ton_per_mw": solar_co2_ton_per_mw,
        "batt_co2_ton_per_mw": batt_co2_ton_per_mw,
        # fixed costs converted to $/MW-yr, carbon included
        "gas_mw_cost": cc_gas_mw + gas_mw_cost * 12 * 1000,
        "gas_mwh_cost": cc_gas_mwh + gas_fuel_cost * HEAT_RATE / 1000 + VOM,
        "batt_mw_cost": cc_batt_mw + batt_cost * 12 * 1000,
        "wind_mw_cost": cc_wind_mw + WIND_KW_MO * 12 * 1000,
        "wind_mwh_cost": cc_wind_mwh + wind_cost,
        "solar_mw_cost": cc_solar_mw + SOLAR_KW_MO * 12 * 1000,
        "solar_mwh_cost": cc_solar_mwh + solar_cost,
    }


def get_base_inputs() -> dict:
    """Build the default input set for run_lp().

    Returns the full headless default parameter set (capacity bounds,
    battery parameters, emission factors, and resource costs). run_lp
    merges any caller-supplied overrides on top of these via
    ``inputs.update(...)``.

    Returns:
        A dict of default inputs for the linear program.
    """
    inputs = {}

    # restrictions from UI
    inputs["peak_load"] = 1000
    inputs["min_obj"] = "minimize cost"
    inputs["max_batt_mw"] = 3000
    inputs["min_batt_mw"] = 0
    inputs["max_gas_mw"] = 1000
    inputs["min_gas_mw"] = 0
    inputs["max_wind_mw"] = 2000
    inputs["min_wind_mw"] = 0
    inputs["max_solar_mw"] = 3000
    inputs["min_solar_mw"] = 0
    inputs["re_outage_start"] = datetime.date(2030, 7, 3)
    inputs["re_outage_days"] = 3

    # restrict gas generation to a percent of total load (0-100).
    # run_lp divides this by 100, so e.g. 20 -> gas may serve at most 20% of load,
    # and 100 -> effectively no limit on gas generation.
    inputs["restrict_gas"] = 20

    # outside energy used for solving
    inputs["use_outside_energy"] = True
    inputs["outside_energy_cost"] = 10000

    # battery parameters
    inputs["min_charge_level"] = 0.1
    inputs["init_ch_level"] = 0.5
    inputs["batt_hours"] = 4
    inputs["batt_eff"] = 0.85

    # emission factors and carbon-inclusive resource costs. The formulas are in
    # cost_inputs (shared with app.build_inputs); these are the headless default
    # knob values. Carbon 2030 $/ton reference was $9.06.
    inputs.update(
        cost_inputs(
            co2_cost=160,
            life_cycle_co2=True,
            wind_cost=41.01,  # $/mwh
            solar_cost=33.51,  # $/mwh
            batt_cost=8.25,  # $/kW-mo
            gas_mw_cost=11.27,  # $/kW-mo
            gas_fuel_cost=4.37,  # $/mmbtu
        )
    )

    return inputs
