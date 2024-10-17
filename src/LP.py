# Author: Justin Fields
# Date: 2021-12-03

import pandas as pd
import numpy as np
import seaborn as sns
import time
from pprint import pprint
# import ortools
from ortools.linear_solver import pywraplp
import sys

import datetime
from joblib import dump, load

# set up logging
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_base_inputs() -> dict:
    """
    Creates a dictionary of default inputs for run_lp(). They will
    be get updated if any user selected inputs are passed to run_lp().
    Args:
        None
    Returns:
        dict - contains default inputs for linear program
    """
    inputs = {}

    # restrictions from UI
    inputs['peak_load'] = 1000
    inputs['min_obj'] = 'minimize cost'
    inputs['max_batt_mw'] = 3000
    inputs['min_batt_mw'] = 0
    inputs['max_gas_mw'] = 1000
    inputs['min_gas_mw'] = 0
    inputs['max_wind_mw'] = 2000
    inputs['min_wind_mw'] = 0
    inputs['max_solar_mw'] = 3000
    inputs['min_solar_mw'] = 0
    inputs['re_outage_start'] = datetime.date(2030, 7, 3)
    inputs['re_outage_days'] = 3

    # restrict gas to a portion of total load, 0-1 or None
    # e.g. 0.05 -> 5% limit on gas generation
    # and  1.0 -> no limit on gas generation
    inputs['restrict_gas'] = 20

    # outside energy used for solving
    inputs['use_outside_energy'] = True
    inputs['outside_energy_cost'] = 10000

    # battery parameters
    inputs['min_charge_level'] = 0.1
    inputs['init_ch_level'] = 0.5
    inputs['batt_hours'] = 4
    inputs['batt_eff'] = 0.85

    # cost of CO2
    # C02 values from CSU study
    gas_co2_ton_per_mwh = (411 + 854) / 2000
    # assumed 20 year life
    gas_co2_ton_per_mw = (5981 + 1000 + 35566 + 8210 + 10165 + 1425) / (6 * 18) / 20

    wind_co2_ton_per_mwh = 0.2 / 2000
    # assumed 20 year life
    wind_co2_ton_per_mw = (754 + 10 - 241) / 20

    solar_co2_ton_per_mwh = 2.1 / 2000
    # assumed 20 year life
    solar_co2_ton_per_mw = (1202 + 250 - 46) / 20

    # battery C02 given in lbs
    # assumed 15 year life
    batt_co2_ton_per_mw = (1940400 - 83481 + 4903) / 2000 / 15

    # Carbon 2030 $/ton = $9.06
    # co2_cost = 9.06
    co2_cost = 160

    # calculate variable costs in $/MWh and $/MW
    # calculate fixed costs in $/MW
    # these costs include the cost of carbon
    cc_gas_mwh = co2_cost * gas_co2_ton_per_mwh
    cc_gas_mw = co2_cost * gas_co2_ton_per_mw

    cc_wind_mwh = co2_cost * wind_co2_ton_per_mwh
    cc_wind_mw = co2_cost * wind_co2_ton_per_mw

    cc_solar_mwh = co2_cost * solar_co2_ton_per_mwh
    cc_solar_mw = co2_cost * solar_co2_ton_per_mw

    cc_batt_mw = co2_cost * batt_co2_ton_per_mw

    # save cost of carbon calcs for
    # fixed and variable charges
    inputs['co2_cost'] = co2_cost
    inputs['gas_co2_ton_per_mwh'] = gas_co2_ton_per_mwh
    inputs['gas_co2_ton_per_mw'] = gas_co2_ton_per_mw
    inputs['wind_co2_ton_per_mwh'] = wind_co2_ton_per_mwh
    inputs['wind_co2_ton_per_mw'] = wind_co2_ton_per_mw
    inputs['solar_co2_ton_per_mwh'] = solar_co2_ton_per_mwh
    inputs['solar_co2_ton_per_mw'] = solar_co2_ton_per_mw
    inputs['batt_co2_ton_per_mw'] = batt_co2_ton_per_mw

    # capacity cost in $/kw-mo
    gas_cap_cost = 11.27
    inputs['gas_mw_cost'] = cc_gas_mw + gas_cap_cost * 12 * 1000  # converted to $/MW-yr

    heat_rate = 8883  # btu/kwh
    vom = 7.16  # $/mwh
    gas_fuel_cost = 4.37  # $/mmbtu
    inputs['gas_mwh_cost'] = cc_gas_mwh + gas_fuel_cost * heat_rate / 1000 + vom

    batt_cost = 8.25
    inputs['batt_mw_cost'] = cc_batt_mw + batt_cost * 12 * 1000  # converted to $/MW-yr

    wind_kw_mo = 1.13
    inputs['wind_mw_cost'] = cc_wind_mw + wind_kw_mo * 12 * 1000  # converted to $/MW-yr
    inputs['wind_mwh_cost'] = cc_wind_mwh + 41.01  # $/mwh

    solar_kw_mo = 1.13
    inputs['solar_mw_cost'] = cc_solar_mw + solar_kw_mo * 12 * 1000  # converted to $/MW-yr
    inputs['solar_mwh_cost'] = cc_solar_mwh + 33.51  # $/mwh

    return inputs


def run_lp(
        run_name: str, 
        inputs_from_usr: dict = {},
        ) -> dict:
    """Run a linear program that minimizes costs with the constraints:
        1. load must be served
        2. battery state of charge limits
        3. RE and hydro fixed profiles
    This optimization does not consider an outside market, it only minimizes
    costs to serve native load with native resources.

    Args:
        run_name: str - a unique run name for given combination of inputs
        inputs_from_usr: dict - a dictionary of keyword/value pairs to pass
        the linear program.

    Returns: dict like:
    {
        'run_name': run_name, - str: unique name given by user
        'inputs': inputs, - dict: inputs used in optimization
        'obj_val': obj_val, - float: final objective returned by solver
        'cap_mw': cap_mw, - dict: sizes of installed generation resources
        'metrics': metrics, - dict: solved metrics - like total co2
        'final_df': final_df - pd.DataFrame - final hourly solved values, used in plot
    }

    Keyword arguments that can updated by user:
        peak_load: float -- peak load used to scale load profile
        min_obj: str -- objective to minimize, 'minimize cost' or 'minimize co2'
        max_batt_mw: float -- max battery capacity to install
        min_batt_mw: float -- min battery capacity to install
        max_gas_mw: float -- max gas capacity to install
        min_gas_mw: float -- min gas capacity to install
        max_wind_mw: float -- max wind capacity to install
        min_wind_mw: float -- min wind capacity to install
        max_solar_mw: float -- max solar capacity to install
        min_solar_mw: float -- min solar capacity to install
        restrict_gas: float -- the maximum amount of gas generation as a percent of load
        min_charge_level: float -- minimum charge level of batteries
        init_ch_level: float -- initial charge level of batteries
        batt_hours: float -- duration of hours for batteries
        batt_eff: float -- efficiency of batteries, like 0.85 for 85% efficiency
        use_outside_energy: bool -- use outside energy to meet load
        outside_energy_cost: float -- cost of outside energy $/MWh
        gas_mw_cost: float -- gas fixed cost $/MW including carbon costs
        gas_mwh_cost: float -- gas variable cost $/MWh including carbon costs
        batt_mw_cost: float -- battery fixed cost $/MW including carbon costs
        wind_mw_cost: float -- wind fixed costs $/MW including carbon costs
        wind_mwh_cost: float -- wind variable costs $/MWh including carbon costs
        solar_mw_cost: float -- solar fixed costs $/MW including carbon costs
        solar_mwh_cost: float -- solar variable costs $/MWh including carbon costs
        re_outage_start: float -- start date for RE outage stress test
        re_outage_days: int -- number of days for RE outage for stress testing
        co2_cost: float -- cost of carbon emissions $/ton
        gas_co2_ton_per_mwh: float -- emissions from energy generation
        gas_co2_ton_per_mw: float -- emissions associated with construction, O&M, decommission
        wind_co2_ton_per_mwh: float -- emissions from energy generation
        wind_co2_ton_per_mw: float-- emissions associated with construction, O&M, decommission
        solar_co2_ton_per_mwh: float -- emissions from energy generation
        solar_co2_ton_per_mw: float -- emissions associated with construction, O&M, decommission
        batt_co2_ton_per_mw: float-- emissions associated with construction, O&M, decommission
    """

    ########################################################
    # set inputs for optimization
    ########################################################

    # get base inputs - includes all expected values for LP
    inputs = get_base_inputs()
    # update base inputs with usr inputs
    inputs.update(inputs_from_usr)
    log.info(f'inputs:')
    pprint(inputs)
    sys.stdout.flush()

    # unpack inputs
    peak_load = inputs['peak_load']
    min_obj = inputs['min_obj']
    max_batt_mw = inputs['max_batt_mw']
    min_batt_mw = inputs['min_batt_mw']
    max_gas_mw = inputs['max_gas_mw']
    min_gas_mw = inputs['min_gas_mw']
    max_wind_mw = inputs['max_wind_mw']
    min_wind_mw = inputs['min_wind_mw']
    max_solar_mw = inputs['max_solar_mw']
    min_solar_mw = inputs['min_solar_mw']
    restrict_gas = inputs['restrict_gas']
    min_charge_level = inputs['min_charge_level']
    init_ch_level = inputs['init_ch_level']
    batt_hours = inputs['batt_hours']
    batt_eff = inputs['batt_eff']
    use_outside_energy = inputs['use_outside_energy']
    outside_energy_cost = inputs['outside_energy_cost']
    gas_mw_cost = inputs['gas_mw_cost']
    gas_mwh_cost = inputs['gas_mwh_cost']
    batt_mw_cost = inputs['batt_mw_cost']
    wind_mw_cost = inputs['wind_mw_cost']
    wind_mwh_cost = inputs['wind_mwh_cost']
    solar_mw_cost = inputs['solar_mw_cost']
    solar_mwh_cost = inputs['solar_mwh_cost']
    re_outage_start = inputs['re_outage_start']
    re_outage_days = inputs['re_outage_days']
    co2_cost = inputs['co2_cost']
    gas_co2_ton_per_mwh = inputs['gas_co2_ton_per_mwh']
    gas_co2_ton_per_mw = inputs['gas_co2_ton_per_mw']
    wind_co2_ton_per_mwh = inputs['wind_co2_ton_per_mwh']
    wind_co2_ton_per_mw = inputs['wind_co2_ton_per_mw']
    solar_co2_ton_per_mwh = inputs['solar_co2_ton_per_mwh']
    solar_co2_ton_per_mw = inputs['solar_co2_ton_per_mw']
    batt_co2_ton_per_mw = inputs['batt_co2_ton_per_mw']

    # restrict gas to a portion of total load, 0-1 or None
    # e.g. 0.05 -> 5% limit on gas generation
    # and  1.0 -> no limit on gas generation
    # divide by 100 since input is in percentages
    restrict_gas = restrict_gas / 100

    # read profile data for load and re gen
    df = pd.read_csv('src/profiles.csv', index_col='Hour')
    df = df[df.index < 8760]
    df['2030_load'] = df.load * peak_load
    # update time to 2030
    df['mst'] = pd.date_range('2030-01-01', periods=8760, freq="h")

    # get RE outage times index
    outage_hours = pd.date_range(re_outage_start, periods=24 * re_outage_days, freq="h")
    outage_idx = df['mst'].isin(outage_hours)

    # apply RE outages
    df.loc[outage_idx, ['solar', 'wind']] = 0
    log.info(df.loc[outage_idx, ['mst', 'solar', 'wind']])

    ########################################################
    # Build optimization model
    # create decision variables and constraints
    ########################################################

    # start timer
    total_time_0 = time.time()

    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver('simple_lp_program', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # build capacity decision variables for the resources
    batt = solver.NumVar(min_batt_mw, max_batt_mw, 'batt')
    solar = solver.NumVar(min_solar_mw, max_solar_mw, 'solar')
    wind = solver.NumVar(min_wind_mw, max_wind_mw, 'wind')
    gas = solver.NumVar(min_gas_mw, max_gas_mw, 'gas')

    log.info('Adding variables for build capacity')
    log.info(f'Number of variables = {solver.NumVariables()}')

    # generation decision variables
    log.info('Adding hourly variables and constraints')
    t0 = time.time()

    # create arrays to hold hourly variables
    batt_ch = [None] * len(df.index)
    batt_disch = [None] * len(df.index)
    wind_gen = [None] * len(df.index)
    solar_gen = [None] * len(df.index)
    gas_gen = [None] * len(df.index)
    hydro_gen = [None] * len(df.index)
    SOC = [None] * len(df.index)
    if use_outside_energy:
        outside_energy = [None] * len(df.index)

    for h in df.index:
        # add hourly decision variables
        batt_ch[h] = solver.NumVar(0, solver.infinity(), 'batt_ch[{}]'.format(h))
        batt_disch[h] = solver.NumVar(0, solver.infinity(), 'batt_disch[{}]'.format(h))
        wind_gen[h] = solver.NumVar(0, solver.infinity(), 'wind_gen[{}]'.format(h))
        solar_gen[h] = solver.NumVar(0, solver.infinity(), 'solar_gen[{}]'.format(h))
        gas_gen[h] = solver.NumVar(0, solver.infinity(), 'gas_gen[{}]'.format(h))
        hydro_gen[h] = solver.NumVar(0, solver.infinity(), 'hydro_gen[{}]'.format(h))
        SOC[h] = solver.NumVar(0, solver.infinity(), 'SOC[{}]'.format(h))
        if use_outside_energy:
            outside_energy[h] = solver.NumVar(0, solver.infinity(), 'outside_energy[{}]'.format(h))

        # add hourly constraints

        # set SOC[h] equal to previous hour SOC
        # plus the change from charging or discharging
        if h == 0:
            solver.Add(SOC[h] <= init_ch_level * batt_hours * batt + batt_ch[h] - batt_disch[h] / batt_eff)
            solver.Add(SOC[h] >= init_ch_level * batt_hours * batt + batt_ch[h] - batt_disch[h] / batt_eff)
        else:
            solver.Add(SOC[h] <= SOC[h - 1] + batt_ch[h] - batt_disch[h] / batt_eff)
            solver.Add(SOC[h] >= SOC[h - 1] + batt_ch[h] - batt_disch[h] / batt_eff)

        # fix hourly hydro profile
        solver.Add(hydro_gen[h] <= df.loc[h, 'hydro'])
        solver.Add(hydro_gen[h] >= df.loc[h, 'hydro'])

        # SOC mwh constraints
        # max mwh constraint
        solver.Add(SOC[h] <= batt_hours * batt)
        # min mwh constraint
        solver.Add(SOC[h] >= min_charge_level * batt_hours * batt)

        # fix hourly RE gen profiles
        solver.Add(solar_gen[h] >= df.solar[h] * solar)
        solver.Add(solar_gen[h] <= df.solar[h] * solar)

        solver.Add(wind_gen[h] >= df.wind[h] * wind)
        solver.Add(wind_gen[h] <= df.wind[h] * wind)

        # hourly demand constraints
        # must be able to serve load
        if use_outside_energy:
            solver.Add(hydro_gen[h] +
                       solar_gen[h] + wind_gen[h] +
                       gas_gen[h] +
                       batt_disch[h] - batt_ch[h] +
                       outside_energy[h]
                       >= df['2030_load'][h])

            # only import 20% of demand
            solver.Add(outside_energy[h] <= 0.2 * df['2030_load'][h])

        else:
            solver.Add(hydro_gen[h] +
                       solar_gen[h] + wind_gen[h] +
                       gas_gen[h] +
                       batt_disch[h] - batt_ch[h]
                       >= df['2030_load'][h])

        # hourly generation constraints base on installed capacity
        solver.Add(batt_ch[h] <= batt)
        solver.Add(batt_disch[h] <= batt)
        solver.Add(gas_gen[h] <= gas)

    # total gas gen constraint
    if restrict_gas is not None:
        solver.Add(solver.Sum(gas_gen) <= restrict_gas * sum(df['2030_load']))

    if use_outside_energy:
        # no more than 5% total imports
        solver.Add(solver.Sum(outside_energy) <= float(0.05 * df['2030_load'].sum()))

    t1 = time.time()

    log.info('time to build model (seconds): {0:,.2f}\n'.format((t1 - t0), 1))
    log.info('Number of variables: {0:,}'.format(solver.NumVariables()))
    log.info('Number of constraints: {0:,}'.format(solver.NumConstraints()))
    print('\n', flush=True)

    ########################################################
    # Build objective function
    ########################################################

    objective = solver.Objective()

    if min_obj == 'minimize cost':
        # set the coefficients in the objective function for the capacity variables
        objective.SetCoefficient(batt, batt_mw_cost)
        objective.SetCoefficient(solar, solar_mw_cost)
        objective.SetCoefficient(wind, wind_mw_cost)
        objective.SetCoefficient(gas, gas_mw_cost)

        # add energy costs
        for h in df.index:
            objective.SetCoefficient(gas_gen[h], gas_mwh_cost)
            objective.SetCoefficient(wind_gen[h], wind_mwh_cost)
            objective.SetCoefficient(solar_gen[h], solar_mwh_cost)

            if use_outside_energy:
                objective.SetCoefficient(outside_energy[h], outside_energy_cost)

    else:
        # set the co2 per mw coefficients
        objective.SetCoefficient(batt, batt_co2_ton_per_mw)
        objective.SetCoefficient(solar, solar_co2_ton_per_mw)
        objective.SetCoefficient(wind, wind_co2_ton_per_mw)
        objective.SetCoefficient(gas, gas_co2_ton_per_mw)

        for h in df.index:
            objective.SetCoefficient(gas_gen[h], gas_co2_ton_per_mwh)
            objective.SetCoefficient(solar_gen[h], solar_co2_ton_per_mwh)
            objective.SetCoefficient(wind_gen[h], wind_co2_ton_per_mwh)
            # assume outside energy is worse than gas
            if use_outside_energy:
                objective.SetCoefficient(outside_energy[h], 2 * gas_co2_ton_per_mwh)

    for h in df.index:
        # dis-incentivize charging and discharging at the same time
        # this removes hours that both charge and discharge
        objective.SetCoefficient(batt_disch[h], 0.0000001)

        # benefit to keeping the batteries charged
        objective.SetCoefficient(SOC[h], -0.0000001)

    # minimize the cost to serve the system
    objective.SetMinimization()

    ########################################################
    # solve the system
    ########################################################

    log.info('Starting optimization...')
    t0 = time.time()
    status = solver.Solve()
    t1 = time.time()

    log.info('time to solve (minutes): {0:,.2f}\n'.format((t1 - t0) / 60, 1))

    log.info(f'Solution is optimal: {status == solver.OPTIMAL}')

    obj_val = objective.Value()
    log.info('Solution:')
    log.info('Objective value = {0:,.0f}\n'.format(obj_val))

    log.info('Build variables:')
    batt_mw = batt.solution_value()
    solar_mw = solar.solution_value()
    wind_mw = wind.solution_value()
    gas_mw = gas.solution_value()
    cap_mw = {'batt_mw': batt_mw,
              'solar_mw': solar_mw,
              'wind_mw': wind_mw,
              'gas_mw': gas_mw}

    # print results for build variables
    for r in [batt, solar, wind, gas]:
        log.info('{0} \t= {1:,.0f}'.format(r, r.solution_value()) + '\n')

    print('\n')

    ########################################################
    # get the solved values to return to user
    ########################################################

    # create a new data frame to hold the final solution values
    log.info('Gathering hourly data...\n')
    final_df = df[['Date', '2030_load', 'solar', 'wind']].copy()
    final_df['solar'] = final_df['solar'] * solar.solution_value()
    final_df['wind'] = final_df['wind'] * wind.solution_value()
    final_df['gas'] = 0
    final_df['batt_charge'] = 0
    final_df['batt_discharge'] = 0
    final_df['SOC'] = 0
    final_df['crsp'] = 0
    final_df['lap'] = 0
    if use_outside_energy:
        final_df['outside_energy'] = 0

    # get the battery charge and discharge by hour
    batt_ch_hourly = [None] * len(df.index)
    batt_disch_hourly = [None] * len(df.index)
    for h in df.index:
        batt_ch_hourly[h] = batt_ch[h].solution_value()
        batt_disch_hourly[h] = batt_disch[h].solution_value()

    # get cumulative sums for calculating SOC
    batt_ch_hourly = np.cumsum(batt_ch_hourly)
    batt_disch_hourly = np.cumsum(batt_disch_hourly)

    # get the hourly data
    final_df['gas'] = [gas_gen[h].solution_value() for h in range(df.shape[0])]
    final_df['batt_charge'] = [batt_ch[h].solution_value() for h in range(df.shape[0])]
    final_df['batt_discharge'] = [batt_disch[h].solution_value() for h in range(df.shape[0])]
    final_df['SOC'] = [SOC[h].solution_value() for h in range(df.shape[0])]
    final_df['hydro'] = [hydro_gen[h].solution_value() for h in range(df.shape[0])]

    if use_outside_energy:
        final_df['outside_energy'] = [outside_energy[h].solution_value() for h in range(df.shape[0])]

    # calc net load for a check on the results
    if use_outside_energy:
        final_df['net_load'] = round((final_df['hydro'] +
                                      final_df['solar'] +
                                      final_df['wind'] +
                                      final_df['gas'] +
                                      final_df['batt_discharge'] -
                                      final_df['batt_charge'] +
                                      final_df['outside_energy'] -
                                      final_df['2030_load']), 2)

    else:
        final_df['net_load'] = round((final_df['hydro'] +
                                      final_df['solar'] +
                                      final_df['wind'] +
                                      final_df['gas'] +
                                      final_df['batt_discharge'] -
                                      final_df['batt_charge'] -
                                      final_df['2030_load']), 2)

    final_df['load_and_charge'] = round((final_df['batt_charge'] +
                                         final_df['2030_load']), 2)

    # set the index to hours in 2030
    final_df.set_index(
        pd.date_range(start='2030-01-01 01:00:00', periods=final_df.shape[0], freq='h'),
        inplace=True
    )

    # summarize the data
    # log.info('Summary of hourly data:\n')
    # log.info(final_df.describe().T)
    # print('\n')

    # this should be empty...
    # log.info('Any negative net load? Should be empty...')
    # log.info(final_df[(final_df.net_load < 0)].T)
    # print('\n')

    # this should be empty...
    # log.info('Any hours with both charging and discharging? Should be empty...')
    # log.info(final_df[(final_df.batt_discharge > 0) & (final_df.batt_charge > 0)].T)
    # print('\n')

    ########################################################
    # calculate metrics to return to the user
    ########################################################

    metrics = {}

    if use_outside_energy:
        outside_energy_percent = 100 * final_df.outside_energy.sum() / final_df['2030_load'].sum()
        log.info('Outside energy as a percentage of load: {0:,.3f}%\n'.format(outside_energy_percent))

        total_outside_energy = 100 * final_df.outside_energy.sum()
        log.info('Total outside energy: {0:,.2f} MWh\n'.format(total_outside_energy))

        metrics['outside_energy_percent'] = outside_energy_percent
        metrics['total_outside_energy'] = total_outside_energy
    else:
        outside_energy_percent = 0
        log.info('Outside energy as a percentage of load: {0:,.3f}%\n'.format(outside_energy_percent))

        total_outside_energy = 0
        log.info('Total outside energy: {0:,.2f} MWh\n'.format(total_outside_energy))

        metrics['outside_energy_percent'] = outside_energy_percent
        metrics['total_outside_energy'] = total_outside_energy

    gas_percent = 100 * final_df.gas.sum() / final_df['2030_load'].sum()
    log.info('Gas generation as a percentage of load: {0:,.2f}%\n'.format(gas_percent))
    metrics['gas_percent'] = gas_percent

    re_percent = 100 * ((final_df.solar.sum() + final_df.wind.sum()) / final_df['2030_load'].sum())
    log.info('RE generation as a percentage of load: {0:,.2f}%\n'.format(re_percent))
    metrics['re_percent'] = re_percent

    excess_gen_percent = 100 * (final_df.net_load.sum() / final_df['2030_load'].sum())
    log.info('Excess generation as a percentage of load: {0:,.2f}%\n'.format(excess_gen_percent))
    metrics['excess_gen_percent'] = excess_gen_percent

    batt_efficiency = 100 * final_df.batt_discharge.sum() / final_df.batt_charge.sum()
    log.info('Batt discharge as a percentage of batt charge: {0:,.2f}%\n'.format(batt_efficiency))
    metrics['batt_efficiency'] = batt_efficiency

    # calculate total co2 generation
    gas_gen = final_df.gas.sum()
    wind_gen = final_df.wind.sum()
    solar_gen = final_df.solar.sum()

    total_co2 = (
            gas_co2_ton_per_mw * cap_mw['gas_mw'] + gas_co2_ton_per_mwh * gas_gen +
            wind_co2_ton_per_mw * cap_mw['wind_mw'] + wind_co2_ton_per_mwh * wind_gen +
            solar_co2_ton_per_mw * cap_mw['solar_mw'] + solar_co2_ton_per_mwh * solar_gen +
            batt_co2_ton_per_mw * cap_mw['batt_mw']
    )

    total_co2_cost = total_co2 * co2_cost
    metrics['total_co2_thou_tons'] = total_co2 / 1000
    metrics['total_co2_cost_mill'] = total_co2_cost / 1000000

    total_cost = (
            gas_mw_cost * cap_mw['gas_mw'] + gas_mwh_cost * gas_gen +
            wind_mw_cost * cap_mw['wind_mw'] + wind_mwh_cost * wind_gen +
            solar_mw_cost * cap_mw['solar_mw'] + solar_mwh_cost * solar_gen +
            batt_mw_cost * cap_mw['batt_mw']
    )

    metrics['total_cost_mill'] = total_cost / 1000000
    metrics['total_gen_cost_mill'] = metrics['total_cost_mill'] - metrics['total_co2_cost_mill']

    total_time_1 = time.time()
    log.info('total time to build, solve, and verify (minutes): {0:,.2f}\n'.format((total_time_1 - total_time_0) / 60))
    print('\n', flush=True)
    
    # return dictionary for displaying results
    return {'run_name': run_name,
            'inputs': inputs,
            'obj_val': obj_val,
            'cap_mw': cap_mw,
            'metrics': metrics,
            'final_df': final_df}


# for testing
if __name__ == '__main__':
    print('\n')
        
    results = run_lp(run_name='test')
    log.info(results)

    log.info('saving results')
    dump(results, 'results.joblib')

    # create 2nd run for testing db.py
    results2 = load('results.joblib')
    results['run_name'] = 'test2'
    dump(results, 'results2.joblib')

    log.info('Finished')
