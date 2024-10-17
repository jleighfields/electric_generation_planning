# Author: Justin Fields
# Date: 2021-12-03

import streamlit as st
import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import db
import os

from LP import run_lp
import utils


sns.set_style("white")
sns.set_palette("colorblind")
st.set_page_config(layout="wide")

########################################################
# flush output to console during rerun for docker
########################################################

print('', flush=True)

########################################################
# set up db
########################################################

if not ('db' in st.session_state):
    print('creating database')
    st.session_state.db = db.ResultsDB()

########################################################
# set up inputs sidebar
########################################################

with st.sidebar:
    st.write('### Create run')
    run_name = st.text_input("Unique run name", "rename me")

    st.write('---')
    st.write("#### Objective to minimize")
    min_obj = st.radio("", ('minimize cost', 'minimize co2'))

    st.write('---')
    st.write('#### Generation capacity and peak load')
    peak_load = st.slider('Peak load MW', 800, 1500, 1000, 50)
    wind_mw = st.slider('Wind MW', 0, 5000, (225, 3000), 25)
    solar_mw = st.slider('Solar MW', 0, 5000, (150, 3000), 25)
    batt_mw = st.slider('Battery MW', 0, 5000, (0, 3000), 100)
    gas_mw = st.slider('Gas MW', 0, 1000, (0, 750), 50)
    restrict_gas = st.slider('Restrict gas generation (% of load)', 0.0, 50.0, 25.0, 0.5)
    # hard code some parameters to simplify input
    min_charge_level = 0.1  # min charge level of batteries
    init_ch_level = 0.5  # initial battery charge level
    batt_hours = 4  # battery duration in hours
    batt_eff = 0.85  # battery efficiency
    use_outside_energy = True

    st.write('---')
    st.write('#### Resource costs')
    wind_cost = st.slider('Wind cost ($/MWh)', 15, 40, 26)
    solar_cost = st.slider('Solar cost ($/MWh)', 15, 45, 34)
    batt_cost = st.slider('Battery capacity cost ($/KW-Mo)', 4, 12, 8)
    gas_mw_cost = st.slider('Gas capacity cost ($/KW-Mo)', 6, 20, 11)
    gas_fuel_cost = st.slider('Gas fuel cost ($/MMBTu)', 2, 20, 5)
    outside_energy_cost = st.slider('Emergency energy cost ($/MWh)', 1000, 50000, 20000, 500)
    life_cycle_co2 = st.checkbox('Use lifecycle carbon emissions', value=True)
    co2_cost = st.slider('CO2 cost ($/ton)', 0, 1000, 100, 50)

    st.write('---')
    st.write('#### Stress test parameters')
    re_outage_start_date = st.date_input('Renewable energy outage start date',
                                         datetime.date(2030, 7, 10),
                                         datetime.date(2030, 1, 1),
                                         datetime.date(2030, 12, 31))
    re_outage_days = st.slider('Length renewable energy outage in days', 0, 21, 3, 1)

    run_button = st.button('create run')

    if 'results' in st.session_state:
        st.write('---')
        st.write(f"## Showings results from run: {st.session_state.results['run_name']}")
        st.write("Existing runs with the same name will be over written")
        save_button = st.button('Save run')

        if save_button:
            print(f'saving run: {save_button}')
            st.session_state.db.add_run(st.session_state.results)
            st.session_state.db.zip_results()
            # rerun to update selectbox
            st.rerun()

    st.write('---')
    st.write('### Delete run')

    st.session_state.num_runs = len(st.session_state.db.get_runs())
    delete_run = st.selectbox('Select run to delete', st.session_state.db.get_runs())
    delete_button = st.button('delete run')
    delete_all_button = st.button('delete all runs')

    if delete_button:
        print(f'deleting run: {delete_run}')
        st.session_state.db.delete_run(delete_run)
        # update zip
        if len(st.session_state.db.get_runs()) > 0:
            st.session_state.db.zip_results()
        # rerun to update selectbox
        st.rerun()

    if delete_all_button:
        print(f'deleting all runs')
        st.session_state.db.clear_db()
        # remove zip
        if os.path.exists('results.zip'):
            os.remove('results.zip')
        # rerun to update selectbox
        st.rerun()

    if os.path.exists('results.zip') and (st.session_state.num_runs > 0):
        st.session_state.db.zip_results()
        st.write('---')
        st.write('### Download results')
        with open('results.zip', 'rb') as fp:
            download_results_button = st.download_button(
                label='Download results',
                data=fp,
                file_name='planning_results.zip',
                mime='application/zip')


#################################################
# amount of CO2 per MW and MWh
# CO2 values from CSU study
#################################################

gas_co2_ton_per_mwh = (411 + 854) / 2000
# assumed 30 year life
gas_co2_ton_per_mw = ((5981 + 1000 + 35566 + 8210 + 10165 + 1425) / (6 * 18) / 30)

wind_co2_ton_per_mwh = life_cycle_co2 * (0.2 / 2000)
# assumed 30 year life
wind_co2_ton_per_mw = life_cycle_co2 * ((754 + 10 - 241) / 30)

solar_co2_ton_per_mwh = life_cycle_co2 * (2.1 / 2000)
# assumed 30 year life
solar_co2_ton_per_mw = life_cycle_co2 * ((1202 + 250 - 46) / 30)

# battery C02 given in lbs
# assumed 15 year life
batt_co2_ton_per_mw = life_cycle_co2 * ((1940400 - 83481 + 4903) / 2000 / 15)

cc_gas_mwh = co2_cost * gas_co2_ton_per_mwh
cc_gas_mw = co2_cost * gas_co2_ton_per_mw

cc_wind_mwh = co2_cost * wind_co2_ton_per_mwh
cc_wind_mw = co2_cost * wind_co2_ton_per_mw

cc_solar_mwh = co2_cost * solar_co2_ton_per_mwh
cc_solar_mw = co2_cost * solar_co2_ton_per_mw

cc_batt_mw = co2_cost * batt_co2_ton_per_mw


#################################################
# calculate fixed and variable costs
#################################################

# capacity cost in $/kw-mo
gas_fixed_cost = cc_gas_mw + gas_mw_cost * 12 * 1000  # converted to $/MW-yr

heat_rate = 8883  # btu/kwh
vom = 7.16  # $/mwh
gas_variable_cost = cc_gas_mwh + gas_fuel_cost * heat_rate / 1000 + vom

batt_fixed_cost = cc_batt_mw + batt_cost * 12 * 1000  # converted to $/MW-yr

wind_kw_mo = 1.13
wind_fixed_cost = cc_wind_mw + wind_kw_mo * 12 * 1000  # converted to $/MW-yr
wind_variable_cost = cc_wind_mwh + wind_cost  # $/mwh

solar_kw_mo = 1.13
solar_fixed_cost = cc_solar_mw + solar_kw_mo * 12 * 1000  # converted to $/MW-yr
solar_variable_cost = cc_solar_mwh + solar_cost  # $/mwh


if run_button:
    st.session_state.life_cycle_co2 = life_cycle_co2

    inputs = {}
    inputs['peak_load'] = peak_load
    inputs['min_obj'] = min_obj
    inputs['max_batt_mw'] = batt_mw[1]
    inputs['min_batt_mw'] = batt_mw[0]
    inputs['max_gas_mw'] = gas_mw[1]
    inputs['min_gas_mw'] = gas_mw[0]
    inputs['max_wind_mw'] = wind_mw[1]
    inputs['min_wind_mw'] = wind_mw[0]
    inputs['max_solar_mw'] = solar_mw[1]
    inputs['min_solar_mw'] = solar_mw[0]
    inputs['restrict_gas'] = restrict_gas
    inputs['min_charge_level'] = min_charge_level
    inputs['init_ch_level'] = init_ch_level
    inputs['batt_hours'] = batt_hours
    inputs['batt_eff'] = batt_eff
    inputs['use_outside_energy'] = use_outside_energy
    inputs['outside_energy_cost'] = outside_energy_cost
    inputs['gas_mw_cost'] = gas_fixed_cost
    inputs['gas_mwh_cost'] = gas_variable_cost
    inputs['batt_mw_cost'] = batt_fixed_cost
    inputs['wind_mw_cost'] = wind_fixed_cost
    inputs['wind_mwh_cost'] = wind_variable_cost
    inputs['solar_mw_cost'] = solar_fixed_cost
    inputs['solar_mwh_cost'] = solar_variable_cost
    inputs['re_outage_start'] = re_outage_start_date
    inputs['re_outage_days'] = re_outage_days
    inputs['co2_cost'] = co2_cost
    inputs['gas_co2_ton_per_mwh'] = gas_co2_ton_per_mwh
    inputs['gas_co2_ton_per_mw'] = gas_co2_ton_per_mw
    inputs['wind_co2_ton_per_mwh'] = wind_co2_ton_per_mwh
    inputs['wind_co2_ton_per_mw'] = wind_co2_ton_per_mw
    inputs['solar_co2_ton_per_mwh'] = solar_co2_ton_per_mwh
    inputs['solar_co2_ton_per_mw'] = solar_co2_ton_per_mw
    inputs['batt_co2_ton_per_mw'] = batt_co2_ton_per_mw

    st.session_state.inputs = inputs
    st.session_state.run_name = run_name

    st.session_state.results = run_lp(
        run_name=st.session_state.run_name,  
        inputs_from_usr=st.session_state.inputs
        )

    # rerun to show save button
    st.rerun()


########################################################
# display results and plot
########################################################
# st.write('# Electricity generation planning model')
# st.write('## Introduction')
st.write('''
# Electricity generation planning model
## Introduction
This tool models the cost and carbon emissions for scenarios that optimize the amount of electricity generation resource
 (wind, solar, batteries, and gas) to serve the electricity use (load). 
The loads are based on the forecasted requirements of four northern Colorado communities. 
The calculations are based on simplified modeling sourced from detailed utility resource software tools. 
Users can vary the available capacity and costs by generation type, optimize for either cost or carbon emissions and 
visualize stress tests of limited renewable resource availability. 

## Instructions
The user can adjust input parameters in the left side panel. 
Adjustments can be made to the allowable range of capacity that can be installed, the cost of each resource type, 
limits on gas capacity and lifecycle carbon value. 
Default values represent reasonable starting points. 

Once the inputs are set, clicking the **create run** button will start the optimization. 
It will take approximately a minute to return results. 
There will be a running icon in the upper right hand corner to let you know the optimization is running. 

Results are displayed in the main panel. The **save run** button will be displayed in the left side panel after a run 
is created.
The results will be available for download after they have been saved. 
The download button will be displayed below the Delete Run section after the results have been saved. 
This will download all the inputs and results for each saved run.
The run can be deleted by using the delete button at the bottom of the left side panel. 

Results include values for how much of each generation type is needed (megawatts), costs (total, generation and carbon 
in millions of dollars), and generation information are displayed below (excess, renewable, gas, emergency and carbon 
emissions). 
A plot showing the hourly results for up to 21 days will be displayed below the run metrics. 
Below the plot the input parameters are shown so the inputs can be verified.
''')

if 'results' in st.session_state:
    st.write('---')
    st.write("### Resource capacities")
    cap_mw = st.session_state.results['cap_mw']
    r1col1, r1col2, r1col3, r1col4 = st.columns(4)
    r1col1.metric("Wind MW", int(cap_mw['wind_mw']))
    r1col2.metric("Solar MW", int(cap_mw['solar_mw']))
    r1col3.metric("Battery MW", int(cap_mw['batt_mw']))
    r1col4.metric("Gas MW", int(cap_mw['gas_mw']))

    st.write('---')
    st.write('### Cost metrics')
    # st.write(f"Objective value: {int(st.session_state.results['obj_val'])}")
    metrics = st.session_state.results['metrics']
    r2col1, r2col2, r2col3 = st.columns(3)
    r2col1.metric("Total cost (mill)", int(metrics['total_cost_mill']))
    r2col2.metric("Gen cost (mill)", int(metrics['total_gen_cost_mill']))
    r2col3.metric("CO2 cost (mill)", int(metrics['total_co2_cost_mill']))

    st.write('---')
    st.write('### Generation metrics')
    r3col1, r3col2, r3col3, r3col4, r3col5 = st.columns(5)
    r3col1.metric("% excess gen", int(metrics['excess_gen_percent']))
    r3col2.metric("% RE gen", np.round(metrics['re_percent'], 1))
    r3col3.metric("% Gas gen", np.round(metrics['gas_percent'], 1))
    r3col4.metric("Tons CO2 (thou)", int(metrics['total_co2_thou_tons']))
    r3col5.metric("Emergency MWh", int(metrics['total_outside_energy']))

    st.write('---')
    st.write('### Hourly load and generation plot')
    start_date = st.date_input(
        'Plot start date',
        datetime.date(2030, 7, 7),
        datetime.date(2030, 1, 1),
        datetime.date(2030, 12, 31)
    )
    num_days = st.slider('Number of days to plot', 1, 28, 14, 1)

    # create default plot ranges from user inputs
    plot_range_start_default = start_date.strftime("%Y-%m-%d %H:%M:%S")
    plot_range_end_default = (
            start_date + pd.Timedelta(f'{num_days}d')
    ).strftime("%Y-%m-%d %H:%M:%S")

    fig = utils.get_resource_stack_plot(
        st.session_state.results['final_df'],
        plot_range_start_default,
        plot_range_end_default
    )
    st.plotly_chart(fig)

    # utils.plot_hourly(st.session_state.results['final_df'], start_date, num_days)
    # st.pyplot(fig=plt)

    st.write('---')
    st.write('### Inputs')
    st.write(st.session_state.results['inputs'])
