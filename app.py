# Author: Justin Fields
"""Shiny for Python UI for the electricity generation planning model.

Defines the sidebar inputs, launches the LP solve as a non-blocking
background task, and renders the result metrics, hourly plot, and
save/delete/download controls. The model itself lives in ``src`` (LP.py,
db.py, parameters.py, utils.py); this module is UI only.
"""

import asyncio
import os

import pandas as pd
import shinyswatch
from shiny import App, reactive, render, req, ui
from shinywidgets import output_widget, render_widget

from src import parameters
from src.db import RESULTS_ZIP, ResultsDB
from src.LP import run_lp
from src.utils import get_resource_stack_plot

INTRO = """
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
It will take approximately a minute to return results. The optimization runs in the background, so the interface stays
responsive while it solves.

Results are displayed in the main panel. The **Save run** button appears in the left side panel after a run is created.
Saved runs can be downloaded (all inputs and results per run) using the download button below the Delete run section,
and removed with the delete buttons.
"""


def metric_section(title: str, boxes: list, col_widths) -> ui.TagList:
    """Wrap a row of value boxes in the standard results-section shell.

    Args:
        title: Section heading shown above the boxes.
        boxes: The ui.value_box elements to lay out in one row.
        col_widths: Bootstrap column widths passed to ui.layout_columns.

    Returns:
        A TagList with a divider, heading, and the value-box row.
    """
    return ui.TagList(
        ui.hr(),
        ui.h4(title),
        ui.layout_columns(*boxes, col_widths=col_widths),
    )


# ------------------------------------------------------------------ #
# UI
# ------------------------------------------------------------------ #

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Create run"),
        ui.input_text("run_name", "Unique run name", "rename me"),
        ui.hr(),
        ui.input_radio_buttons(
            "min_obj", "Objective to minimize", ["minimize cost", "minimize co2"]
        ),
        ui.hr(),
        ui.h6("Generation capacity and peak load"),
        ui.input_slider("peak_load", "Peak load MW", 800, 1500, 1000, step=50),
        ui.input_slider("wind_mw", "Wind MW", 0, 5000, [225, 3000], step=25),
        ui.input_slider("solar_mw", "Solar MW", 0, 5000, [150, 3000], step=25),
        ui.input_slider("batt_mw", "Battery MW", 0, 5000, [0, 3000], step=100),
        ui.input_slider("gas_mw", "Gas MW", 0, 1000, [0, 750], step=50),
        ui.input_slider(
            "restrict_gas", "Restrict gas generation (% of load)", 0.0, 50.0, 25.0, step=0.5
        ),
        ui.hr(),
        ui.h6("Resource costs"),
        ui.input_slider("wind_cost", "Wind cost ($/MWh)", 15, 40, 26),
        ui.input_slider("solar_cost", "Solar cost ($/MWh)", 15, 45, 34),
        ui.input_slider("batt_cost", "Battery capacity cost ($/KW-Mo)", 4, 12, 8),
        ui.input_slider("gas_mw_cost", "Gas capacity cost ($/KW-Mo)", 6, 20, 11),
        ui.input_slider("gas_fuel_cost", "Gas fuel cost ($/MMBTu)", 2, 20, 5),
        ui.input_slider(
            "outside_energy_cost", "Emergency energy cost ($/MWh)", 1000, 50000, 20000, step=500
        ),
        ui.input_checkbox("life_cycle_co2", "Use lifecycle carbon emissions", True),
        ui.input_slider("co2_cost", "CO2 cost ($/ton)", 0, 1000, 100, step=50),
        ui.hr(),
        ui.h6("Stress test parameters"),
        ui.input_date(
            "re_outage_start",
            "Renewable energy outage start date",
            value="2030-07-10",
            min="2030-01-01",
            max="2030-12-31",
        ),
        ui.input_slider(
            "re_outage_days", "Length renewable energy outage in days", 0, 21, 3, step=1
        ),
        ui.input_action_button("run_button", "create run", class_="btn-primary"),
        ui.output_ui("save_section"),
        ui.hr(),
        ui.h4("Delete run"),
        ui.output_ui("delete_section"),
        ui.output_ui("download_section"),
        width=350,
    ),
    ui.busy_indicators.use(spinners=True, pulse=True),
    ui.markdown(INTRO),
    ui.output_ui("capacity_boxes"),
    ui.output_ui("cost_boxes"),
    ui.output_ui("generation_boxes"),
    ui.output_ui("plot_section"),
    ui.output_ui("inputs_section"),
    title="Electricity generation planning",
    theme=shinyswatch.theme.flatly,
    fillable=False,
)


# ------------------------------------------------------------------ #
# Server
# ------------------------------------------------------------------ #


def server(input, output, session):
    """Wire up the app's reactive state, background solve, and renderers.

    Creates a per-session results store and reactive state, launches the LP
    solve as a non-blocking background task when create-run is clicked, and
    renders the metric boxes, hourly plot, and save/delete/download controls
    from the solved results.
    """
    # per-session in-memory results store (replaces st.session_state.db)
    db = ResultsDB()

    # holds the latest solved results dict (replaces st.session_state.results)
    results = reactive.value(None)
    # bumped after any save/delete so the delete select + download refresh
    runs_version = reactive.value(0)

    # -------------------------------------------------------------- #
    # assemble the inputs dict for run_lp from the sidebar controls
    # -------------------------------------------------------------- #
    def build_inputs() -> dict:
        """Assemble the run_lp inputs dict from the sidebar controls.

        Reads the current sidebar values and returns the keyword dict run_lp
        merges over its headless defaults. The emission factors and
        carbon-inclusive costs come from parameters.cost_inputs (the shared
        formulas, also used by parameters.get_base_inputs).

        Returns:
            The inputs dict passed to run_lp (capacity bounds, resource
            costs, emission factors, and stress-test settings).
        """
        wind_range = input.wind_mw()
        solar_range = input.solar_mw()
        batt_range = input.batt_mw()
        gas_range = input.gas_mw()

        inputs = {
            "peak_load": input.peak_load(),
            "min_obj": input.min_obj(),
            "max_batt_mw": batt_range[1],
            "min_batt_mw": batt_range[0],
            "max_gas_mw": gas_range[1],
            "min_gas_mw": gas_range[0],
            "max_wind_mw": wind_range[1],
            "min_wind_mw": wind_range[0],
            "max_solar_mw": solar_range[1],
            "min_solar_mw": solar_range[0],
            "restrict_gas": input.restrict_gas(),
            # hard-coded to simplify input
            "min_charge_level": 0.1,
            "init_ch_level": 0.5,
            "batt_hours": 4,
            "batt_eff": 0.85,
            "use_outside_energy": True,
            "outside_energy_cost": input.outside_energy_cost(),
            "re_outage_start": input.re_outage_start(),
            "re_outage_days": input.re_outage_days(),
        }
        # emission factors + carbon-inclusive $/MW and $/MWh costs
        inputs.update(
            parameters.cost_inputs(
                co2_cost=input.co2_cost(),
                life_cycle_co2=input.life_cycle_co2(),
                wind_cost=input.wind_cost(),
                solar_cost=input.solar_cost(),
                batt_cost=input.batt_cost(),
                gas_mw_cost=input.gas_mw_cost(),
                gas_fuel_cost=input.gas_fuel_cost(),
            )
        )
        return inputs

    # -------------------------------------------------------------- #
    # non-blocking solve: run_lp is CPU-bound, so offload to a thread
    # so the UI event loop stays responsive during the ~1-minute solve
    # -------------------------------------------------------------- #
    @reactive.extended_task
    async def solve_task(run_name: str, inputs: dict) -> dict:
        """Run the LP solve in a worker thread so the UI stays responsive."""
        return await asyncio.to_thread(run_lp, run_name, inputs)

    @reactive.effect
    @reactive.event(input.run_button)
    def _launch_solve():
        """Start the background solve when create-run is clicked."""
        solve_task(input.run_name(), build_inputs())

    @reactive.effect
    def _collect_solve():
        """Store the solved results once the background task completes."""
        # raises SilentException until the task completes, then stores results
        results.set(solve_task.result())

    @reactive.effect
    def _solve_notification():
        """Show a progress toast while the solve runs; clear it when done."""
        # show a toast while the background solve runs; clear it when it finishes
        if solve_task.status() == "running":
            ui.notification_show(
                "Optimization running… (~15–60s)",
                id="solve_progress",
                duration=None,
                close_button=False,
                type="message",
            )
        else:
            ui.notification_remove("solve_progress")

    # -------------------------------------------------------------- #
    # result metrics (value boxes)
    # -------------------------------------------------------------- #
    @render.ui
    def capacity_boxes():
        """Resource-capacity value boxes; hidden until a run exists."""
        res = results.get()
        if res is None:
            return None
        cap = res["cap_mw"]
        return metric_section(
            "Resource capacities",
            [
                ui.value_box("Wind MW", f"{int(cap['wind_mw']):,}"),
                ui.value_box("Solar MW", f"{int(cap['solar_mw']):,}"),
                ui.value_box("Battery MW", f"{int(cap['batt_mw']):,}"),
                ui.value_box("Gas MW", f"{int(cap['gas_mw']):,}"),
            ],
            col_widths=3,
        )

    @render.ui
    def cost_boxes():
        """Cost-metric value boxes; hidden until a run exists."""
        res = results.get()
        if res is None:
            return None
        m = res["metrics"]
        return metric_section(
            "Cost metrics",
            [
                ui.value_box("Total cost (mill)", f"{int(m['total_cost_mill']):,}"),
                ui.value_box("Gen cost (mill)", f"{int(m['total_gen_cost_mill']):,}"),
                ui.value_box("CO2 cost (mill)", f"{int(m['total_co2_cost_mill']):,}"),
            ],
            col_widths=4,
        )

    @render.ui
    def generation_boxes():
        """Generation-metric value boxes; hidden until a run exists."""
        res = results.get()
        if res is None:
            return None
        m = res["metrics"]
        return metric_section(
            "Generation metrics",
            [
                ui.value_box("% excess gen", f"{int(m['excess_gen_percent'])}"),
                ui.value_box("% RE gen", f"{round(m['re_percent'], 1)}"),
                ui.value_box("% Gas gen", f"{round(m['gas_percent'], 1)}"),
                ui.value_box("Tons CO2 (thou)", f"{int(m['total_co2_thou_tons']):,}"),
                ui.value_box("Emergency MWh", f"{int(m['total_outside_energy']):,}"),
            ],
            col_widths=[2, 2, 2, 3, 3],
        )

    # -------------------------------------------------------------- #
    # hourly plot
    # -------------------------------------------------------------- #
    @render.ui
    def plot_section():
        """Plot controls and chart container; hidden until a run exists."""
        if results.get() is None:
            return None
        return ui.TagList(
            ui.hr(),
            ui.h4("Hourly load and generation plot"),
            ui.input_date(
                "start_date",
                "Plot start date",
                value="2030-07-07",
                min="2030-01-01",
                max="2030-12-31",
            ),
            ui.input_slider("num_days", "Number of days to plot", 1, 28, 14, step=1),
            output_widget("stack_plot"),
        )

    @render_widget
    def stack_plot():
        """Build the hourly stack chart for the selected date window."""
        res = req(results.get())
        start_date = input.start_date()
        num_days = input.num_days()
        req(start_date is not None, num_days is not None)

        plot_start = pd.Timestamp(start_date).strftime("%Y-%m-%d %H:%M:%S")
        plot_end = (pd.Timestamp(start_date) + pd.Timedelta(f"{num_days}d")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return get_resource_stack_plot(res["final_df"], plot_start, plot_end)

    @render.ui
    def inputs_section():
        """Inputs-table container; hidden until a run exists."""
        res = results.get()
        if res is None:
            return None
        return ui.TagList(
            ui.hr(),
            ui.h4("Inputs"),
            ui.output_data_frame("inputs_table"),
        )

    @render.data_frame
    def inputs_table():
        """Render the run's input dict as a two-column data grid."""
        res = req(results.get())
        items = res["inputs"]
        return render.DataGrid(
            pd.DataFrame({"input": list(items.keys()), "value": [str(v) for v in items.values()]}),
            height="400px",
        )

    # -------------------------------------------------------------- #
    # save / delete / download
    # -------------------------------------------------------------- #
    @reactive.calc
    def run_names() -> list[str]:
        """Saved run names, recomputed after each save/delete."""
        runs_version()  # take a dependency so this refreshes after mutations
        return [str(r) for r in db.get_runs()]

    @render.ui
    def save_section():
        """Save-run button and run label; hidden until a run exists."""
        res = results.get()
        if res is None:
            return None
        return ui.TagList(
            ui.hr(),
            ui.p(f"Showing results from run: {res['run_name']}"),
            ui.help_text("Existing runs with the same name will be overwritten."),
            ui.input_action_button("save_button", "Save run"),
        )

    @reactive.effect
    @reactive.event(input.save_button)
    def _save_run():
        """Persist the current run and refresh the run list."""
        res = results.get()
        if res is None:
            return
        db.add_run(res)
        db.zip_results()
        runs_version.set(runs_version.get() + 1)
        ui.notification_show(f"Saved run: {res['run_name']}", type="message")

    @render.ui
    def delete_section():
        """Delete-run select and its action buttons."""
        return ui.TagList(
            ui.input_select("delete_run", "Select run to delete", choices=run_names()),
            ui.input_action_button("delete_button", "delete run"),
            ui.input_action_button("delete_all_button", "delete all runs"),
        )

    @reactive.effect
    @reactive.event(input.delete_button)
    def _delete_run():
        """Delete the selected run and refresh the run list."""
        name = input.delete_run()
        if not name:
            return
        db.delete_run(name)
        if len(db.get_runs()) > 0:
            db.zip_results()
        runs_version.set(runs_version.get() + 1)
        ui.notification_show(f"Deleted run: {name}", type="warning")

    @reactive.effect
    @reactive.event(input.delete_all_button)
    def _delete_all_runs():
        """Clear all runs and remove the results archive."""
        db.clear_db()
        if os.path.exists(RESULTS_ZIP):
            os.remove(RESULTS_ZIP)
        runs_version.set(runs_version.get() + 1)
        ui.notification_show("Deleted all runs", type="warning")

    @render.ui
    def download_section():
        """Download button; hidden until at least one run is saved."""
        runs_version()  # refresh when runs change
        if len(db.get_runs()) == 0:
            return None
        return ui.TagList(
            ui.hr(),
            ui.h4("Download results"),
            ui.download_button("download_results", "Download results"),
        )

    @render.download(filename="planning_results.zip")
    def download_results():
        """Zip the saved runs and stream the archive to the browser."""
        db.zip_results()
        with open(RESULTS_ZIP, "rb") as fp:
            yield fp.read()


app = App(app_ui, server)
