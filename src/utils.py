import pandas as pd
import plotly.graph_objects as go


def get_resource_stack_plot(
    final_df: pd.DataFrame,
    plot_range_start_default: str = "2030-07-01",
    plot_range_end_default: str = "2030-07-14",
) -> go.Figure:
    """Build the stacked hourly generation-vs-load chart.

    Stacks the hourly resource dispatch (hydro, solar, wind, battery
    discharge, gas, and emergency/outside energy when any is used) as
    filled areas and overlays the load and load-plus-charge lines, with a
    range selector and slider defaulting to the given window.

    Args:
        final_df: Hourly solved values from run_lp, indexed by timestamp,
            with the resource, 2030_load, and load_and_charge columns.
        plot_range_start_default: Initial x-axis window start (ISO string).
        plot_range_end_default: Initial x-axis window end (ISO string).

    Returns:
        A Plotly figure of the hourly resource stack and load lines.
    """
    plot_df = final_df.copy()

    fig = go.Figure()
    # add hydro
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.hydro,
            name="hydro",
            mode="lines",
            line=dict(width=0.5, color="steelblue"),
            stackgroup="one",
        )
    )
    # add solar
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.solar,
            name="solar",
            mode="lines",
            line=dict(width=0.5, color="gold"),
            stackgroup="one",
        )
    )

    # add wind
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.wind,
            name="wind",
            mode="lines",
            line=dict(width=0.5, color="mediumseagreen"),
            stackgroup="one",
        )
    )

    # add batt discharge
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.batt_discharge,
            name="batt_discharge",
            mode="lines",
            line=dict(width=0.5, color="darkorchid"),
            stackgroup="one",
        )
    )

    # add gas
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.gas,
            name="gas",
            mode="lines",
            line=dict(width=0.5, color="coral"),
            stackgroup="one",
        )
    )

    if plot_df.outside_energy.sum() > 0:
        # add outside_energy
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df.outside_energy,
                name="outside_energy",
                mode="lines",
                line=dict(width=0.5, color="crimson"),
                stackgroup="one",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["2030_load"],
            name="2030_load",
            mode="lines",
            line=dict(width=1.5, color="black"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.load_and_charge,
            name="load_and_charge",
            mode="lines",
            line=dict(width=1.5, color="black", dash="dash"),
        )
    )

    fig.update_layout(
        height=600,
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
            range=[plot_range_start_default, plot_range_end_default],
        ),
    )

    return fig
