
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def get_resource_stack_plot(
        final_df: pd.DataFrame,
        plot_range_start_default: str = "2030-07-01",
        plot_range_end_default: str = "2030-07-14",
) -> go.Figure:
    plot_df = final_df.copy()

    fig = go.Figure()
    # add hydro
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.hydro,
            name='hydro',
            mode='lines',
            line=dict(width=0.5, color='steelblue'),
            stackgroup='one',
        )
    )
    # add solar
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.solar,
            name='solar',
            mode='lines',
            line=dict(width=0.5, color='gold'),
            stackgroup='one',
        )
    )

    # add wind
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.wind,
            name='wind',
            mode='lines',
            line=dict(width=0.5, color='mediumseagreen'),
            stackgroup='one',
        )
    )

    # add batt discharge
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.batt_discharge,
            name='batt_discharge',
            mode='lines',
            line=dict(width=0.5, color='darkorchid'),
            stackgroup='one',
        )
    )

    # add gas
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.gas,
            name='gas',
            mode='lines',
            line=dict(width=0.5, color='coral'),
            stackgroup='one',
        )
    )

    if plot_df.outside_energy.sum() > 0:
        # add outside_energy
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df.outside_energy,
                name='outside_energy',
                mode='lines',
                line=dict(width=0.5, color='crimson'),
                stackgroup='one',
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df['2030_load'],
            name='2030_load',
            mode='lines',
            line=dict(width=1.5, color='black'),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df.load_and_charge,
            name='load_and_charge',
            mode='lines',
            line=dict(width=1.5, color='black', dash='dash'),
        )
    )

    fig.update_layout(
        height=600,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date",
            range=[plot_range_start_default, plot_range_end_default]
        )
    )

    return fig


def plot_hourly(final_df, start_date, num_days):

    cols = ['hydro', 'solar', 'wind', 'batt_discharge', 'gas', 'outside_energy']

    # get valid columns
    cols = [c for c in cols if c in final_df.columns]

    if (start_date is not None) and (np.any(final_df.index == pd.to_datetime(start_date))):
        t0 = np.where(final_df.index == pd.to_datetime(start_date))[0][0]
    else:
        # randomly sample day
        t0 = int(np.random.randint(0, len(final_df.index) - 24 * num_days, size=1))

    start_time = final_df.index[t0]
    end_time = final_df.index[t0 + 24 * num_days]

    # find the max y value for placing the legend
    ymax = final_df.loc[start_time:end_time, cols].sum(axis=1).max()

    # col pallete
    pal = ["steelblue", "gold", "mediumseagreen", "darkorchid", "coral", 'crimson']

    ax = final_df.loc[start_time:end_time, ['2030_load']].plot.line(color='black');
    final_df.loc[start_time:end_time, ['load_and_charge']].plot.line(ax=ax, color='black', linestyle='--');
    final_df.loc[start_time:end_time, cols].plot.area(ax=ax, linewidth=0, figsize=(12, 6), color=pal);

    ax.set_ylim(0, ymax + 250)
    plt.ylabel('MW')
