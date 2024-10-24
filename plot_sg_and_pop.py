import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_baseline_and_subgroup(subgroups_dict, baseline, dates, freq_xaxis_labels=5, aggregate_subgroup=False, baseline_color="blue", subgroup_color="green", subgroup_opacity=0.7, short_legend=False, linewidth=2, title='Subgroups plotted against the population baseline', yaxisname='percentage change (%)'):
    """
    Plots time series data of subgroups against a population baseline using Plotly.

    Args:
        subgroups_dict (dict): Dictionary containing subgroup time series data with keys as subgroup names.
        baseline (list): List of baseline values to plot.
        dates (list): List of datetime objects for the x-axis.
        freq_xaxis_labels (int, optional): Frequency of x-axis labels (e.g., every 5th date). Default is 5.
        aggregate_subgroup (bool, optional): If True, aggregate all subgroups into one by averaging. Default is False.
        baseline_color (str, optional): Color of the baseline line. Default is "blue".
        subgroup_color (str, optional): Color of the subgroup lines. Default is "green".
        subgroup_opacity (float, optional): Opacity for subgroup lines (between 0 and 1). Default is 0.7.
        short_legend (bool, optional): If True, display a single legend entry for all subgroups. Default is False.
        linewidth (int, optional): Line thickness for both baseline and subgroups. Default is 2.
        title (str, optional): Plot title. Default is 'Subgroups plotted against the population baseline'.
        yaxisname (str, optional): Label for the y-axis. Default is 'percentage change (%)'.

    Returns:
        fig (go.Figure): Plotly figure object with the plot.
    """
    # Convert datetime objects to formatted strings
    date_labels = [dt.strftime('%d-%m-%Y') for dt in dates]

    # Determine which date labels to show based on freq_xaxis_labels
    tick_indices = list(range(0, len(date_labels), freq_xaxis_labels))
    tick_text = [date_labels[i] for i in tick_indices]

    # Create the Plotly figure
    fig = go.Figure()

    if aggregate_subgroup:
        # Average all subgroup time series to create an aggregate line
        aggregate_subgroup_ts = np.mean(np.array(list(subgroups_dict.values())), axis=0).tolist()

        # Add the aggregated subgroup trace (full opacity, single legend entry)
        fig.add_trace(go.Scatter(
            x=date_labels,
            y=aggregate_subgroup_ts,
            mode='lines',
            name='Subgroup',  # One legend entry for aggregated subgroups
            line=dict(color=subgroup_color, width=linewidth),
            opacity=1  # Full opacity for aggregate
        ))
    else:
        if short_legend:
            # Plot each subgroup with one legend entry for all (first trace only)
            for idx, (key, subgroup) in enumerate(subgroups_dict.items()):
                fig.add_trace(go.Scatter(
                    x=date_labels,
                    y=subgroup,
                    mode='lines',
                    name='Subgroup',  # Single legend entry for all subgroups
                    line=dict(color=subgroup_color, width=linewidth),
                    opacity=subgroup_opacity,
                    showlegend=(idx == 0)  # Show legend for the first subgroup only
                ))
        else:
            # Plot each subgroup with its own legend entry
            for key, subgroup in subgroups_dict.items():
                fig.add_trace(go.Scatter(
                    x=date_labels,
                    y=subgroup,
                    mode='lines',
                    name=key.capitalize(),  # Individual legend entry per subgroup
                    line=dict(color=subgroup_color, width=linewidth),
                    opacity=subgroup_opacity
                ))

    # Update layout with titles, axis labels, and customized x-axis tick formatting
    fig.update_layout(
        template='plotly_white',
        title=title,
        xaxis_title='Date',
        yaxis_title=yaxisname,
        xaxis=dict(
            tickmode='array',
            tickvals=tick_indices,
            ticktext=tick_text,
            tickangle=45,  # Rotate x-axis labels for clarity
            tickfont=dict(size=10)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=80)  # Set plot margins for better spacing
    )

    # Add the baseline trace (blue line)
    fig.add_trace(go.Scatter(
        x=date_labels,
        y=baseline,
        mode='lines',
        name='Baseline',
        line=dict(color=baseline_color, width=linewidth)
    ))

    return fig




