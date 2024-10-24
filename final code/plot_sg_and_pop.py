import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_baseline_and_subgroup(subgroups_dict, baseline, dates, freq_xaxis_labels=5, aggregate_subgroup=False, baseline_color="blue", subgroup_color="green", subgroup_opacity=0.7, short_legend=False, linewidth=2, title = 'Subgroups plotted against the population baseline', yaxisname= 'percentage change (%)'):
    # Convert datetime to correct format
    date_labels = [dt.strftime('%d-%m-%Y') for dt in dates]
    
    # Determine which dates to display based on parameter freq_xaxis_labels
    tick_indices = list(range(0, len(date_labels), freq_xaxis_labels))
    tick_vals = [date_labels[i] for i in tick_indices]
    tick_text = [date_labels[i] for i in tick_indices]

    # Create the Plotly figure
    fig = go.Figure()



    if aggregate_subgroup:
        # Aggregate all subgroups into a single line by computing the mean at each time point
        aggregate_subgroup_ts = np.mean(np.array(list(subgroups_dict.values())), axis=0).tolist()

        # Add the aggregated subgroup trace in green with full opacity and short legend
        fig.add_trace(go.Scatter(
            x=date_labels,
            y=aggregate_subgroup_ts,
            mode='lines',
            name='Subgroup',  # Single legend entry for all subgroups --> default when aggregated
            line=dict(color=subgroup_color, width=linewidth),
            opacity=1  # Full opacity --> default when aggregated
        ))
    else:
        if short_legend:
            # Add each subgroup trace in green with adjustable opacity but without individual legend entries
            for idx, (key, subgroup) in enumerate(subgroups_dict.items()):
                fig.add_trace(go.Scatter(
                    x=date_labels,
                    y=subgroup,
                    mode='lines',
                    name='Subgroup',  # Single legend entry for all subgroups
                    line=dict(color=subgroup_color, width=linewidth),
                    opacity=subgroup_opacity,
                    showlegend=(idx == 0)  # Show legend only for the first subgroup trace
                ))
        else:
            # Add each subgroup trace in green with adjustable opacity and individual legend entries
            for key, subgroup in subgroups_dict.items():
                fig.add_trace(go.Scatter(
                    x=date_labels,
                    y=subgroup,
                    mode='lines',
                    name=key.capitalize(),
                    line=dict(color=subgroup_color, width=linewidth),
                    opacity=subgroup_opacity
                ))

    # Update the layout of the plot
    fig.update_layout(
        template='plotly_white',
        title=title,
        xaxis_title='Date',
        yaxis_title=yaxisname,
        xaxis=dict(
            tickmode='array',
            tickvals=tick_indices,
            ticktext=tick_text,
            tickangle=45,  # Rotate x-axis labels for better readability
            tickfont=dict(size=10)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=80)  # Adjust margins for better spacing
    )
    # Add the baseline trace in blue
    fig.add_trace(go.Scatter(
        x=date_labels,
        y=baseline,
        mode='lines',
        name='Baseline',
        line=dict(color=baseline_color, width=2)
    ))

    return fig

def plot_baseline_and_subgroup_heatmap(subgroups_dict, baseline, dates, freq_xaxis_labels=5, aggregate_subgroup=False, baseline_color="blue", subgroup_color="green", subgroup_opacity=0.7, short_legend=False, linewidth=2):
    # Convert datetime to correct format
    date_labels = [dt.strftime('%d-%m-%Y') for dt in dates]

    # Determine which dates to display based on parameter freq_xaxis_labels
    tick_indices = list(range(0, len(date_labels), freq_xaxis_labels))
    tick_vals = [date_labels[i] for i in tick_indices]
    tick_text = [date_labels[i] for i in tick_indices]

    # Create the Plotly figure
    fig = go.Figure()

    # Normalize the subgroups for coloring intensity based on density
    num_subgroups = len(subgroups_dict)
    cmap = cm.get_cmap('Reds')  # Choose Reds colormap
    norm = Normalize(vmin=0, vmax=num_subgroups)

    if aggregate_subgroup:
        # Aggregate all subgroups into a single line by computing the mean at each time point
        aggregate_subgroup_ts = np.mean(np.array(list(subgroups_dict.values())), axis=0).tolist()

        # Add the aggregated subgroup trace in green with full opacity and short legend
        fig.add_trace(go.Scatter(
            x=date_labels,
            y=aggregate_subgroup_ts,
            mode='lines',
            name='Subgroups',  # Single legend entry for all subgroups --> default when aggregated
            line=dict(color=subgroup_color, width=linewidth),
            opacity=1  # Full opacity --> default when aggregated
        ))
    else:
        if short_legend:
            # Add each subgroup trace in green with adjustable opacity but without individual legend entries
            for idx, (key, subgroup) in enumerate(subgroups_dict.items()):
                color = 'rgba' + str(cmap(norm(idx)))  # Convert Matplotlib color to RGBA
                fig.add_trace(go.Scatter(
                    x=date_labels,
                    y=subgroup,
                    mode='lines',
                    name='Subgroups',  # Single legend entry for all subgroups
                    line=dict(color=color,width=linewidth),
                    opacity=subgroup_opacity,
                    showlegend=(idx == 0)  # Show legend only for the first subgroup trace
                ))
        else:
            # Add each subgroup trace in a heatmap-like color scale with individual legend entries
            for idx, (key, subgroup) in enumerate(subgroups_dict.items()):
                color = 'rgba' + str(cmap(norm(idx)))  # Convert Matplotlib color to RGBA
                fig.add_trace(go.Scatter(
                    x=date_labels,
                    y=subgroup,
                    mode='lines',
                    name=key.capitalize(),
                    line=dict(color=color,width=linewidth),
                    opacity=subgroup_opacity
                ))

    # Update the layout of the plot
    fig.update_layout(
        template='plotly_white',
        title='Subgroups plotted against the population baseline',
        xaxis_title='Date',
        yaxis_title='Value',
        xaxis=dict(
            tickmode='array',
            tickvals=tick_indices,
            ticktext=tick_text,
            tickangle=45,  # Rotate x-axis labels for better readability
            tickfont=dict(size=10)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=80)  # Adjust margins for better spacing
    )
    # Add the baseline trace in blue
    fig.add_trace(go.Scatter(
        x=date_labels,
        y=baseline,
        mode='lines',
        name='Baseline',
        line=dict(color=baseline_color, width=2)
    ))

    return fig

def plot_baseline_and_subgroup_windows(subgroup_avg, baseline, dates, window_size=20, baseline_color="blue", subgroup_color="green", original_opacity=0.8, simplified_opacity=0.8, linewidth=2, title='Subgroups plotted against the population baseline', yaxisname='percentage change (%)'):
    # Convert datetime to correct format
    date_labels = [dt.strftime('%d-%m-%Y') for dt in dates]

    window_boundaries = list(range(0, len(dates), window_size))

    # Ensure the last date is included
    if window_boundaries[-1] != len(dates) - 1:
        window_boundaries.append(len(dates) - 1)

    # Determine tick labels and vertical lines at window boundaries
    tick_vals = window_boundaries
    tick_text = [date_labels[i] for i in window_boundaries]

    # Create the Plotly figure
    fig = go.Figure()

    # Add the subgroup average trace in green with adjusted opacity
    fig.add_trace(go.Scatter(
        x=date_labels,
        y=subgroup_avg,
        mode='lines',
        name='Subgroup Average',
        line=dict(color=subgroup_color, width=linewidth),
        opacity=original_opacity
    ))

    # Add the baseline trace in blue with adjusted opacity
    fig.add_trace(
        go.Scatter(
            x=date_labels,
            y=baseline,
            mode='lines',
            name='Baseline',
            line=dict(color=baseline_color, width=linewidth),
            opacity=original_opacity
        )
    )

    # Add vertical grey dashed lines at window boundaries
    for boundary in window_boundaries:
        fig.add_vline(x=date_labels[boundary], line=dict(color='grey', dash='dash', width=1))

    # Compute and plot simplified baseline (average per window)
    baseline_simplified = []
    subgroup_simplified = []
    window_midpoints = []

    for i in range(len(window_boundaries) - 1):
        start_idx = window_boundaries[i]
        end_idx = window_boundaries[i+1]

        # Compute average for baseline and subgroup within each window
        baseline_window_avg = np.mean(baseline[start_idx:end_idx+1])
        subgroup_window_avg = np.mean(subgroup_avg[start_idx:end_idx+1])

        baseline_simplified.append(baseline_window_avg)
        subgroup_simplified.append(subgroup_window_avg)

        # Get the midpoint date for plotting
        midpoint = start_idx + (end_idx - start_idx) // 2
        window_midpoints.append(date_labels[midpoint])

    # Plot simplified baseline with dots and connecting lines
    fig.add_trace(go.Scatter(
        x=window_midpoints,
        y=baseline_simplified,
        mode='lines+markers',
        name='Baseline (windows)',
        line=dict(color=baseline_color, width=linewidth),
        marker=dict(size=8, symbol='circle', color=baseline_color),
        opacity=simplified_opacity
    ))

    # Plot simplified subgroup average with dots and connecting lines
    fig.add_trace(go.Scatter(
        x=window_midpoints,
        y=subgroup_simplified,
        mode='lines+markers',
        name='Subgroup Average (windows)',
        line=dict(color=subgroup_color, width=linewidth),
        marker=dict(size=8, symbol='circle', color=subgroup_color),
        opacity=simplified_opacity
    ))

    # Update the layout of the plot
    fig.update_layout(
        template='plotly_white',
        title=title,
        xaxis_title='Date',
        yaxis_title=yaxisname,
        xaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=45,  # Rotate x-axis labels for better readability
            tickfont=dict(size=10)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=80)  # Adjust margins for better spacing
    )

    # Show the figure
    return fig



