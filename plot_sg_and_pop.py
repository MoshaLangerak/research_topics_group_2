import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_baseline_and_subgroup(subgroups_dict, baseline, dates, freq_xaxis_labels=5, aggregate_subgroup=False, baseline_color="blue", subgroup_color="green", subgroup_opacity=0.7, short_legend=False):
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
            name='Subgroups',  # Single legend entry for all subgroups --> default when aggregated
            line=dict(color=subgroup_color, width=2),
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
                    name='Subgroups',  # Single legend entry for all subgroups
                    line=dict(color=subgroup_color),
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
                    line=dict(color=subgroup_color),
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

def plot_baseline_and_subgroup_heatmap(subgroups_dict, baseline, dates, freq_xaxis_labels=5, aggregate_subgroup=False, baseline_color="blue", subgroup_color="green", subgroup_opacity=0.7, short_legend=False):
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
            line=dict(color=subgroup_color, width=2),
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
                    line=dict(color=color),
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
                    line=dict(color=color),
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

# # Plot two test cases:
#
# # TEST CASE 1: Three time series each with 100 points
#
# # Create subgroups
# subgroup_ts1 = list(3 + np.random.normal(0, 0.5, 100))  # Mean 3, small noise with standard deviation of 0.5
# subgroup_ts2 = list(5 + np.random.normal(0, 0.5, 100))  # Mean 5, small noise with standard deviation of 0.5
# subgroup_all_ts = {'ts1': subgroup_ts1, 'ts2': subgroup_ts2}
#
# # Create baseline (population line)
# baseline = list(0 + np.random.normal(0, 0.5, 100))  # Mean 0, small noise with standard deviation of 0.5
#
# # Create the dates
# start_date = datetime.datetime.now()
# dates = [start_date + datetime.timedelta(days=i) for i in range(100)]
#
# fig = plot_baseline_and_subgroup(subgroup_all_ts, baseline, dates)
# fig.show()
#
# # TEST CASE 2: Create n time series
#
# num_subgroups = 10
#
# baseline = list(0 + np.random.normal(0, 0.5, 100))
#
# # Initialize an empty dictionary to store subgroup time series
# subgroups_dict = {}
#
# for i in range(1, num_subgroups + 1):
#     # For diversity, adjust the mean for each subgroup
#     mean = 3 + i * 0.5
#     subgroup = list(mean + np.random.normal(0, 0.5, 100))
#     subgroup_all_ts[f'ts{i}'] = subgroup  # Keys: 'ts1', 'ts2', ..., 'tsn'
#
# # Create the dates
# start_date = datetime.datetime.now()
# dates = [start_date + datetime.timedelta(days=i) for i in range(100)]
#
# fig = plot_baseline_and_subgroup(subgroup_all_ts, baseline, dates, subgroup_opacity=0.4, short_legend=True)
# fig.show()
