"""
Chart generation utilities for the Facticity dashboard.
"""
from plotly.colors import sequential
from plotly.colors import qualitative
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PRIMARY_BLUE, LIGHT_BLUE, MODERN_ORANGE


def generate_chart(
    title,
    x_labels,
    y_queries,
    auth0_overlay=None,
    secondary_y_label="Users",
    sort_key=None,
    use_secondary_y=True,
    is_daily=False  # Flag to disable user plots specifically for daily charts
):
    """
    Generates a Plotly bar chart with optional secondary Y-axis for Auth0 user data.

    Parameters:
    - title: str, title of the chart
    - x_labels: list of str, X-axis labels
    - y_queries: list of int, primary Y-axis values (queries)
    - auth0_overlay: DataFrame, Auth0 user data with 'new_users' and 'total_users'
    - secondary_y_label: str, label for the secondary Y-axis (default "Users")
    - sort_key: function, optional sorting function for x_labels (for quarterly charts)
    - use_secondary_y: bool, whether to include a secondary Y-axis
    - is_daily: bool, whether this is the daily chart (removes scatter plots)
    """
    # Sort the data if needed
    if sort_key:
        sorted_labels = sorted(x_labels, key=sort_key)
        label_to_query = dict(zip(x_labels, y_queries))
        sorted_queries = [label_to_query[label] for label in sorted_labels]
        x_labels = sorted_labels
        y_queries = sorted_queries
        if auth0_overlay is not None:
            auth0_overlay = auth0_overlay.reindex(x_labels, fill_value=0)

    # Create figure with secondary y-axis if it's NOT a daily chart
    fig = make_subplots(specs=[[{"secondary_y": not is_daily}]])

    # Primary Y-axis (Queries)
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_queries,
            name=f"<span style='color:{LIGHT_BLUE}'>Queries</span>",
            marker_color=LIGHT_BLUE,
            text=y_queries,
            textposition="outside"
        ),
        secondary_y=False
    )

    # Only add scatter plots if this is NOT a daily chart and auth0_overlay is provided
    if not is_daily and use_secondary_y and auth0_overlay is not None:
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=auth0_overlay["total_users"],
                name=f"<span style='color:{MODERN_ORANGE}'>Total Users</span>",
                mode="lines+markers",
                line=dict(color=MODERN_ORANGE, width=3)
            ),
            secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=auth0_overlay["new_users"],
                name=f"<span style='color:{MODERN_ORANGE}'>New Users</span>",
                mode="lines+markers",
                line=dict(color=MODERN_ORANGE, width=3, dash="dot")
            ),
            secondary_y=True
        )

    # Totals for annotation
    total_users = int(auth0_overlay["total_users"].iloc[-1]) if not is_daily and auth0_overlay is not None else 0
    total_queries = sum(y_queries)

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Period"
    )

    fig.update_yaxes(title_text="Queries", secondary_y=False, title_font_color=LIGHT_BLUE)
    if not is_daily and use_secondary_y:
        fig.update_yaxes(title_text=secondary_y_label, secondary_y=True, title_font_color=MODERN_ORANGE)

    # Annotation (Total Queries & Users)
    annotation_text = (
        f"<b><span style='color:{LIGHT_BLUE}; font-size:20px;'>Total Queries:</span></b> "
        f"<span style='font-size:20px; color:#2a2a2a;'>{total_queries}</span>"
    )
    if not is_daily and auth0_overlay is not None:
        annotation_text = (
            f"<b><span style='color:{MODERN_ORANGE}; font-size:20px;'>Total Users:</span></b> "
            f"<span style='font-size:20px; color:#2a2a2a;'>{total_users}</span><br><br>"
        ) + annotation_text

    fig.add_annotation(
        x=0.02,
        y=1.02,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        align="left",
        bgcolor="#f5f5f5",
        borderwidth=2,
        borderpad=6,
        font=dict(size=20, color="#2a2a2a"),
        xanchor="left",
        yanchor="top"
    )

    st.plotly_chart(fig, use_container_width=True)


def get_table_download_link(df, filename, link_text):
    """
    Creates a download link for a DataFrame.
    
    Args:
        df: DataFrame to download
        filename: Name of the downloaded file
        link_text: Text to display for the download link
        
    Returns:
        str: HTML link tag for downloading the DataFrame as CSV
    """
    import base64
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" target="_blank" download="{filename}">{link_text}</a>'


def generate_url_breakdown_chart(title, dates, url_counts):
    """
    Generates a stacked bar chart where each segment corresponds to a URL, using the Viridis palette.
    Shows the total sum on the outside top of each bar.

    Parameters:
    - title: str, title of the chart
    - dates: list of str, X-axis labels (e.g., dates)
    - url_counts: dict, mapping each URL (str) to a list of int counts (same length as dates)
    """
    from config import PRIMARY_BLUE, MODERN_ORANGE, LIGHT_BLUE
    import plotly.graph_objects as go
    import streamlit as st
    import numpy as np

    palette = [PRIMARY_BLUE, MODERN_ORANGE, LIGHT_BLUE]
    fig = go.Figure()

    # Calculate the total for each date
    totals = np.zeros(len(dates))
    for url, counts in url_counts.items():
        totals += np.array(counts)

    # Add one Bar trace per URL for stacking
    for i, (url, counts) in enumerate(url_counts.items()):
        fig.add_trace(
            go.Bar(
                x=dates,
                y=counts,
                name=url,
                marker_color=palette[i % len(palette)],
                text=counts,
                textposition="inside"
            )
        )

    # Add annotations for the total sum at the top of each bar
    for i, date in enumerate(dates):
        fig.add_annotation(
            x=date,
            y=totals[i],
            text=f"<b>{int(totals[i])}</b>",
            showarrow=False,
            yshift=10,  # Shift the annotation 10 pixels above the bar
            font=dict(size=14)
        )

    # Layout with legend placed below
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Request Count",
        barmode='stack',
        legend_title="Requester URL",
        legend=dict(
            orientation="h",         # horizontal layout
            yanchor="bottom",
            y=-.5,                  # pushes legend below chart
        )
    )

    # Render the chart
    st.plotly_chart(fig, use_container_width=True)


def generate_user_breakdown_chart(title, dates, url_users, color_map):
    """
    Generates a stacked bar chart where each segment corresponds to a URL, showing unique users per URL.
    Shows the total sum on the outside top of each bar.

    Parameters:
    - title: str, title of the chart
    - dates: list of str, X-axis labels (e.g., dates)
    - url_users: dict, mapping each URL (str) to a list of int user counts (same length as dates)
    """
    from config import PRIMARY_BLUE, MODERN_ORANGE, LIGHT_BLUE
    import plotly.graph_objects as go
    import streamlit as st
    import numpy as np

    default_palette = [PRIMARY_BLUE, MODERN_ORANGE, LIGHT_BLUE]
    fig = go.Figure()

    # Calculate the total for each date
    totals = np.zeros(len(dates))
    for url, counts in url_users.items():
        totals += np.array(counts)

    # Add one Bar trace per URL for stacking
    for i, (url, counts) in enumerate(url_users.items()):

        if color_map and url in color_map:
            marker_color = color_map[url]
        else:
            marker_color = default_palette[i % len(default_palette)]

        fig.add_trace(
            go.Bar(
                x=dates,
                y=counts,
                name=url,
                marker_color= marker_color,
                text=counts,
                textposition="inside"
            )
        )

    # Add annotations for the total sum at the top of each bar
    for i, date in enumerate(dates):
        fig.add_annotation(
            x=date,
            y=totals[i],
            text=f"<b>{int(totals[i])}</b>",
            showarrow=False,
            yshift=10,  # Shift the annotation 10 pixels above the bar
            font=dict(size=14)
        )

    # Layout with legend placed below
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Unique Users",  # Changed from "Request Count" to "Unique Users"
        barmode='stack',
        legend_title="Requester URL",
        legend=dict(
            orientation="h",         # horizontal layout
            yanchor="bottom",
            y=-.5,                  # pushes legend below chart
        )
    )

    # Render the chart
    st.plotly_chart(fig, use_container_width=True)


def generate_url_breakdown_chart(title, dates, url_counts, color_map=None):
    """
    Generates a stacked bar chart where each segment corresponds to a URL, using the Viridis palette.
    Shows the total sum on the outside top of each bar.

    Parameters:
    - title: str, title of the chart
    - dates: list of str, X-axis labels (e.g., dates)
    - url_counts: dict, mapping each URL (str) to a list of int counts (same length as dates)
    - color_map: dict, optional mapping of URL to specific color (e.g., {"API": GREEN_1})
    """
    from config import PRIMARY_BLUE, MODERN_ORANGE, LIGHT_BLUE
    import plotly.graph_objects as go
    import streamlit as st
    import numpy as np

    default_palette = [PRIMARY_BLUE, MODERN_ORANGE, LIGHT_BLUE]
    fig = go.Figure()

    # Calculate the total for each date
    totals = np.zeros(len(dates))
    for url, counts in url_counts.items():
        totals += np.array(counts)

    # Add one Bar trace per URL for stacking
    for i, (url, counts) in enumerate(url_counts.items()):
        # Use custom color if provided for this URL, otherwise use default palette
        if color_map and url in color_map:
            marker_color = color_map[url]
        else:
            marker_color = default_palette[i % len(default_palette)]

        fig.add_trace(
            go.Bar(
                x=dates,
                y=counts,
                name=url,
                marker_color=marker_color,
                text=counts,
                textposition="inside"
            )
        )

    # Add annotations for the total sum at the top of each bar
    for i, date in enumerate(dates):
        fig.add_annotation(
            x=date,
            y=totals[i],
            text=f"<b>{int(totals[i])}</b>",
            showarrow=False,
            yshift=10,  # Shift the annotation 10 pixels above the bar
            font=dict(size=14)
        )

    # Layout with legend placed below
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Request Count",
        barmode='stack',
        legend_title="Requester URL",
        legend=dict(
            orientation="h",         # horizontal layout
            yanchor="bottom",
            y=-.8,                  # pushes legend below chart
        )
    )

    # Render the chart
    st.plotly_chart(fig, use_container_width=True)
