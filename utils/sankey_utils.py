"""
Sankey diagram utilities for the Facticity dashboard.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import time
import gzip
import io
import requests

from config import BLACKLIST_EMAILS, BLACKLIST_DOMAINS
from database.mongo_client import get_db_connection, is_valid_user
from database.auth0_client import get_auth0_token, export_all_users
from utils.posthog_utils import get_posthog_data
import streamlit as st


def process_auth0_users(file_url):
    """
    Process Auth0 user data and filter by date range.

    Args:
        file_url: URL to the Auth0 export file
        start_date: Start date as datetime object
        end_date: End date as datetime object

    Returns:
        DataFrame: DataFrame with Auth0 user data filtered by date range
    """
    if not file_url:
        return pd.DataFrame()

    response = requests.get(file_url)
    with gzip.open(io.BytesIO(response.content), 'rb') as f:
        df = pd.read_csv(f, encoding='utf-8')
    # Clean up email column
    if "email" in df.columns:
        df["email"] = df["email"].astype(str).str.lower().str.strip()

    # Convert timestamp columns to datetime
    if "last_login" in df.columns:
        df["last_login"] = pd.to_datetime(
            df["last_login"], errors="coerce", utc=True)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(
            df["created_at"], errors="coerce", utc=True)

    filtered_df = df
    # Extract ip_address from last_ip
    filtered_df['ip_address'] = filtered_df['last_ip'].astype(
        str).str.split("'").str[1]
    filtered_df['userEmail'] = filtered_df['email'].astype(
        str).str.replace("'", "", regex=False).str.lower().str.strip()

    return filtered_df


def export_auth0_users():
    """
    Exports Auth0 users within a date range.

    Args:
        start_date: Start date as datetime object
        end_date: End date as datetime object

    Returns:
        str: URL to the exported file or None if failed
    """
    mgmt_api_token = get_auth0_token()
    if not mgmt_api_token:
        return None

    headers = {
        "Authorization": f"Bearer {mgmt_api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "format": "csv",
        "fields": [
            {"name": "user_id"},
            {"name": "email"},
            {"name": "name"},
            {"name": "logins_count"},
            {"name": "created_at"},
            {"name": "updated_at"},
            {"name": "last_login"},
            {"name": "last_ip"}
        ]
    }

    try:
        from config import AUTH0_DOMAIN
        response = requests.post(f"https://{AUTH0_DOMAIN}/api/v2/jobs/users-exports",
                                 headers=headers, json=payload)
        response.raise_for_status()
        job_id = response.json().get("id")

        # Poll for job completion (max 15 iterations, 5 sec apart)
        for _ in range(15):
            time.sleep(5)
            status_resp = requests.get(f"https://{AUTH0_DOMAIN}/api/v2/jobs/{job_id}",
                                       headers=headers)
            if status_resp.json().get("status") == "completed":
                return status_resp.json().get("location")
        raise Exception("Auth0 export timed out")
    except Exception as e:
        print("Error exporting Auth0 users:", e)
        return None


def get_mongodb_data(start_date, end_date):
    """
    Get MongoDB query data for a specific date range.

    Args:
        start_date: Start date as datetime object
        end_date: End date as datetime object

    Returns:
        DataFrame: DataFrame with MongoDB query data
    """
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()

    pipeline = [
        {"$match": {"timestamp": {"$gte": start_iso, "$lt": end_iso}}},
        {"$project": {
            "userEmail": {"$ifNull": ["$userEmail", ""]},
            "timestamp": 1
        }},
        {"$group": {
            "_id": "$userEmail",
            "count": {"$sum": 1}
        }},
        {"$project": {
            "userEmail": "$_id",
            "count": 1,
            "_id": 0
        }}
    ]

    collection = get_db_connection()
    agg_result = list(collection.aggregate(pipeline))

    if not agg_result:
        return pd.DataFrame(columns=["userEmail", "count"])

    agg_df = pd.DataFrame(agg_result)

    # Ensure userEmail is properly formatted
    agg_df["userEmail"] = agg_df["userEmail"].apply(
        lambda x: "" if pd.isna(x) or x is None else str(x).lower().strip())

    return agg_df


def process_week_data(start_date, end_date):
    """Process all data sources for a specific week."""
    # Debug info
    st.info(f"Attempting to process data from {start_date} to {end_date}")

    # 1. Get Auth0 users data - add debug
    st.info("Requesting Auth0 export...")
    auth0_file_url = export_auth0_users(start_date, end_date)
    if not auth0_file_url:
        st.error("Failed to get Auth0 export URL")
        return None

    st.info("Processing Auth0 users...")
    auth0_users = process_auth0_users(auth0_file_url, start_date, end_date)
    # etc.

    if auth0_users.empty:
        print(
            f"No Auth0 users found for week {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        return None

    # 2. Get PostHog sessions data
    posthog_df = get_posthog_data(start_date, end_date)

    if posthog_df.empty:
        print(
            f"No PostHog data found for week {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        return None

    # 3. Get MongoDB query data
    mongodb_df = get_mongodb_data(start_date, end_date)

    # 4. Filter valid Auth0 users (exclude blacklisted domains/emails)
    auth0_valid_users = auth0_users[auth0_users.apply(
        lambda row: is_valid_user(row['userEmail']), axis=1)].copy()

    # 5. Process session data
    session_df = posthog_df.copy()
    session_df["ip_address"] = session_df["ip_addresses"].apply(
        lambda ips: ips[0] if isinstance(ips, list) and len(ips) > 0 else None
    )

    # Identify shared IP addresses
    auth0_shared_ip_addresses = auth0_valid_users[auth0_valid_users.groupby(
        'ip_address')['ip_address'].transform('count') > 1]

    # Mark sessions that come from a shared IP address
    session_df["is_shared_ip"] = session_df["ip_address"].isin(
        auth0_shared_ip_addresses["ip_address"])

    # Sort sessions by latest session_start
    session_df = session_df.sort_values(by="session_start", ascending=False)

    # For sessions from shared IPs, keep only a limited number
    shared_ips_df = session_df[session_df["is_shared_ip"] == True].copy()
    filtered_shared_ips = pd.DataFrame()
    if not shared_ips_df.empty:
        filtered_shared_ips = shared_ips_df.groupby("ip_address", group_keys=False).apply(
            lambda x: x.head(
                auth0_shared_ip_addresses["ip_address"].value_counts().get(x.name, 1))
        )

    # For non-shared IPs, keep the first occurrence per IP
    non_shared_ips_df = session_df[~session_df["is_shared_ip"]].drop_duplicates(
        subset=["ip_address"], keep="first")

    # Combine both subsets to form final_df
    final_df = pd.concat([filtered_shared_ips, non_shared_ips_df])
    final_df = final_df.drop(columns=["is_shared_ip"]).reset_index(drop=True)

    # Flag each session as logged in if its IP is found in the Auth0 user list
    auth0_ip_set = set(auth0_users['ip_address'].dropna().unique())
    final_df['logged_in'] = final_df['ip_address'].apply(
        lambda ip: ip in auth0_ip_set)

    # 6. Merge Auth0 users with MongoDB query counts
    merged_df = pd.merge(auth0_valid_users, mongodb_df,
                         on="userEmail", how="left")
    merged_df["count"] = merged_df["count"].fillna(0).astype(int)

    # Define bins based on query count
    def assign_bin(count):
        if count == 0:
            return "0"
        elif count == 1:
            return "1"
        else:
            return ">1"

    merged_df["bin"] = merged_df["count"].apply(assign_bin)

    # 7. Compute metrics for Sankey diagram
    # Count new vs. returning users
    new_users_count = auth0_valid_users[auth0_valid_users['logins_count'] == 1].shape[0]
    returning_users_count = auth0_valid_users[auth0_valid_users['logins_count'] > 1].shape[0]

    # Get counts for event bins
    bin_1 = merged_df['bin'].value_counts().get('1', 0)
    bin_more = merged_df['bin'].value_counts().get('>1', 0)
    bin_0 = merged_df['bin'].value_counts().get('0', 0)

    # Calculate query counts
    not_logged_in_total = mongodb_df.loc[~mongodb_df['userEmail'].isin(
        auth0_valid_users['userEmail']), 'count'].sum()
    logged_in_total = merged_df['count'].sum()

    # Return all metrics needed for the Sankey diagram
    return {
        'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'logged_in_count': final_df['logged_in'].value_counts().get(True, 0),
        'not_logged_in_count': final_df['logged_in'].value_counts().get(False, 0),
        'new_users_count': new_users_count,
        'returning_users_count': returning_users_count,
        'bin_1': bin_1,
        'bin_more': bin_more,
        'bin_0': bin_0,
        'not_logged_in_total': not_logged_in_total,
        'logged_in_total': logged_in_total
    }


def create_sankey_diagram(metrics):
    """
    Create a Sankey diagram from the provided metrics.

    Args:
        metrics: Dictionary with metrics

    Returns:
        Figure: Plotly Figure object
    """
    node_labels = [
        "All PostHog Users",      # 0
        "Estimated Logged In",    # 1
        "Estimated Not Logged In",  # 2
        "No. of New Users",       # 3
        "No. of Returning Users",  # 4
        "1 Query",                # 5
        ">1 Query",               # 6
        "No Queries",             # 7
        "Non-Login Queries",      # 8
        "Logged In Queries",      # 9
        "Total Queries"           # 10
    ]

    source = [
        0, 0,  # All Sessions -> Logged In, Not Logged In
        1, 1,  # Logged In -> New Users, Returning Users
        3, 4,  # New Users -> 1 Query, Returning Users -> 1 Query
        3, 4,  # New Users -> >1 Query, Returning Users -> >1 Query
        3, 4,  # New Users -> No Queries, Returning Users -> No Queries
        2,     # Not Logged In -> Non-Login Queries
        5, 6, 7,  # 1 Query, >1 Query, No Queries -> Logged In Queries
        8, 9   # Non Login Queries and Logged In Queries -> Total Queries
    ]

    target = [
        1, 2,  # All Sessions -> Logged In, Not Logged In
        3, 4,  # Logged In -> New Users, Returning Users
        5, 5,  # New Users -> 1 Query, Returning Users -> 1 Query
        6, 6,  # New Users -> >1 Query, Returning Users -> >1 Query
        7, 7,  # New Users -> No Queries, Returning Users -> No Queries
        8,     # Not Logged In -> Non-Login Queries
        9, 9, 9,  # 1 Query, >1 Query, No Queries -> Logged In Queries
        10, 10  # Non Login Queries and Logged In Queries -> Total Queries
    ]

    # Calculate proportional distribution for new vs returning users
    total_users = metrics['new_users_count'] + metrics['returning_users_count']
    new_ratio = metrics['new_users_count'] / \
        total_users if total_users > 0 else 0
    returning_ratio = metrics['returning_users_count'] / \
        total_users if total_users > 0 else 0

    value = [
        metrics['logged_in_count'],  # All Sessions -> Logged In
        metrics['not_logged_in_count'],  # All Sessions -> Not Logged In
        metrics['new_users_count'],  # Logged In -> New Users
        metrics['returning_users_count'],  # Logged In -> Returning Users
        metrics['bin_1'] * new_ratio,  # New Users -> 1 Query
        metrics['bin_1'] * returning_ratio,  # Returning Users -> 1 Query
        metrics['bin_more'] * new_ratio,  # New Users -> >1 Query
        metrics['bin_more'] * returning_ratio,  # Returning Users -> >1 Query
        metrics['bin_0'] * new_ratio,  # New Users -> No Queries
        metrics['bin_0'] * returning_ratio,  # Returning Users -> No Queries
        metrics['not_logged_in_total'],  # Not Logged In -> Non-Login Queries
        metrics['bin_1'],  # 1 Query -> Logged In Queries
        metrics['bin_more'],  # >1 Query -> Logged In Queries
        0,  # No Queries -> Logged In Queries (should be 0)
        metrics['not_logged_in_total'],  # Non-Login Queries -> Total Queries
        metrics['logged_in_total']  # Logged in Queries -> Total Queries
    ]

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="blue",
            x=[0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.6,
                0.8, 0.8, 1.0],  # Set x positions
            y=[0.5, 0.2, 0.8, 0.1, 0.3, 0.05, 0.25,
                0.45, 0.5, 0.7, 0.5]  # set y positions
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
        )
    )])

    fig.update_layout(
        title_text=f"PostHog Users: {metrics['period']}",
        font_size=12
    )

    return fig


def generate_week_over_week_analysis(num_weeks=4):
    """
    Generate Sankey diagrams for the specified number of past weeks.

    Args:
        num_weeks: Number of weeks to analyze

    Returns:
        tuple: (Figures, Summary DataFrame)
    """
    now = datetime.now(timezone.utc)

    # Create a list to store weekly metrics
    weekly_metrics = []
    weekly_figures = []

    # Process data for each week
    for i in range(num_weeks):
        end_date = now - timedelta(days=7*i)
        start_date = end_date - timedelta(days=7)

        print(
            f"Processing week {i+1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        metrics = process_week_data(start_date, end_date)
        if metrics:
            weekly_metrics.append(metrics)
            weekly_figures.append(create_sankey_diagram(metrics))

    # Create a summary DataFrame for easy comparison
    if weekly_metrics:
        summary_df = pd.DataFrame(weekly_metrics)

        # Calculate week-over-week changes
        for col in ['logged_in_count', 'not_logged_in_count', 'new_users_count',
                    'returning_users_count', 'bin_1', 'bin_more', 'bin_0',
                    'not_logged_in_total', 'logged_in_total']:
            summary_df[f'{col}_wow_change'] = summary_df[col].pct_change(
                -1) * 100

        # Display all Sankey diagrams
        return weekly_figures, summary_df

    return [], pd.DataFrame()


def create_combined_visualization(figures, summary_df):
    """
    Create a combined visualization with all Sankey diagrams and a trend chart.

    Args:
        figures: List of Plotly Figure objects
        summary_df: DataFrame with summary metrics

    Returns:
        Figure: Combined Plotly Figure object
    """
    if not figures or summary_df.empty:
        return None

    # Create a subplot for the Sankey diagrams
    num_weeks = len(figures)
    fig_combined = make_subplots(
        rows=num_weeks + 1,
        cols=1,
        subplot_titles=[
            f"Week {i+1}: {summary_df.iloc[i]['period']}" for i in range(num_weeks)],
        # + ["Weekly Trends"],
        vertical_spacing=0.1,
        specs=[[{"type": "sankey"}]
               for _ in range(num_weeks)] + [[{"type": "scatter"}]]
    )

    # Add each Sankey diagram
    for i, fig in enumerate(figures):
        for trace in fig.data:
            fig_combined.add_trace(trace, row=i+1, col=1)

    # # Add trend lines for key metrics
    # metrics_to_plot = [
    #     'new_users_count',
    #     'returning_users_count',
    #     'logged_in_total',
    #     'not_logged_in_total'
    # ]

    # colors = ['blue', 'green', 'orange', 'red']

    # for i, metric in enumerate(metrics_to_plot):
    #     fig_combined.add_trace(
    #         go.Scatter(
    #             x=summary_df['period'],
    #             y=summary_df[metric],
    #             mode='lines+markers',
    #             name=metric,
    #             line=dict(color=colors[i % len(colors)]),
    #         ),
    #         row=num_weeks+1,
    #         col=1
    #     )

    # Update layout
    fig_combined.update_layout(
        height=400 * (num_weeks + 1),
        width=1000,
        title_text="Week-over-Week PostHog User Analysis",
    )

    return fig_combined
