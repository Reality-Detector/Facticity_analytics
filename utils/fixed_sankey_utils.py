"""
Fixed Sankey diagram utilities that address the login detection issue.
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
from database.auth0_client import get_auth0_token
from utils.sankey_utils import (
    export_auth0_users, process_auth0_users, get_mongodb_data, create_combined_visualization
)


def create_time_bound_sankey(start_date, end_date, exclude_blacklisted):
    """
    Create a Sankey diagram that correctly handles time-bound logins.
    
    Args:
        start_date: Start date as datetime object
        end_date: End date as datetime object
        
    Returns:
        tuple: (Figure, Metrics)
    """
    # Get ALL Auth0 users without date filtering
    auth0_file_url = export_auth0_users()
    # Remove date filtering from process_auth0_users call
    auth0_users = process_auth0_users(auth0_file_url)
    if auth0_users.empty:
        print("No Auth0 users found")
        return None, None

    # Filter valid Auth0 users
    if exclude_blacklisted:
        auth0_valid_users = auth0_users[auth0_users.apply(
            lambda row: is_valid_user(row['userEmail']), axis=1)].copy()
    else:
        auth0_valid_users = auth0_users.copy()
        
    # Get query data for the period (this keeps the date filtering)
    mongodb_df = get_mongodb_data(start_date, end_date)

    # Count active users (users who made at least one query during this period)
    active_users = mongodb_df[mongodb_df['userEmail'].isin(
        auth0_valid_users['userEmail'])]
    active_user_count = len(
        active_users['userEmail'].unique()) if not active_users.empty else 0

    # Count inactive users (users with accounts but no queries during this period)
    inactive_user_count = len(auth0_valid_users) - active_user_count

    # Count anonymous queries (queries without a valid userEmail)
    anonymous_queries = mongodb_df[~mongodb_df['userEmail'].isin(
        auth0_valid_users['userEmail'])]
    anonymous_query_count = anonymous_queries['count'].sum(
    ) if not anonymous_queries.empty else 0

    # Distinguish between new and returning users
    # Note: For this time period, "new" should be users created during this period
    new_users = auth0_valid_users[
        (pd.to_datetime(auth0_valid_users['created_at'], errors='coerce', utc=True) >= start_date) &
        (pd.to_datetime(
            auth0_valid_users['created_at'], errors='coerce', utc=True) <= end_date)
    ]
    returning_users = auth0_valid_users[~auth0_valid_users['userEmail'].isin(
        new_users['userEmail'])]

    # Count active new/returning users
    active_new_users = active_users[active_users['userEmail'].isin(
        new_users['userEmail'])]
    active_returning_users = active_users[active_users['userEmail'].isin(
        returning_users['userEmail'])]

    # Group active users by query count
    active_user_data = pd.merge(
        active_users,
        auth0_valid_users[['userEmail', 'logins_count']],
        on='userEmail',
        how='left'
    )

    # Define query frequency buckets
    def categorize_queries(count):
        if count == 1:
            return '1 Query'
        elif count <= 5:
            return '2-5 Queries'
        else:
            return '6+ Queries'

    active_user_data['query_category'] = active_user_data['count'].apply(
        categorize_queries)

    # Count users in each query frequency bucket
    query_distribution = active_user_data['query_category'].value_counts(
    ).to_dict()
    single_query_users = query_distribution.get('1 Query', 0)
    few_query_users = query_distribution.get('2-5 Queries', 0)
    many_query_users = query_distribution.get('6+ Queries', 0)

    # Calculate total queries from registered users
    registered_query_count = active_user_data['count'].sum(
    ) if not active_user_data.empty else 0


    anonymous_queries = mongodb_df[~mongodb_df['userEmail'].isin(
        auth0_valid_users['userEmail'])]
    anonymous_query_count = anonymous_queries['count'].sum()

    # Count new vs returning users
    new_users

    # Prepare metrics
    metrics = {
        'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'total_users': len(auth0_valid_users),
        'active_users': active_user_count,
        'inactive_users': inactive_user_count,
        'new_users': len(new_users),
        'returning_users': len(returning_users),
        'active_new_users': len(active_new_users),
        'active_returning_users': len(active_returning_users),
        'single_query_users': single_query_users,
        'few_query_users': few_query_users,
        'many_query_users': many_query_users,
        'registered_query_count': registered_query_count,
        'anonymous_query_count': anonymous_query_count,
        'total_query_count': registered_query_count + anonymous_query_count
    }

    # Create the Sankey diagram
    node_labels = [
        "All Users",            # 0
        "Active Users",         # 1
        "Inactive Users",       # 2
        "New Users",            # 3
        "Returning Users",      # 4
        "1 Query",              # 5
        "2-5 Queries",          # 6
        "6+ Queries",           # 7
        "Anonymous Queries",    # 8
        "Registered Queries",   # 9
        "Total Queries"         # 10
    ]

    source = [
        0, 0,                   # All Users -> Active, Inactive
        1, 1,                   # Active Users -> New, Returning Users
        3, 3, 3,                # New Users -> 1, 2-5, 6+ Queries
        4, 4, 4,                # Returning Users -> 1, 2-5, 6+ Queries
        5, 6, 7,                # All query categories -> Registered Queries
        8, 9                    # Anonymous/Registered Queries -> Total Queries
    ]

    target = [
        1, 2,                   # All Users -> Active, Inactive
        3, 4,                   # Active Users -> New, Returning
        5, 6, 7,                # New Users -> Query categories
        5, 6, 7,                # Returning Users -> Query categories
        9, 9, 9,                # Query categories -> Registered Queries
        10, 10                  # Anonymous/Registered Queries -> Total
    ]

    # Calculate proportional values for query distribution
    new_user_ratio = metrics['active_new_users'] / \
        metrics['active_users'] if metrics['active_users'] > 0 else 0
    returning_user_ratio = metrics['active_returning_users'] / \
        metrics['active_users'] if metrics['active_users'] > 0 else 0

    new_single = metrics['single_query_users'] * new_user_ratio
    new_few = metrics['few_query_users'] * new_user_ratio
    new_many = metrics['many_query_users'] * new_user_ratio

    returning_single = metrics['single_query_users'] * returning_user_ratio
    returning_few = metrics['few_query_users'] * returning_user_ratio
    returning_many = metrics['many_query_users'] * returning_user_ratio

    values = [
        metrics['active_users'],        # All Users -> Active
        metrics['inactive_users'],      # All Users -> Inactive
        metrics['active_new_users'],    # Active -> New
        metrics['active_returning_users'],  # Active -> Returning
        new_single,                     # New -> 1 Query
        new_few,                        # New -> 2-5 Queries
        new_many,                       # New -> 6+ Queries
        returning_single,               # Returning -> 1 Query
        returning_few,                  # Returning -> 2-5 Queries
        returning_many,                 # Returning -> 6+ Queries
        metrics['single_query_users'],  # 1 Query -> Registered
        metrics['few_query_users'],     # 2-5 Queries -> Registered
        metrics['many_query_users'],    # 6+ Queries -> Registered
        metrics['anonymous_query_count'],  # Anonymous -> Total
        metrics['registered_query_count']  # Registered -> Total
    ]

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=values
        )
    )])

    # Update layout
    fig.update_layout(
        title_text=f"User Activity Flow: {metrics['period']}",
        font_size=12
    )

    return fig, metrics


def generate_time_bound_analysis(num_weeks=4, exclude_blacklisted=False):
    """
    Generate time-bound Sankey diagrams for the specified number of past weeks.
    
    Args:
        num_weeks: Number of weeks to analyze
        
    Returns:
        tuple: (Figures, Summary DataFrame)
    """
    now = datetime.now(timezone.utc)

    # Create lists to store data
    weekly_metrics = []
    weekly_figures = []

    # Process data for each week
    for i in range(num_weeks):
        end_date = now - timedelta(days=7*i)
        start_date = end_date - timedelta(days=7)

        print(
            f"Processing week {i+1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        fig, metrics = create_time_bound_sankey(start_date, end_date, exclude_blacklisted)
        if fig and metrics:
            weekly_metrics.append(metrics)
            weekly_figures.append(fig)

    # Create a summary DataFrame for easy comparison
    if weekly_metrics:
        summary_df = pd.DataFrame(weekly_metrics)

        # Calculate week-over-week changes
        for col in [
            'active_users', 'inactive_users', 'new_users', 'returning_users',
            'single_query_users', 'few_query_users', 'many_query_users',
            'registered_query_count', 'anonymous_query_count'
        ]:
            summary_df[f'{col}_wow_change'] = summary_df[col].pct_change(
                -1) * 100

        return weekly_figures, summary_df

    return [], pd.DataFrame()
