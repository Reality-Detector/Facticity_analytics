"""
SQL query for posthog to get user session data.
Used in user flow/fixed_sankey_view.py tab to estimate proportion of logged in/non-logged in users.
"""

import requests
import pandas as pd
from datetime import datetime
from config import POSTHOG_API_KEY, POSTHOG_PROJECT_ID


def get_posthog_data(start_date, end_date):
    """
    Query PostHog for session data within a specific date range.
    
    Args:
        start_date: Start date as datetime object
        end_date: End date as datetime object
        
    Returns:
        DataFrame: DataFrame with PostHog session data
    """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    posthog_query = f"""
    WITH sessions_data AS (
        SELECT
            properties.$session_id AS session_id,
            MIN(timestamp) AS session_start,
            groupArray(DISTINCT properties.$ip) AS ip_addresses,
            groupArray(DISTINCT properties.$geoip_country_name)[1] AS country,
            groupArray(DISTINCT properties.$referrer)[1] AS referrer
        FROM events
        WHERE timestamp >= toDateTime('{start_date_str}') AND timestamp < toDateTime('{end_date_str}')
        GROUP BY properties.$session_id
    )
    SELECT
        session_id,
        ip_addresses,
        country,
        session_start,
        referrer
    FROM sessions_data
    ORDER BY session_start DESC
    LIMIT 10000
    """

    headers = {
        "Authorization": f"Bearer {POSTHOG_API_KEY}",
        "Content-Type": "application/json"
    }
    posthog_url = f"https://app.posthog.com/api/projects/{POSTHOG_PROJECT_ID}/query/"
    payload = {
        "query": {
            "kind": "HogQLQuery",
            "query": posthog_query
        },
        "refresh": "force_blocking"
    }

    response = requests.post(posthog_url, json=payload, headers=headers)
    if response.status_code == 200:
        ph_data = response.json()
    else:
        print(f"PostHog API Error: {response.status_code} {response.text}")
        return pd.DataFrame()

    # Create DataFrame with the results
    if not ph_data.get('results'):
        print("No data fetched from PostHog.")
        return pd.DataFrame()

    posthog_df = pd.DataFrame(ph_data['results'], columns=[
        'session_id', 'ip_addresses', 'country', 'session_start', 'referrer'
    ])

    # Clean up ip_addresses
    posthog_df['ip_addresses'] = posthog_df['ip_addresses'].apply(
        lambda ips: [ip.replace("'", "")
                     for ip in ips] if isinstance(ips, list) else ips
    )

    # Ensure session_start is datetime
    posthog_df["session_start"] = pd.to_datetime(posthog_df["session_start"])

    # Get the earliest fetched session date
    earliest_date = posthog_df["session_start"].min()
    print(f"Earliest fetched session date: {earliest_date}")

    # Check if we hit the limit of 10,000 rows
    if len(posthog_df) == 10000:
        print(
            "Warning: Query returned 10,000 sessions. Consider increasing the query limit.")

    return posthog_df
