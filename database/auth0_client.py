"""
Auth0 client module for the Facticity dashboard.
Handles API calls and data processing for Auth0 data.
Used in email_segments.py which also saves data in data/auth0.
Used in metrics_view.py and overview_dashboard.py for user account creation/login counts. 
"""
import streamlit as st
import pandas as pd
import requests
import gzip
import io
import time
from datetime import datetime, timedelta
from auth0.authentication import GetToken
import os

from config import AUTH0_DOMAIN, CLIENT_ID, CLIENT_SECRET, API_AUDIENCE



def get_auth0_token():
    """
    Retrieve the Auth0 Management API token.
    
    Returns:
        str: Access token or None if failed
    """
    try:
        print(f"Attempting Auth0 authentication...")
        print(f"   Domain: {AUTH0_DOMAIN}")
        print(f"   Client ID: {CLIENT_ID}")
        print(f"   API Audience: {API_AUDIENCE}")
        
        # Check if required variables are set
        if not all([AUTH0_DOMAIN, CLIENT_ID, CLIENT_SECRET]):
            print("Missing Auth0 configuration variables")
            return None
        
        get_token = GetToken(AUTH0_DOMAIN, CLIENT_ID)
        token = get_token.client_credentials(
            CLIENT_ID, CLIENT_SECRET, API_AUDIENCE)
        
        access_token = token.get("access_token")
        if access_token:
            print(f"Auth0 token retrieved successfully!")
            return access_token
        else:
            print(f"No access token in response: {token}")
            return None
            
    except Exception as e:
        print(f"Failed to get Auth0 token: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            print(f"   Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'Unknown'}")
            print(f"   Response text: {e.response.text if hasattr(e.response, 'text') else 'Unknown'}")
        
        # DPreventing the error from being shown in the Streamlit UI
        print("Auth0 authentication failed - continuing without Auth0 features")
        return None


def export_all_users():
    """
    Exports all users from Auth0 as a CSV file.
    Fields must be specified individually for data to be included in export file. 
    If more than 30 fields are present, an error will be thrown by the API. Keep only required fields here.

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
            {"name": "email_verified"},
            {"name": "name"},
            {"name": "nickname"},
            {"name": "family_name"},
            {"name": "given_name"},
            {"name": "nickname"},
            {"name": "picture"},
            {"name": "created_at"},
            {"name": "updated_at"},
            {"name": "last_login"},
            {"name": "last_ip"},
            {"name": "logins_count"},
            {"name": "blocked_for"},
            {"name": "identities[0].provider"},
            {"name": "identities[0].user_id"},
            {"name": "identities[0].connection"},
            {"name": "identities[0].isSocial"},
            {"name": "idp_tenant_domain"},

            # User metadata fields
            {"name": "user_metadata.agreed"},
            {"name": "user_metadata.company"},
            {"name": "user_metadata.country"},
            {"name": "user_metadata.discovery"},
            {"name": "user_metadata.first_name"},
            {"name": "user_metadata.industry"},
            {"name": "user_metadata.intended_use"},
            {"name": "user_metadata.last_name"},
            {"name": "user_metadata.linkedin"},
            {"name": "user_metadata.occupation"},

            # App metadata if needed
            # {"name": "app_metadata.role"},
            # {"name": "app_metadata.permissions"},
        ]
    }
    try:
        response = requests.post(
            f"https://{AUTH0_DOMAIN}/api/v2/jobs/users-exports",
            headers=headers, 
            json=payload
        )
        if response.status_code == 400:
            st.error(f"Export failed: {response.json().get('message', 'Bad Request')}")
            return None
        response.raise_for_status()
        job_id = response.json().get("id")
        
        # Poll for job completion
        for _ in range(12):
            time.sleep(5)
            status_resp = requests.get(
                f"https://{AUTH0_DOMAIN}/api/v2/jobs/{job_id}",
                headers=headers
            )
            if status_resp.json().get("status") == "completed":
                return status_resp.json().get("location")
                
        raise Exception("Export timed out")
    except Exception as e:
        st.error(f"Export failed: {e}")
        return None


@st.cache_data(ttl=10800)
def get_auth0_user_list():
    """
    Retrieve Auth0 user list.
    
    Returns:
        DataFrame: DataFrame with Auth0 user data
    """
    file_url = export_all_users()
    if file_url:
        response = requests.get(file_url)
        with gzip.open(io.BytesIO(response.content), 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f, encoding='utf-8')
            
        if "email" in df.columns:
            df["email"] = df["email"].astype(str).str.lower().str.strip().str.replace("'", "", regex=False)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(
                df["created_at"], errors="coerce", utc=True)
        return df
    return pd.DataFrame()


def aggregate_auth0_daily(auth0_df, window_start, window_end):
    """
    Aggregates Auth0 new and cumulative users by day.
    
    Args:
        auth0_df: DataFrame with Auth0 data
        window_start: Start date in ISO format
        window_end: End date in ISO format
        
    Returns:
        DataFrame: DataFrame with daily new and total users
    """
    if auth0_df is None or auth0_df.empty:
        return pd.DataFrame(columns=['new_users', 'total_users'])
    
    df = auth0_df.copy().dropna(
        subset=["created_at"]).sort_values("created_at")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    
    # Group by day
    daily_new = df.groupby(pd.Grouper(
        key="created_at", freq="D")).size().rename("new_users")
    daily_total = daily_new.cumsum().rename("total_users")
    df_daily = pd.concat([daily_new, daily_total], axis=1)
    
    # Build full date range and reindex
    start = pd.to_datetime(window_start, utc=True)
    end = pd.to_datetime(window_end, utc=True)
    full_range = pd.date_range(
        start=start, end=end - timedelta(days=1), freq="D")
    df_daily = df_daily.reindex(full_range, fill_value=0)
    
    # Fill missing values
    df_daily["total_users"] = df_daily["total_users"].ffill().fillna(
        0).astype(int)
    df_daily["new_users"] = df_daily["new_users"].astype(int)
    
    # Convert index to string for plotting
    df_daily.index = df_daily.index.strftime("%Y-%m-%d")
    return df_daily


def aggregate_auth0_weekly(auth0_df, window_start, window_end):
    """
    Aggregates Auth0 new and cumulative users by week ending on Sunday.
    
    Args:
        auth0_df: DataFrame with Auth0 data
        window_start: Start date in ISO format
        window_end: End date in ISO format
        
    Returns:
        DataFrame: DataFrame with weekly new and total users
    """
    if auth0_df is None or auth0_df.empty:
        return pd.DataFrame(columns=['new_users', 'total_users'])
    
    df = auth0_df.copy().dropna(
        subset=["created_at"]).sort_values("created_at")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # Calculate end of week (Sunday) for each user
    df["end_of_week"] = (df["created_at"] +
                         pd.to_timedelta(
                             (6 - df["created_at"].dt.dayofweek) % 7, unit='d')
                         ).dt.normalize()

    # Group by the normalized Sunday dates
    weekly_new = df.groupby("end_of_week").size().rename("new_users")
    weekly_total = weekly_new.cumsum().rename("total_users")
    df_weekly = pd.concat([weekly_new, weekly_total], axis=1)

    # Generate a full range of Sundays
    start = pd.to_datetime(window_start, utc=True)
    end = pd.to_datetime(window_end, utc=True)
    first_sunday = (
        start + pd.DateOffset(days=(6 - start.dayofweek) % 7)).normalize()
    last_sunday = (
        end + pd.DateOffset(days=(6 - end.dayofweek) % 7)).normalize()
    full_range = pd.date_range(
        start=first_sunday, end=last_sunday, freq='W-SUN', tz='UTC')

    # Reindex and fill missing values
    df_weekly = df_weekly.reindex(full_range, fill_value=0)
    df_weekly["total_users"] = df_weekly["total_users"].ffill().fillna(
        0).astype(int)

    # Handle special case for the last value
    if df_weekly["total_users"].iloc[-1] == 0 and len(df_weekly) > 1:
        df_weekly["total_users"].iloc[-1] = df_weekly["total_users"].iloc[-2]

    # Convert index to string format
    df_weekly.index = df_weekly.index.strftime("%Y-%m-%d")
    return df_weekly


def aggregate_auth0_monthly(auth0_df, window_start, window_end):
    """
    Aggregates Auth0 new and cumulative users by month.
    
    Args:
        auth0_df: DataFrame with Auth0 data
        window_start: Start date in ISO format
        window_end: End date in ISO format
        
    Returns:
        DataFrame: DataFrame with monthly new and total users
    """
    if auth0_df is None or auth0_df.empty:
        return pd.DataFrame(columns=['new_users', 'total_users'])
    
    df = auth0_df.copy().dropna(
        subset=["created_at"]).sort_values("created_at")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    
    # Group by month
    monthly_new = df.groupby(pd.Grouper(
        key="created_at", freq="M")).size().rename("new_users")
    monthly_total = monthly_new.cumsum().rename("total_users")
    df_monthly = pd.concat([monthly_new, monthly_total], axis=1)

    # Build full date range
    start = pd.to_datetime(window_start, utc=True)
    end = pd.to_datetime(window_end, utc=True)
    full_range = pd.date_range(start=start, end=(
        end + pd.offsets.MonthEnd(0)), freq="M")
    
    # Reindex and fill missing values
    df_monthly = df_monthly.reindex(full_range, fill_value=0)
    df_monthly["total_users"] = df_monthly["total_users"].ffill().fillna(
        0).astype(int)
    df_monthly["new_users"] = df_monthly["new_users"].astype(int)

    # Format index as MM/YYYY
    df_monthly.index = df_monthly.index.strftime("%m/%Y")
    return df_monthly


def aggregate_auth0_quarterly(auth0_df, window_start, window_end):
    """
    Derives quarterly new and cumulative user counts with custom quarters.
    
    Args:
        auth0_df: DataFrame with Auth0 data
        window_start: Start date in ISO format
        window_end: End date in ISO format
        
    Returns:
        DataFrame: DataFrame with quarterly new and total users
    """
    if auth0_df is None or auth0_df.empty:
        return pd.DataFrame(columns=['new_users', 'total_users'])
    
    from utils.date_utils import quarter_sort_key
    
    # Prepare monthly data
    df = auth0_df.copy().dropna(
        subset=["created_at"]).sort_values("created_at")
    df["created_at"] = pd.to_datetime(
        df["created_at"], utc=True)
    
    monthly_new = df.groupby(pd.Grouper(
        key="created_at", freq="M")).size().rename("new_users")
    monthly_total = monthly_new.cumsum().rename("total_users")
    df_monthly = pd.concat([monthly_new, monthly_total], axis=1)

    # Ensure consistent timezone
    start_dt = pd.to_datetime(window_start, utc=True)
    end_dt = pd.to_datetime(window_end, utc=True)

    # Determine the full date range
    if df.empty:
        earliest_dt = start_dt
    else:
        earliest_dt = df["created_at"].min().to_period('M').to_timestamp()

    # Ensure earliest_dt is in UTC
    earliest_dt = earliest_dt.tz_localize(None).tz_localize("UTC")

    # Create full date range
    full_range = pd.date_range(
        start=earliest_dt,
        end=end_dt + pd.offsets.MonthEnd(0),
        freq="M",
        tz="UTC"
    )
    
    # Reindex and fill missing values
    df_monthly = df_monthly.reindex(full_range, fill_value=0)
    df_monthly["total_users"] = df_monthly["total_users"].ffill().fillna(
        0).astype(int)
    df_monthly["new_users"] = df_monthly["new_users"].astype(int)

    # Define quarter labeling function
    def get_quarter_label(dt):
        threshold = pd.Timestamp("2024-06-01", tz="UTC")
        if dt < threshold:
            return "June-Aug 2024"
        month = dt.month
        year = dt.year
        if month in [6, 7, 8]:
            return f"June-Aug {year}"
        elif month in [9, 10, 11]:
            return f"Sep-Nov {year}"
        elif month == 12:
            return f"Dec {year}-Feb {year+1}"
        elif month in [1, 2]:
            return f"Dec {year-1}-Feb {year}"
        elif month in [3, 4, 5]:
            return f"Mar-May {year}"
        elif month in [6, 7, 8]:
            return f"June-Aug {year}"
        elif month in [9, 10, 11]:
            return f"Sep-Nov {year}"
        elif month == 12:
            return f"Dec {year}-Feb {year+1}"
        return None

    # Assign quarters and filter out months without a valid label
    df_monthly["quarter"] = df_monthly.index.map(get_quarter_label)
    df_monthly = df_monthly[df_monthly["quarter"].notna()]

    # Group by quarter and sum new_users
    quarterly_new = df_monthly.groupby("quarter")["new_users"].sum()

    # Generate expected quarters based on full_range to maintain order
    expected_quarters = sorted(
        {get_quarter_label(dt)
         for dt in full_range if get_quarter_label(dt) is not None},
        key=quarter_sort_key
    )
    quarterly_new = quarterly_new.reindex(expected_quarters, fill_value=0)

    # Calculate cumulative total_users correctly
    quarterly_total = []
    cumulative_total = 0
    for quarter in expected_quarters:
        cumulative_total += quarterly_new[quarter]
        quarterly_total.append(cumulative_total)

    # Create final DataFrame
    df_quarterly = pd.DataFrame({
        "new_users": quarterly_new,
        "total_users": quarterly_total
    }, index=expected_quarters)

    return df_quarterly
