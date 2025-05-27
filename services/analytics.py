"""
Analytics to segment activity level and is presented in emaiL_segments.py
"""
import pandas as pd
from datetime import datetime, timedelta, timezone

from database.mongo_client import get_db_connection, get_mongo_active_users, is_valid_user
from database.auth0_client import get_auth0_user_list


def normalize_timestamps(df):
    """
    Normalizes timestamps in a DataFrame.
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        DataFrame: DataFrame with normalized timestamps
    """
    def convert_timestamp(ts):
        try:
            if isinstance(ts, (int, float)):
                return pd.to_datetime(ts, unit='ms', utc=True)
            return pd.to_datetime(ts, utc=True, errors='coerce')
        except Exception:
            return pd.NaT
    df["normalizedTimestamp"] = df["timestamp"].apply(convert_timestamp)
    return df[df["normalizedTimestamp"].notna()]


def convert_timestamp_engagement(ts):
    """
    Converts timestamps for engagement analysis, handling different formats.
    
    Args:
        ts: Timestamp in various possible formats
        
    Returns:
        Timestamp: Normalized pandas timestamp or NaT if conversion fails
    """
    try:
        if isinstance(ts, (int, float)):
            if ts > 1e10:
                return pd.to_datetime(ts, unit='ms', utc=True)
            else:
                return pd.to_datetime(ts, unit='s', utc=True)
        return pd.to_datetime(ts, utc=True, errors='coerce')
    except Exception:
        return pd.NaT


def filter_emails_exclusively(df, now, exclude_blacklisted=True):
    """
    Filter emails based on their last activity time period.
    
    Args:
        df: DataFrame with userEmail and timestamp columns
        now: Current datetime
        exclude_blacklisted: Whether to exclude blacklisted users (default: True)
        
    Returns:
        DataFrame: Categorized user emails by activity period
    """
    df = normalize_timestamps(df)
    df["daysAgo"] = (now - df["normalizedTimestamp"]).dt.days
    df_group = df.groupby("userEmail", as_index=False)["daysAgo"].min()
    df_group["userEmail"] = df_group["userEmail"].astype(
        str).str.lower().str.strip()

    # Filter blacklisted users if requested
    if exclude_blacklisted:
        df_group = df_group[df_group["userEmail"].apply(is_valid_user)]

    # Define time-based categories
    conditions = [
        (df_group["daysAgo"] <= 7),
        (df_group["daysAgo"] >= 8) & (df_group["daysAgo"] <= 30),
        (df_group["daysAgo"] >= 31) & (df_group["daysAgo"] <= 90),
        (df_group["daysAgo"] >= 181)
    ]
    labels = [
        "Last 7 days",
        "Last 30 days",
        "Last 90 days",
        "More than 6 months ago"
    ]
    df_group["Category"] = None
    for condition, label in zip(conditions, labels):
        df_group.loc[condition, "Category"] = label

    # Left merge with Auth0 user list to include all users
    auth0_df = get_auth0_user_list()
    if not auth0_df.empty and "email" in auth0_df.columns:
        auth0_df["email"] = auth0_df["email"].astype(
            str).str.replace("'", "").str.lower().str.strip()

        # Filter Auth0 user list if requested
        if exclude_blacklisted:
            auth0_df = auth0_df[auth0_df["email"].apply(is_valid_user)]

        base_df = auth0_df[["email"]].drop_duplicates().rename(
            columns={"email": "userEmail"})
        merged = base_df.merge(df_group, on="userEmail", how="left")
        merged["Category"] = merged["Category"].fillna("No Activity")
        return merged
    else:
        df_group["Category"] = df_group["Category"].fillna("No Activity")
        return df_group


def classify_engagement_with_pandas(df, now, exclude_blacklisted=True):
    """
    Classifies user engagement levels based on activity patterns.
    
    Args:
        df: DataFrame with userEmail and timestamp columns
        now: Current datetime
        exclude_blacklisted: Whether to exclude blacklisted users (default: True)
        
    Returns:
        DataFrame: User engagement classification
    """
    df["normalizedTimestamp"] = df["timestamp"].apply(
        convert_timestamp_engagement)
    df = df[df["normalizedTimestamp"].notna()]
    df["daysAgo"] = (now - df["normalizedTimestamp"]).dt.days

    records = []
    for email, group in df.groupby("userEmail"):
        email_lower = str(email).lower().strip()

        # Skip blacklisted users if requested
        if exclude_blacklisted and not is_valid_user(email_lower):
            continue

        query_count = len(group)
        latest_activity = group["daysAgo"].min()

        # Assign engagement category
        category = None
        if latest_activity <= 7 and query_count >= 3:
            category = "Highly Engaged: 3+ in last 7 days"
        elif latest_activity <= 30 and query_count >= 1:
            category = "Moderately Engaged: 1+ in last 30 days"
        elif latest_activity <= 90 and query_count >= 1:
            category = "Low Engagement: 1+ in last 90 days"
        elif latest_activity > 180:
            category = "Inactive: nothing in last 180 days"

        if category:
            records.append({"userEmail": email_lower, "Engagement": category})

    engagement_df = pd.DataFrame(records)

    # Left merge with Auth0 user list
    auth0_df = get_auth0_user_list()
    if not auth0_df.empty and "email" in auth0_df.columns:
        auth0_df["email"] = auth0_df["email"].astype(
            str).str.replace("'", "").str.lower().str.strip()

        # Filter Auth0 user list if requested
        if exclude_blacklisted:
            auth0_df = auth0_df[auth0_df["email"].apply(is_valid_user)]

        base_df = auth0_df[["email"]].drop_duplicates().rename(
            columns={"email": "userEmail"})
        merged = base_df.merge(engagement_df, on="userEmail", how="left")
        merged["Engagement"] = merged["Engagement"].fillna("No Activity")
        return merged
    else:
        engagement_df["Engagement"] = engagement_df["Engagement"].fillna(
            "No Activity")
        return engagement_df



def calculate_retention_rate(timeframe, exclude_blacklisted):
    """
    Calculate the retention rate for the specified timeframe.
    
    Args:
        timeframe: "Daily", "Weekly", or "Monthly"
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        tuple: (retention_rate, period_start, period_end)
    """
    from datetime import datetime, timedelta, timezone
    from database.mongo_client import get_db_connection, is_valid_user
    
    # Get MongoDB collection
    collection = get_db_connection()
    
    # Calculate current time and period boundaries
    now = datetime.now(timezone.utc)
    
    if timeframe == "Daily":
        current_period_start = now - timedelta(days=1)
        previous_period_start = current_period_start - timedelta(days=1)
    elif timeframe == "Weekly":
        current_period_start = now - timedelta(days=7)
        previous_period_start = current_period_start - timedelta(days=7)
    else:  # Monthly
        current_period_start = now - timedelta(days=30)
        previous_period_start = current_period_start - timedelta(days=30)
    
    # Convert to ISO format for MongoDB queries
    current_period_iso = current_period_start.isoformat()
    previous_period_iso = previous_period_start.isoformat()
    now_iso = now.isoformat()
    
    # Get users in current period
    current_query = {
        "timestamp": {"$gte": current_period_iso, "$lt": now_iso},
        "userEmail": {"$ne": None}
    }
    
    current_users = set()
    for doc in collection.find(current_query, {"userEmail": 1}):
        email = doc.get("userEmail", "").lower().strip()
        if email and (not exclude_blacklisted or is_valid_user(email)):
            current_users.add(email)
    
    # If no users in current period, return None
    if not current_users:
        return None, previous_period_start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")
    
    # Get users in previous period
    previous_query = {
        "timestamp": {"$gte": previous_period_iso, "$lt": current_period_iso},
        "userEmail": {"$ne": None}
    }
    
    previous_users = set()
    for doc in collection.find(previous_query, {"userEmail": 1}):
        email = doc.get("userEmail", "").lower().strip()
        if email and (not exclude_blacklisted or is_valid_user(email)):
            previous_users.add(email)
    
    # Calculate retention rate (% of current users who were also active in previous period)
    retained_users = current_users.intersection(previous_users)
    retention_rate = (len(retained_users) / len(current_users)) * 100 if current_users else 0
    
    return retention_rate, previous_period_start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def get_active_users(timeframe="daily", exclude_blacklisted=True):
    """
    Get active users for the specified timeframe.
    
    Args:
        timeframe: "daily", "weekly", or "monthly"
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        set: Set of active user emails
    """
    from datetime import datetime, timedelta, timezone
    from database.mongo_client import get_mongo_active_users
    
    # Calculate current time and cutoff based on timeframe
    now = datetime.now(timezone.utc)
    
    if timeframe == "daily":
        cutoff = now - timedelta(days=1)
    elif timeframe == "weekly":
        cutoff = now - timedelta(days=7)
    else:  # monthly
        cutoff = now - timedelta(days=30)
    
    # Get active users after cutoff
    return get_mongo_active_users(cutoff, exclude_blacklisted)