"""
Overview dashboard view showing user activity metrics and visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict

from database.mongo_client import (
    get_db_connection,
    get_mongo_active_users,
    aggregate_daily_with_users,
    aggregate_weekly_with_users,
    aggregate_monthly_with_users,
    is_valid_user,
    fetch_all_emails_with_timestamps,
    aggregate_daily_by_url,
    aggregate_daily_users_by_url
)
from database.auth0_client import get_auth0_user_list
from services.analytics import get_active_users

from utils.chart_utils import generate_chart, generate_url_breakdown_chart, generate_user_breakdown_chart

from config import PRIMARY_BLUE, LIGHT_BLUE, MODERN_ORANGE, API_DB_CONNECTION_STRING, green_1


def show_overview_view():
    """
    Display the Overview Dashboard with user metrics and visualizations.
    """
    # Sidebar options
    with st.sidebar:
        exclude_blacklisted = st.checkbox("Exclude Internal/Blacklist Users", value=True,
                                          help="Filter out emails in the blacklist and from blacklisted domains")

    # Calculate current time
    now = datetime.now(timezone.utc)

    # Define date ranges for all charts
    now_ts = pd.Timestamp.now(tz='UTC')
    daily_start = (now_ts - timedelta(days=14)).isoformat()
    daily_end = now_ts.isoformat()

    # Get active users for different time periods
    with st.spinner("Loading dashboard data..."):
        daily_active_users = get_active_users(
            "daily", exclude_blacklisted=exclude_blacklisted)
        weekly_active_users = get_active_users(
            "weekly", exclude_blacklisted=exclude_blacklisted)
        monthly_active_users = get_active_users(
            "monthly", exclude_blacklisted=exclude_blacklisted)

        # Get Auth0 user data
        auth0_df = get_auth0_user_list()
        auth0_df["created_at"] = pd.to_datetime(
            auth0_df["created_at"], errors="coerce", utc=True)

        # Get the total number of registered users
        total_users = len(auth0_df)

        # Count new users in the last 30 days
        thirty_days_ago = now - timedelta(days=30)
        new_users_30d = len(
            auth0_df[auth0_df["created_at"] >= thirty_days_ago])

        thirty_days_ago = now - timedelta(days=30)
        sixty_days_ago = now - timedelta(days=60)

        current_users = len(
            auth0_df[auth0_df["created_at"] <= thirty_days_ago])
        previous_users = len(
            auth0_df[auth0_df["created_at"] <= sixty_days_ago])

        growth_rate = (
            (current_users - previous_users) / previous_users) * 100

    # Title and key metrics in a single row
    st.write("Overview")

    # Top metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("DAU", len(daily_active_users))
    with col2:
        st.metric("WAU", len(weekly_active_users))
    with col3:
        st.metric("MAU", len(monthly_active_users))
    with col4:
        st.metric("Total Users", total_users)
    with col5:
        st.metric("New Users (30d)", new_users_30d)
    with col6:
        st.metric("Monthly Growth", f"{growth_rate:.1f}%")

    # Main content in two columns (wider left column for the charts)
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Get URL breakdown data for queries by URL
        breakdown = aggregate_daily_by_url(
            daily_start, daily_end, exclude_blacklisted)

        # Get API query data
        daily_results_api = aggregate_daily_with_users(
            daily_start, daily_end, exclude_blacklisted, db_string=API_DB_CONNECTION_STRING)

        # Create a dictionary of API query counts by date
        api_counts = {}
        if daily_results_api:
            for doc in daily_results_api:
                api_counts[doc["_id"]] = doc["query_count"]

        # Extract sorted dates
        dates = [row["_id"] for row in breakdown]

        # Normalize URLs: remove trailing slashes and map them together
        def normalize_url(url):
            return url.rstrip("/") if url else "writer"

        # Build normalized URL counts per day
        normalized_breakdown = []
        for row in breakdown:
            normalized_urls = defaultdict(int)
            for url, count in row["urls"].items():
                norm_url = normalize_url(url)
                normalized_urls[norm_url] += count

            # Add API counts as a new category/URL
            date = row["_id"]
            if date in api_counts:
                normalized_urls["API"] = api_counts[date]

            normalized_breakdown.append({
                "_id": date,
                "urls": dict(normalized_urls)
            })

        # Extract sorted dates
        dates = [row["_id"] for row in normalized_breakdown]

        # Collect all unique normalized URLs
        all_urls = sorted(
            {url for row in normalized_breakdown for url in row["urls"].keys()})

        # Build per-URL series with zero fill
        url_counts = {
            url: [row["urls"].get(url, 0) for row in normalized_breakdown]
            for url in all_urls
        }

        # Custom color map to ensure API is GREEN_1
        color_map = {"API": green_1, "chrome-extension://mlackneplpmmomaobipjjpebhgcgmocp/sidebar.html":MODERN_ORANGE, "https://app.facticity.ai":LIGHT_BLUE, "writer": PRIMARY_BLUE}

        # Generate the chart with custom colors
        generate_url_breakdown_chart(
            title="Daily Queries by URL (Last 14 Days)",
            dates=dates,
            url_counts=url_counts,
            color_map=color_map  # Pass custom color map
        )

        # Daily Active Users by URL Chart
        user_breakdown = aggregate_daily_users_by_url(
            daily_start, daily_end, exclude_blacklisted)

        if user_breakdown:
            # Extract sorted dates
            user_dates = [row["_id"] for row in user_breakdown]

            # Build normalized URL user counts per day
            normalized_user_breakdown = []
            for row in user_breakdown:
                normalized_urls = defaultdict(int)
                for url, count in row["urls"].items():
                    norm_url = normalize_url(url)
                    normalized_urls[norm_url] += count
                normalized_user_breakdown.append({
                    "_id": row["_id"],
                    "urls": dict(normalized_urls)
                })

            # Extract sorted dates
            user_dates = [row["_id"] for row in normalized_user_breakdown]

            # Collect all unique normalized URLs
            all_user_urls = sorted(
                {url for row in normalized_user_breakdown for url in row["urls"].keys()})

            # Build per-URL series with zero fill
            url_users = {
                url: [row["urls"].get(url, 0)
                      for row in normalized_user_breakdown]
                for url in all_user_urls
            }

            # Generate the chart
            generate_user_breakdown_chart(
                title="Daily Active Users by URL (Last 14 Days)",
                dates=user_dates,
                url_users=url_users,
                color_map=color_map
            )
        else:
            st.write("No daily active users by URL data available.")

    with right_col:
        # User activity breakdown
        st.subheader("User Breakdown")

        # Define time ranges for analysis
        time_ranges = {
            "Last 7 days": now - timedelta(days=7),
            "Last 30 days": now - timedelta(days=30),
            "Last 90 days": now - timedelta(days=90)
        }

        # Create data for visualization
        user_data = []

        for label, cutoff in time_ranges.items():
            # Get active users in this time range
            active_users = get_mongo_active_users(cutoff, exclude_blacklisted)

            # Get new users (created after cutoff)
            new_users = auth0_df[auth0_df["created_at"]
                                 >= cutoff]["email"].tolist()
            new_users = [email for email in new_users
                         if (not exclude_blacklisted or is_valid_user(email))]

            # Calculate new vs existing active users
            active_new = len(set(active_users).intersection(new_users))
            active_existing = len(active_users) - active_new

            # Add to data
            user_data.append({
                "time_range": label,
                "New Users": active_new,
                "Existing Users": active_existing
            })

        # Create dataframe
        user_df = pd.DataFrame(user_data)

        # Create stacked bar chart
        fig_breakdown = px.bar(
            user_df,
            x="time_range",
            y=["New Users", "Existing Users"],
            title=None,
            labels={"time_range": "Time Range", "value": "Users"},
            color_discrete_map={"New Users": PRIMARY_BLUE,
                                "Existing Users": MODERN_ORANGE}
        )

        fig_breakdown.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=230,
            legend_title=None,
            xaxis_title=None,
            yaxis_title=None
        )

        st.plotly_chart(fig_breakdown, use_container_width=True)

        # Cohort Retention Heatmap
        st.subheader("Cohort Retention")

        # Get raw query data with timestamps and emails
        with st.spinner("Generating cohort analysis..."):
            collection = get_db_connection()

            # Get data for last 6 months
            start_date = (now - timedelta(days=180)).replace(day=1).replace(hour=0,
                                                                            minute=0, second=0, microsecond=0)
            start_iso = start_date.isoformat()
            end_iso = now.isoformat()

            # Fetch all emails with timestamps
            email_data = fetch_all_emails_with_timestamps(collection)

            if not email_data.empty:
                # Convert timestamps to datetime
                email_data["timestamp"] = pd.to_datetime(
                    email_data["timestamp"], errors="coerce")

                # Filter by date range
                email_data = email_data[
                    (email_data["timestamp"] >= start_date) &
                    (email_data["timestamp"] <= now)
                ]

                # Add month column
                email_data["month"] = email_data["timestamp"].dt.strftime(
                    "%Y-%m")

                # Process cohort data
                user_cohorts = {}
                user_monthly_activity = {}

                for _, row in email_data.iterrows():
                    email = row["userEmail"]
                    month = row["month"]

                    if exclude_blacklisted:
                        if not is_valid_user(email):
                            continue

                    # Determine cohort (first month user was seen)
                    if email not in user_cohorts:
                        user_cohorts[email] = month

                    # Track monthly activity
                    if email not in user_monthly_activity:
                        user_monthly_activity[email] = set()
                    user_monthly_activity[email].add(month)

                # Create cohort analysis
                if user_cohorts:
                    # Get all months in period
                    all_months = sorted(set(email_data["month"]))

                    # Create cohort retention matrix
                    cohort_sizes = {}
                    retention_matrix = {}

                    for email, cohort in user_cohorts.items():
                        if cohort not in cohort_sizes:
                            cohort_sizes[cohort] = 0
                        cohort_sizes[cohort] += 1

                        # For each month after cohort, check if user was active
                        if email in user_monthly_activity:
                            active_months = user_monthly_activity[email]
                            for month in all_months:
                                if month >= cohort:  # Only count months on or after cohort
                                    month_idx = all_months.index(
                                        month) - all_months.index(cohort)
                                    key = (cohort, month_idx)

                                    if key not in retention_matrix:
                                        retention_matrix[key] = 0

                                    if month in active_months:
                                        retention_matrix[key] += 1

                    # Convert to percentage and create heatmap data
                    heatmap_data = []
                    cohorts = sorted(cohort_sizes.keys())

                    # Limit to recent cohorts with sufficient data
                    if len(cohorts) > 6:
                        cohorts = cohorts[-6:]

                    for cohort in cohorts:
                        cohort_size = cohort_sizes[cohort]
                        if cohort_size == 0:
                            continue

                        row = {"cohort": cohort, "size": cohort_size}

                        for i in range(len(all_months)):
                            if all_months[i] >= cohort:
                                month_idx = i - all_months.index(cohort)
                                key = (cohort, month_idx)
                                retention = retention_matrix.get(
                                    key, 0) / cohort_size * 100
                                row[f"month{month_idx}"] = retention

                        heatmap_data.append(row)

                    # Create heatmap dataframe
                    if heatmap_data:
                        df = pd.DataFrame(heatmap_data)

                        # Get month columns
                        month_cols = [
                            col for col in df.columns if col.startswith('month')]

                        # Limit to first 4 months to save space
                        if len(month_cols) > 4:
                            month_cols = month_cols[:4]

                        # Filter dataframe to selected columns
                        df_display = df[['cohort', 'size'] + month_cols]

                        # Prepare data for heatmap
                        z_data = df_display[month_cols].values
                        x_labels = [f'M{i}' for i in range(len(month_cols))]
                        y_labels = [f"{c} ({s})" for c, s in zip(
                            df_display['cohort'], df_display['size'])]

                        # Create text annotations (percentage values)
                        text = [[f"{val:.1f}%" for val in row]
                                for row in z_data]

                        # Create heatmap
                        fig_retention = go.Figure(data=go.Heatmap(
                            z=z_data,
                            x=x_labels,
                            y=y_labels,
                            text=text,
                            texttemplate="%{text}",
                            colorscale="Blues",
                            hoverongaps=False
                        ))

                        fig_retention.update_layout(
                            margin=dict(l=20, r=20, t=10, b=20),
                            height=230,
                            xaxis_title=None,
                            yaxis_title=None,
                            # To have the most recent cohort at the top
                            yaxis=dict(autorange="reversed")
                        )

                        st.plotly_chart(
                            fig_retention, use_container_width=True)

                        # Add explanation text
                        st.caption(
                            "M0 = First month, M1 = Second month, etc. Percentages show retained users.")
                    else:
                        st.info("Insufficient data for cohort analysis.")
                else:
                    st.info("No user activity data available.")
            else:
                st.info("No data available for retention analysis.")
