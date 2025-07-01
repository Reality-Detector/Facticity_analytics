"""
Interactive user engagement view for the Facticity dashboard.
"""
import streamlit as st
import os
import json
import plotly.express as px
from streamlit_plotly_events import plotly_events
from datetime import datetime, timedelta, timezone
import pandas as pd
import re

from database.mongo_client import get_db_connection, get_distribution_data, is_valid_user, fetch_query_data, aggregate_daily_with_users
from utils.user_profile_utils import (
    create_bedrock_client, load_iab_categories, generate_category_embeddings,
    load_query_embeddings, save_query_embeddings, fetch_user_queries_with_date_range,
    categorize_queries, generate_user_profiles, generate_query_embeddings_batch
)

from config import AUTH0_IP_LOOKUP_FILEPATH
DATA_FOLDER = "data/email_segments"
USER_PROFILES_FILE = os.path.join(DATA_FOLDER, "user_profiles.json")


def is_video_transcript(text: str) -> bool:
    has_timestamps = bool(re.search(r"\d{1,2}:\d{2}", text))
    has_speakers = bool(
        re.search(r"(Speaker \d+|Narrator|Host):", text, re.IGNORECASE))
    return has_timestamps or has_speakers

def is_url(string):
    url_pattern = re.compile(
        r'^(https?:\/\/)?'              # http:// or https:// (optional)
        r'([\da-z\.-]+)\.([a-z\.]{2,6})'  # domain name
        r'([\/\w \.-]*)*\/?$'           # optional path
    )
    return bool(url_pattern.match(string))


def load_user_profiles():
    """Load user profiles data from JSON file."""
    try:
        # Try current directory first, then parent directory
        with open(USER_PROFILES_FILE, 'r') as f:
            data = json.load(f)
        st.success(
            f"Loaded {len(data)} user profiles from {USER_PROFILES_FILE}")
        return data
    except Exception as e:
        st.error(f"Error loading user profiles: {e}")
        return {}

profile_data = load_user_profiles()
user_emails = list(profile_data.keys())


def show_interactive_engagement_view():
    """
    Display the Interactive User Engagement view.
    """
    
    st.title("Usage Distribution")

    # Sidebar options
    with st.sidebar:
        exclude_blacklisted = st.checkbox("Exclude Internal/Blacklist Users", value=True,
                                          help="Filter out emails in the blacklist and from blacklisted domains")


    # Time period selection
    days = st.slider("Select Time Period (in days)",
                    min_value=1, max_value=180, value=7)

    # Calculate date range based on selected days
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    selected_period = f"Last {days} Day{'s' if days > 1 else ''}"
    start_date = datetime.utcnow() - timedelta(days=days)

    # Engagement threshold adjustment
    threshold = st.slider(
        "Minimum Number of Queries",
        min_value=1,
        max_value=100,
        value=10
    )

    # Build aggregation pipeline for engagement distribution
    pipeline = [
        {"$match": {"timestamp": {"$gte": start_date.isoformat()}}},
        {"$group": {"_id": "$userEmail", "query_count": {"$sum": 1}}},
    ]

    # Get distribution data
    with st.spinner("Analyzing user engagement..."):
        collection = get_db_connection()
        result = list(collection.aggregate(pipeline))

        if not result:
            st.warning("No data available for the selected period.")
            return

        # Process results in Python to handle blacklist filtering
        user_query_counts = []
        for item in result:
            email = item["_id"]
            count = item["query_count"]

            # Add user to the list if they should be included based on blacklist settings
            if not exclude_blacklisted or is_valid_user(email):
                user_query_counts.append(
                    {"userEmail": email, "query_count": count})

        if not user_query_counts:
            st.warning("No valid users found for the selected period.")
            return

        # Calculate metrics
        total_users = len(user_query_counts)
        users_above_threshold = sum(
            1 for user in user_query_counts if user["query_count"] > threshold)
        percentage = (users_above_threshold / total_users *
                      100) if total_users > 0 else 0

        # Display engagement metrics
        st.metric(
            f"High-Engagement Users (> {threshold} queries)",
            f"{percentage:.1f}% of total users"
        )
        st.write("Click on a bar to see user list")

        # Create histogram buckets
        buckets = {}
        for i in range(0, 100, 5):
            buckets[f"{i}-{i+4}"] = 0
        buckets["100+"] = 0

        # Fill the buckets
        for user in user_query_counts:
            count = user["query_count"]
            if count >= 100:
                buckets["100+"] += 1
            else:
                bucket_key = f"{(count // 5) * 5}-{(count // 5) * 5 + 4}"
                buckets[bucket_key] += 1

        # Build histogram data for Plotly
        bucket_labels = list(buckets.keys())
        bucket_counts = list(buckets.values())

        # Create interactive bar chart
        fig = px.bar(
            x=bucket_labels,
            y=bucket_counts,
            labels={'x': 'Query Count Range', 'y': 'Number of Users'},
            title=f"User Engagement Distribution ({selected_period})"
        )

        # Capture click events on the chart
        clicked_points = plotly_events(
            fig, click_event=True, hover_event=False)

        # Handle bar click event
        if clicked_points:
            clicked_label = clicked_points[0]['x']
            st.write(f"### Users in bucket: {clicked_label}")

            # Filter users based on the clicked bucket label
            if clicked_label == "100+":
                filtered_users = [
                    user for user in user_query_counts if user["query_count"] >= 100]
            else:
                lower_str, upper_str = clicked_label.split("-")
                lower, upper = int(lower_str), int(upper_str)
                filtered_users = [user for user in user_query_counts
                                  if lower <= user["query_count"] <= upper]

            if filtered_users:
                st.dataframe(pd.DataFrame(filtered_users))
            else:
                st.write("No users found in this bucket.")
        #User Engagement Category Distribution
        # Load activity data again with timestamps
        st.subheader("User Engagement Category Distribution")

        pipeline_full = [
            {"$match": {"timestamp": {"$gte": start_date.isoformat()}}},
            {"$project": {"userEmail": 1, "timestamp": 1}},
        ]
        activity_data = list(collection.aggregate(pipeline_full))

        if not activity_data:
            st.warning("No user activity data found.")
            st.stop()

        df = pd.DataFrame(activity_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if exclude_blacklisted:
            df = df[df["userEmail"].apply(is_valid_user)]


        def categorize_user_activity(df):
            now = datetime.now(timezone.utc)
            user_categories = []

            for email, group in df.groupby("userEmail"):
                timestamps = sorted(group["timestamp"].tolist())
                days_ago = [(now - ts).days for ts in timestamps]
                total = len(timestamps)
                latest = min(days_ago)

                query = ""
                if email and email in profile_data:
                    user_data = profile_data[email]
                    query = user_data.get("recent_queries", [""])[0]

                weekdays = [ts.weekday() for ts in timestamps]
                hours = [ts.hour for ts in timestamps]
                only_weekends = all(day in [5, 6] for day in weekdays)
                late_night_count = sum(1 for hour in hours if hour >= 20 or hour < 2)
                late_night_ratio = late_night_count / total if total > 0 else 0
                is_late_night_user = late_night_ratio > 0.7

                categories = []

                if only_weekends:
                    categories.append("Weekend-only users")
                if is_late_night_user:
                    categories.append("Late-night users (8pm–2am)")
                if total == 1:
                    categories.append("Completed First Fact-Check")
                if total >= 5 and max(days_ago) <= 7:
                    categories.append("Completed 5 Fact-Checks in a Week")
                if total >= 3:
                    sorted_dates = sorted(ts.date() for ts in timestamps)
                    for i in range(len(sorted_dates) - 2):
                        if (sorted_dates[i+1] - sorted_dates[i]).days == 1 and \
                        (sorted_dates[i+2] - sorted_dates[i+1]).days == 1:
                            categories.append("Active 3 Consecutive Days")
                            break
                if latest > 7 and any(d <= 7 for d in days_ago):
                    categories.append("Reactivated After 7+ Days")
                if latest >= 7:
                    categories.append("Inactive for 7+ Days")
                elif latest >= 3:
                    categories.append("Inactive for 3 Days")
                if total <= 5:
                    categories.append("1–5 Fact-Checks")
                elif total <= 20:
                    categories.append("6–20 Fact-Checks")
                else:
                    categories.append("20+ Fact-Checks")

                # Query-based categorization
                if is_url(query):
                    categories.append("Mostly URLs")
                if is_video_transcript(query):
                    categories.append("Video transcript")
                if query and len(query) < 100:
                    categories.append("Mostly short form")
                if query and len(query) >= 100:
                    categories.append("Long form")

                if not categories:
                    categories.append("Uncategorized")

                user_categories.append({
                    "userEmail": email,
                    "Categories": categories
                })

            return pd.DataFrame(user_categories)


        cat_df = categorize_user_activity(df)

        if cat_df.empty:
            st.warning("No users to categorize.")
            st.stop()

        # Flatten unique categories
        all_categories = sorted(
            set(cat for sublist in cat_df["Categories"] for cat in sublist))
        selected_categories = st.multiselect(
            "Select Engagement Categories", all_categories)

        # Filter users who belong to ANY of the selected categories
        if selected_categories:
            filtered_users = cat_df[cat_df["Categories"].apply(
                lambda user_cats: any(cat in user_cats for cat in selected_categories)
            )]
        else:
            filtered_users = cat_df

        st.subheader("📧 Emails in selected categories")
        st.dataframe(filtered_users[["userEmail", "Categories"]])

        # Download
        csv_data = filtered_users[["userEmail"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Emails",
            data=csv_data,
            file_name="filtered_users.csv",
            mime="text/csv"
        )
# daily_start = (now_ts - timedelta(days=14)).isoformat()
# daily_end = now_ts.isoformat()
# st.text()
