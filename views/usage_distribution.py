"""
Interactive user engagement view for the Facticity dashboard.
"""
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events
from datetime import datetime, timedelta
import pandas as pd

from database.mongo_client import get_db_connection, get_distribution_data, is_valid_user


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
    selected_period = st.selectbox(
        "Select Time Period", ("Last 7 Days", "Last 30 Days"))
    days = 7 if selected_period == "Last 7 Days" else 30
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
