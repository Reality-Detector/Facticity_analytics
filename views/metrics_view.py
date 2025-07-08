"""
Metrics dashboard view for the Facticity application.
"""
from collections import defaultdict
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
from pymongo import MongoClient
import altair as alt

from database.mongo_client import (
    aggregate_daily_with_users,
    aggregate_weekly_with_users,
    aggregate_monthly_with_users,
    aggregate_quarterly_with_users,
    aggregate_daily_by_url,
    aggregate_daily_users_by_url
)
from database.auth0_client import (
    get_auth0_user_list,
    aggregate_auth0_daily,
    aggregate_auth0_weekly,
    aggregate_auth0_monthly,
    aggregate_auth0_quarterly
)
from utils.chart_utils import (
    generate_chart,
    generate_url_breakdown_chart,
    generate_user_breakdown_chart
)

from utils.date_utils import quarter_sort_key

from config import API_DB_CONNECTION_STRING, PRIMARY_BLUE, LIGHT_BLUE, MODERN_ORANGE, green_1, blue_5
from config import DB_CONNECTION_STRING


def show_metrics_view():
    """
    Display the Metrics dashboard view.
    """
    st.title("Metrics Dashboard")

    # Add sidebar option for excluding blacklisted users
    with st.sidebar:
        exclude_blacklisted = st.checkbox("Exclude Internal/Blacklist Users", value=True,
                                          help="Filter out emails in the blacklist and from blacklisted domains")

    # Define date ranges
    now = pd.Timestamp.now(tz='UTC')
    daily_start = (now - timedelta(days=14)).isoformat()
    daily_end = now.isoformat()

    monthly_start = "2024-07-01T00:00:00Z"
    monthly_end = now.isoformat()
    quarterly_start = "2024-06-01T00:00:00Z"
    quarterly_end = now.isoformat()
    weekly_start = monthly_start
    weekly_end = monthly_end

    # Show filter status
    if exclude_blacklisted:
        st.info("📊 Showing metrics with blacklisted users excluded")

    left_col, right_col = st.columns(2)
    with left_col:
        # Get URL breakdown data for queries by URL
        breakdown = aggregate_daily_by_url(
            daily_start, daily_end, exclude_blacklisted)

        total = sum(
            value
            for item in breakdown
            for value in item["urls"].values()
        )

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

        # Custom color map to ensure API is green_1
        color_map = {"API": green_1, "chrome-extension://mlackneplpmmomaobipjjpebhgcgmocp/sidebar.html": MODERN_ORANGE,
                     "https://app.facticity.ai": LIGHT_BLUE, "writer": blue_5}

        # Generate the chart with custom colors
        generate_url_breakdown_chart(
            title="Daily Queries by URL (Last 14 Days)",
            dates=dates,
            url_counts=url_counts,
            color_map=color_map  # Pass custom color map
        )

    with right_col:
        # Get user breakdown by URL
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
            all_urls = sorted(
                {url for row in normalized_user_breakdown for url in row["urls"].keys()})

            # Build per-URL series with zero fill
            url_users = {
                url: [row["urls"].get(url, 0)
                      for row in normalized_user_breakdown]
                for url in all_urls
            }

            generate_user_breakdown_chart(
                title="Daily Active Users by URL (Last 14 Days)",
                dates=user_dates,
                url_users=url_users,
                color_map=color_map  # Use the same color map for consistency
            )
        else:
            st.write("No daily active users by URL data available.")

    # Get Auth0 DataFrame
    auth0_df = get_auth0_user_list()

    # Fetch aggregated data with blacklist exclusion option
    with st.spinner("Loading metrics data..."):
        weekly_results = aggregate_weekly_with_users(
            weekly_start, weekly_end, exclude_blacklisted)
        monthly_results = aggregate_monthly_with_users(
            monthly_start, monthly_end, exclude_blacklisted)
        quarterly_results = aggregate_quarterly_with_users(
            quarterly_start, quarterly_end, exclude_blacklisted)

    # Weekly Chart
    if weekly_results:
        generate_chart(
            title="Weekly Queries & Auth0 Users",
            x_labels=[doc["_id"] for doc in weekly_results],
            y_queries=[doc["query_count"] for doc in weekly_results],
            auth0_overlay=aggregate_auth0_weekly(
                auth0_df, weekly_start, weekly_end)
        )
    else:
        st.write("No weekly data available.")

    # Monthly Chart
    if monthly_results:
        generate_chart(
            title="Monthly Queries & Auth0 Users",
            x_labels=[
                f'{doc["_id"]["month"]:02d}/{doc["_id"]["year"]}' for doc in monthly_results],
            y_queries=[doc["query_count"] for doc in monthly_results],
            auth0_overlay=aggregate_auth0_monthly(
                auth0_df, monthly_start, monthly_end)
        )
    else:
        st.write("No monthly query data available.")

    # Quarterly Chart
    if quarterly_results:
        q_labels = [record["_id"] for record in quarterly_results]
        q_queries = [record["query_count"] for record in quarterly_results]

        # Get Auth0 quarterly data
        auth0_quarterly = aggregate_auth0_quarterly(
            auth0_df, quarterly_start, quarterly_end)

        # Ensure Auth0 data uses the same labels and sort correctly
        auth0_quarterly = auth0_quarterly.reindex(q_labels, fill_value=0)

        # Sort labels correctly
        sorted_labels = sorted(q_labels, key=quarter_sort_key)
        label_to_query = dict(zip(q_labels, q_queries))
        sorted_queries = [label_to_query[label] for label in sorted_labels]

        # Adjust Auth0 data to match sorted labels
        auth0_quarterly = auth0_quarterly.sort_index(
            key=lambda x: x.map(quarter_sort_key))

        # Generate chart
        generate_chart(
            title="Quarterly Queries & Auth0 Users",
            x_labels=sorted_labels,
            y_queries=sorted_queries,
            auth0_overlay=auth0_quarterly,
            sort_key=quarter_sort_key
        )
    else:
        st.write("No quarterly query data available.")

    # Summary Metrics
    total_api_queries = sum(
        count for count in api_counts.values()) if api_counts else 0
    total_monthly_queries = sum(doc["query_count"]
                                for doc in monthly_results) if monthly_results else 0
    total_users = auth0_df['user_id'].nunique(
    ) if 'user_id' in auth0_df.columns else 0

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries (14 days)", total_api_queries+total)
    with col2:
        st.metric("Total Queries since July", total_monthly_queries)
    with col3:
        st.metric("Total Users since July", total_users)
    # ---------------------------
    # Game Players Per Day Chart
    # ---------------------------
    # from pymongo import MongoClient

    st.subheader("📈 Game Participation Over Time")

    # Connect to MongoDB (reuse or import your existing connection string)
    game_client = MongoClient(DB_CONNECTION_STRING)
    game_db = game_client["facticity"]
    game_collection = game_db["gamefile"]

    # Dictionary: {date → set of player emails}
    # from collections import defaultdict
    # import pandas as pd

    players_per_day = defaultdict(set)

    for doc in game_collection.find({"player_results": {"$exists": True}}):
        timestamp = doc.get("timestamp")
        if not timestamp:
            continue
        date = pd.to_datetime(timestamp).date()

        for result in doc.get("player_results", []):
            email = result.get("email")
            if email:
                players_per_day[date].add(email)

    # Convert to sorted DataFrame
    data = {
        "date": sorted(players_per_day.keys()),
        "unique_players": [len(players_per_day[dt]) for dt in sorted(players_per_day.keys())]
    }
    df_players = pd.DataFrame(data)

    # Plot using Streamlit chart
    st.line_chart(df_players.set_index("date"), use_container_width=True)
    # ------------------------------------------------
    # Discover Posts Activity: New Posts per Day (14d)
    # ------------------------------------------------
    st.subheader("📰 New Discover Posts Per Day (Last 14 Days)")

    discover_client = MongoClient(DB_CONNECTION_STRING)
    discover_db = discover_client["facticity"]
    posts_collection = discover_db["discover_posts"]


    def get_recent_discover_posts(days=14):
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        pipeline = [
            {
                "$match": {
                    "publish_timestamp": {"$gte": cutoff_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$publish_timestamp"
                        }
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        result = list(posts_collection.aggregate(pipeline))
        if not result:
            return pd.DataFrame(columns=["date", "posts"])
        return pd.DataFrame({
            "date": [r["_id"] for r in result],
            "posts": [r["count"] for r in result]
        })

    df_posts = get_recent_discover_posts()

    if not df_posts.empty:

        df_posts['date'] = pd.to_datetime(df_posts['date'])

        # Set the date as index for proper plotting
        df = df_posts.set_index('date')

        # Use Streamlit's built-in line chart
        st.line_chart(df['posts'], use_container_width=True)
    else:

        st.write("No discover post activity in the last 14 days.")
