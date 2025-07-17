"""
User activity view for the Facticity dashboard.
"""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from wordcloud import WordCloud
from pymongo import MongoClient
from config import DB_CONNECTION_STRING

from database.mongo_client import get_db_connection, is_valid_user, fetch_all_emails_with_timestamps
from database.auth0_client import get_auth0_user_list
from services.analytics import get_active_users, calculate_retention_rate, get_active_users_custom

client = MongoClient(DB_CONNECTION_STRING)
db = client["facticity"]
gamef = db["gamefile"]
discover_collection = db["discover_posts"]
collections=db['query_new']
def get_unique_discover_emails(collection, days=1):
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=days)

    recent_posts = collection.find({
        "publish_timestamp": {"$gte": past},
        "user_email": {"$exists": True, "$ne": ""}
    })

    return sorted({post["user_email"] for post in recent_posts})


def get_unique_game_emails(collection, days=1):
    since = datetime.now(timezone.utc) - timedelta(days=days)

    cursor = collection.find(
        {"timestamp": {"$gte": since}},
        {"player_results": 1}
    )

    emails = set()
    for doc in cursor:
        for player in doc.get("player_results", []):
            email = player.get("email")
            if email:
                emails.add(email)
    return sorted(emails)
def get_unique_discover_posters(collection, days=1):
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=days)

    # Filter documents within time range
    recent_posts = collection.find({
        "publish_timestamp": {"$gte": past},
        "user_email": {"$exists": True, "$ne": ""}
    })

    # Collect unique emails
    unique_users = set()
    for post in recent_posts:
        unique_users.add(post["user_email"])

    return len(unique_users)

def get_game_player_counts(collection, days: int) -> int:
    """Count total players from player_results in the past `days`."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    cursor = collection.find(
        {"timestamp": {"$gte": since}},
        {"player_results": 1}
    )

    total_players = 0
    for doc in cursor:
        player_results = doc.get("player_results", [])
        total_players += len(player_results)

    return total_players


def show_user_activity_view():
    """
    Display the User Activity view.
    """
    collection = get_db_connection()
    with st.sidebar:
        exclude_blacklisted = st.checkbox("Exclude Internal/Blacklist Users", value=True,
                                          help="Filter out emails in the blacklist and from blacklisted domains")

    # Word Cloud Generation
    st.subheader("Query Word Cloud")

    # Add user input for custom stopwords
    st.sidebar.subheader("Word Cloud Settings")
    custom_user_stopwords = st.sidebar.text_input(
        "Add custom stopwords (comma-separated)",
        placeholder="Example: word1, word2, word3"
    )

    user_stopwords_list = []
    if custom_user_stopwords:
        user_stopwords_list = [word.strip().lower(
        ) for word in custom_user_stopwords.split(',') if word.strip()]

    generate_button = st.sidebar.button("Regenerate Word Cloud")

    with st.spinner("Generating word cloud..."):
        # Get recent queries for word cloud
        seven_days_ago = (datetime.now(timezone.utc) -
                          timedelta(days=7)).isoformat()
        
        if exclude_blacklisted:
            recent_queries = [
                doc["query"] for doc in collection.find(
                    {"timestamp": {"$gte": seven_days_ago}, "query": {"$exists": True}},
                    {"query": 1, "userEmail": 1}
                ) if is_valid_user(doc.get("userEmail"))
            ]
        else:
            recent_queries = [
                doc["query"] for doc in collection.find(
                    {"timestamp": {"$gte": seven_days_ago},
                        "query": {"$exists": True}},
                    {"query": 1, "userEmail": 1}
                )
            ]

        if recent_queries:
            text = " ".join(recent_queries)

            # Define custom stopwords including common words and specific terms to exclude
            custom_stopwords = {
                'timestamp', 'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by',
                'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as',
                'your', 'all', 'have', 'new', 'more', 'an', 'was', 'we', 'will', 'can', 'us',
                'about', 'if', 'my', 'has', 'but', 'our', 'one', 'other', 'do', 'no', 'they',
                'he', 'she', 'they', 'their', 'what', 'so', 'up', 'when', 'who', 'which', 'its',
                'out', 'into', 'just', 'some', 'there', 'what', 'am', 'been', 'would', 'make',
                'like', 'time', 'did', 'query', 'now', 'get', 'could', 'than', 'used', 'using',
                'show', 'find', 'please', 'need', 'how', 'me', 'because', 'any', 'these', 'those',
                'was', 'were', 'only', 'should', 'also', 'S', 'U'
            }

            # Add user-defined stopwords
            custom_stopwords.update(user_stopwords_list)

            # If WordCloud comes with STOPWORDS, add them to our custom set
            try:
                from wordcloud import STOPWORDS
                custom_stopwords = custom_stopwords.union(STOPWORDS)
            except (ImportError, NameError):
                pass

            # Create color function for #0066ff shades (darker for more frequent words)
            def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                # Map font size to color intensity (200-950 range)
                max_size = 50  # Adjust based on your word cloud settings
                # Calculate weight - higher frequency = darker shade
                weight = min(950, 200 + int((font_size / max_size) * 750))

                # Convert to hex (assuming tailwind-like scale where higher is darker)
                if weight <= 300:
                    return "#bfdbfe"  # 200 shade
                elif weight <= 400:
                    return "#93c5fd"  # 300 shade
                elif weight <= 500:
                    return "#60a5fa"  # 400 shade
                elif weight <= 600:
                    return "#3b82f6"  # 500 shade
                elif weight <= 700:
                    return "#2563eb"  # 600 shade
                elif weight <= 800:
                    return "#1d4ed8"  # 700 shade
                elif weight <= 900:
                    return "#1e40af"  # 800 shade
                else:
                    return "#1e3a8a"  # 900/950 shade

            # Generate word cloud with transparent background
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color=None,
                mode="RGBA",
                stopwords=custom_stopwords,
                color_func=blue_color_func,
                max_words=150,
                contour_width=0,
                contour_color=None
            ).generate(text)

            # Display word cloud
            # Transparent figure background
            plt.figure(figsize=(10, 5), facecolor='none')
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

            # Display the most common words (optional)
            if st.sidebar.checkbox("Show top words"):
                from collections import Counter
                import re

                # Process text to extract words
                words = re.findall(r'\b\w+\b', text.lower())
                # Remove stopwords
                filtered_words = [
                    word for word in words if word not in custom_stopwords]
                # Count word frequency
                word_counts = Counter(filtered_words).most_common(20)

                st.subheader("Top 20 Words")
                word_df = pd.DataFrame(word_counts, columns=["Word", "Count"])
                st.dataframe(word_df)
        else:
            st.write("No recent queries available for word cloud generation.")
    # Active User Metrics
    st.subheader("Active User Metrics")

    with st.spinner("Calculating active users..."):
        # Get active users for different time periods
        
        daily_active_users = get_active_users("daily", exclude_blacklisted=exclude_blacklisted)
        weekly_active_users = get_active_users(
            "weekly", exclude_blacklisted=exclude_blacklisted)
        monthly_active_users = get_active_users(
            "monthly", exclude_blacklisted=exclude_blacklisted)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Daily Active Users (last 24h)", len(daily_active_users))
        with col2:
            st.metric("Weekly Active Users (last 7d)",
                      len(weekly_active_users))
        with col3:
            st.metric("Monthly Active Users (last 30d)",
                      len(monthly_active_users))
    # === New Section for Game and Discover Posting Active Users ===
   

    def get_module_activity_users(collection, module_name: str, days: int):
        since = datetime.now(timezone.utc) - timedelta(days=days)
        match_stage = {
            "timestamp": {"$gte": since.isoformat()},
            "userEmail": {"$ne": None},
            "module": module_name
        }

        pipeline = [
            {"$match": match_stage},
            {"$group": {"_id": "$userEmail"}}
        ]

        result = list(collection.aggregate(pipeline))
        valid_users = [r["_id"] for r in result if not exclude_blacklisted or is_valid_user(r["_id"])]
        return len(set(valid_users))

    with st.spinner("Calculating engagement per module..."):

        st.subheader("Game & Discover Engagement")
        daily_game_players = get_game_player_counts(gamef, 1)
        weekly_game_players = get_game_player_counts(gamef, 7)
        monthly_game_players = get_game_player_counts(gamef, 30)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Game Daily", daily_game_players)
        with col2:
            st.metric("Game Weekly", weekly_game_players)
        with col3:
            st.metric("Game Monthly", monthly_game_players)
        daily_discover_posters = get_unique_discover_posters(discover_collection, days=1)
        weekly_discover_posters = get_unique_discover_posters(discover_collection, days=7)
        monthly_discover_posters = get_unique_discover_posters(discover_collection, days=30)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Discover Daily", daily_discover_posters)
        with col2:
            st.metric("Discover Weekly", weekly_discover_posters)
        with col3:
            st.metric("Discover Monthly", monthly_discover_posters)

    # Retention Rate Calculation
    st.subheader("User Retention")
    st.markdown(r"% of users in current period who were active prior; e.g. in this week how many users used facticity before this week?")

    # Allow user to select timeframe
    timeframe = st.selectbox("Select Timeframe", [
                             "Daily", "Weekly", "Monthly"])

    with st.spinner("Calculating retention rate..."):
        retention_rate, period_start, period_end = calculate_retention_rate(
            timeframe, exclude_blacklisted)

        st.write(
            f"Calculating rates from {period_start} to now ({period_end})")

        if retention_rate is not None:
            st.metric(
                f"{timeframe} Retention Rate (Existing Users)",
                f"{retention_rate:.1f}%"
            )
        else:
            st.write("No data available for the selected period.")

    # Cohort Retention Heatmap
    st.subheader("Cohort Retention Heatmap")

    # Get raw query data with timestamps and emails
    with st.spinner("Generating cohort analysis..."):
        now = datetime.now(timezone.utc)

        # Get data for last 6 months
        start_date = (now - timedelta(days=180)).replace(day=1).replace(hour=0,
                                                                        minute=0, second=0, microsecond=0)
        start_iso = start_date.isoformat()
        end_iso = now.isoformat()

        # Build aggregation pipeline for cohort analysis
        pipeline = [
            {"$match": {"timestamp": {"$gte": start_iso,
                                      "$lt": end_iso}, "userEmail": {"$ne": None}}},
            {"$project": {
                "userEmail": 1,
                "month": {"$dateToString": {"format": "%Y-%m", "date": {"$toDate": "$timestamp"}}},
                "timestamp": 1
            }},
            {"$group": {
                "_id": {"userEmail": "$userEmail", "month": "$month"},
                "count": {"$sum": 1},
                "first_seen": {"$min": "$timestamp"}
            }},
            {"$sort": {"first_seen": 1}}
        ]

        cohort_data = list(collection.aggregate(pipeline))

        if cohort_data:
            # Process cohort data
            user_cohorts = {}
            user_monthly_activity = {}

            for item in cohort_data:
                email = item["_id"]["userEmail"]
                month = item["_id"]["month"]
                first_seen = item["first_seen"]

                if exclude_blacklisted:
                    if not is_valid_user(email):
                        continue

                # Determine cohort (first month user was seen)
                if email not in user_cohorts:
                    # Convert to datetime to extract month
                    first_dt = pd.to_datetime(first_seen)
                    cohort_month = first_dt.strftime('%Y-%m')
                    user_cohorts[email] = cohort_month

                # Track monthly activity
                if email not in user_monthly_activity:
                    user_monthly_activity[email] = set()
                user_monthly_activity[email].add(month)

            # Create cohort analysis dataframe
            if user_cohorts:
                # Get sorted list of all months in the period
                all_months = sorted(set(m["_id"]["month"]
                                    for m in cohort_data))

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

                    # Prepare data for heatmap
                    z_data = df[month_cols].values
                    x_labels = [f'Month {i}' for i in range(len(month_cols))]
                    y_labels = df['cohort'].tolist()

                    # Text annotations (percentage values)
                    text = [[f"{val:.1f}%" for val in row] for row in z_data]

                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=z_data,
                        x=x_labels,
                        y=y_labels,
                        text=text,
                        texttemplate="%{text}",
                        colorscale='Blues',
                        hoverongaps=False))

                    fig.update_layout(
                        title="User Retention by Cohort (%)",
                        xaxis_title="Months Since First Query",
                        yaxis_title="Cohort (First Month)",
                        width=800,
                        height=500
                    )

                    # Display cohort sizes
                    st.info(
                        f"Number of cohorts: {len(cohorts)}. Total users: {sum(cohort_sizes.values())}")

                    # Display heatmap
                    st.plotly_chart(fig)
                else:
                    st.write("Insufficient data for cohort analysis.")
            else:
                st.write("No valid users found for cohort analysis.")
        else:
            st.write("No data available for cohort analysis.")

        st.subheader("📩 Download User Emails Separately (Game | Discover | Active)")

        days = st.slider("Select timeframe (days)", 1, 90, 7)

        game_emails = get_unique_game_emails(gamef, days)
        discover_emails = get_unique_discover_emails(discover_collection, days)
        active_emails = sorted(get_active_users_custom(days))

        max_len = max(len(game_emails), len(discover_emails), len(active_emails))

        # Pad shorter lists with empty strings
        game_emails += [""] * (max_len - len(game_emails))
        discover_emails += [""] * (max_len - len(discover_emails))
        active_emails += [""] * (max_len - len(active_emails))

        df = pd.DataFrame({
            "game_user_email": game_emails,
            "discover_user_email": discover_emails,
            "active_user_email": active_emails
        })

        st.dataframe(df)

        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False),
            file_name=f"user_emails_{days}_days.csv",
            mime="text/csv"
        )
    st.subheader("🔍 Filter User Emails by Requester URL")

    # MongoDB collection access
    collection = collections

    # Fetch relevant query data within selected time window
    query_filter = {
        "timestamp": {"$gte": (datetime.utcnow() - timedelta(days=days)).isoformat()}
    }
    # Correct projection (no filters.requester_url!)
    projection = {"userEmail": 1, "requester_url": 1, "timestamp": 1}

    # Fetch data
    query_data = list(collection.find(query_filter, projection))

    if not query_data:
        st.warning("No query data found in the selected time window.")
        st.stop()

    # Convert to DataFrame
    df = pd.DataFrame(query_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop entries without requester_url
    df["requester_url"] = df["requester_url"].fillna("").astype(str).str.strip()

    # Filter only rows where requester_url is empty
    df.loc[df["requester_url"] == "", "requester_url"] = "Writer"
    # Categorize requester types
    def categorize_requester(url):
        if url == "https://app.facticity.ai":
            return "Facticity App"
        elif url == "chrome-extension://mlackneplpmmomaobipjjpebhgcgmocp/sidebar.html":
            return "Chrome Extension"
        elif url == "x_bot":
            return "Bot"
        elif url=="Writer":
            return "Writer"
        else:
            return "Other"

    df["requester_type"] = df["requester_url"].apply(categorize_requester)

    # Group and display user emails
    for requester_type, group in df.groupby("requester_type"):
        emails = group["userEmail"].dropna().unique()
        email_df = pd.DataFrame(emails, columns=["userEmail"])

        st.markdown(f"### 📂 {requester_type} ({len(emails)} users)")
        st.dataframe(email_df)

        st.download_button(
            label=f"⬇️ Download {requester_type} Emails",
            data=email_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{requester_type.lower().replace(' ', '_')}_emails.csv",
            mime="text/csv"
        )
