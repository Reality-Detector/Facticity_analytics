"""
Twitter Bot Analytics view for the Facticity dashboard.
Analyzes tweet processing, response times, and bot performance metrics.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import numpy as np

from dbutils.DocumentDB import document_db_web2
from utils.chart_utils import generate_chart


def get_tweets_collection():
    """
    Get connection to the tweets collection.
    
    Returns:
        MongoDB collection: The tweets collection
    """
    try:
        client = document_db_web2.get_client()
        db = client["facticity"]
        return db["tweets"]
    except Exception as e:
        st.error(f"Failed to connect to tweets collection: {str(e)}")
        return None


def calculate_response_time_stats(collection, days=30):
    """
    Calculate response time statistics for tweets.
    
    Args:
        collection: MongoDB tweets collection
        days: Number of days to look back
        
    Returns:
        dict: Response time statistics
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    pipeline = [
        {
            "$match": {
                "created_at": {"$gte": cutoff_date},
                "reply_timestamp": {"$exists": True, "$ne": None},
                "replied": True
            }
        },
        {
            "$project": {
                "created_at": 1,
                "reply_timestamp": 1,
                "stage": 1,
                "user_intent.intent": 1,
                "tagger.username": 1,
                "response_time_minutes": {
                    "$divide": [
                        {"$subtract": ["$reply_timestamp", "$created_at"]},
                        60000  # Convert milliseconds to minutes
                    ]
                }
            }
        },
        {
            "$match": {
                "response_time_minutes": {"$gte": 0, "$lte": 1440}  # Filter out negative times and >24h
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    stats = {
        "total_tweets": len(df),
        "avg_response_time": df["response_time_minutes"].mean(),
        "median_response_time": df["response_time_minutes"].median(),
        "min_response_time": df["response_time_minutes"].min(),
        "max_response_time": df["response_time_minutes"].max(),
        "std_response_time": df["response_time_minutes"].std(),
        "data": df
    }
    
    return stats


def get_daily_response_times(collection, days=30):
    """
    Get daily response time trends.
    
    Args:
        collection: MongoDB tweets collection
        days: Number of days to look back
        
    Returns:
        DataFrame: Daily response time data
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    pipeline = [
        {
            "$match": {
                "created_at": {"$gte": cutoff_date},
                "reply_timestamp": {"$exists": True, "$ne": None},
                "replied": True
            }
        },
        {
            "$project": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "response_time_minutes": {
                    "$divide": [
                        {"$subtract": ["$reply_timestamp", "$created_at"]},
                        60000
                    ]
                }
            }
        },
        {
            "$match": {
                "response_time_minutes": {"$gte": 0, "$lte": 1440}
            }
        },
        {
            "$group": {
                "_id": "$date",
                "avg_response_time": {"$avg": "$response_time_minutes"},
                "tweet_count": {"$sum": 1},
                "min_response_time": {"$min": "$response_time_minutes"},
                "max_response_time": {"$max": "$response_time_minutes"},
                "response_times": {"$push": "$response_time_minutes"}
            }
        },
        {
            "$sort": {"_id": 1}
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    
    if not results:
        return pd.DataFrame()
    
    # Calculate median for each date in Python
    for result in results:
        if result.get('response_times'):
            result['median_response_time'] = np.median(result['response_times'])
        else:
            result['median_response_time'] = 0
    
    # Convert to DataFrame and remove the response_times column
    df = pd.DataFrame(results)
    if 'response_times' in df.columns:
        df = df.drop('response_times', axis=1)
    
    return df


def get_intent_breakdown(collection, days=30):
    """
    Get breakdown of tweets by user intent (filtered to relevant intents only).
    
    Args:
        collection: MongoDB tweets collection
        days: Number of days to look back
        
    Returns:
        DataFrame: Intent breakdown data
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Only include relevant user intents
    relevant_intents = ["casual", "fact_check", "continuation", "ill_intentioned", "faq_request"]
    
    pipeline = [
        {
            "$match": {
                "created_at": {"$gte": cutoff_date},
                "user_intent.intent": {"$in": relevant_intents}
            }
        },
        {
            "$group": {
                "_id": "$user_intent.intent",
                "count": {"$sum": 1},
                "avg_response_time": {
                    "$avg": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$reply_timestamp", None]},
                                    {"$gte": [
                                        {"$divide": [
                                            {"$subtract": ["$reply_timestamp", "$created_at"]},
                                            60000
                                        ]}, 0
                                    ]}
                                ]
                            },
                            {"$divide": [
                                {"$subtract": ["$reply_timestamp", "$created_at"]},
                                60000
                            ]},
                            None
                        ]
                    }
                }
            }
        },
        {
            "$sort": {"count": -1}
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def get_stage_breakdown(collection, days=30):
    """
    Get breakdown of tweets by processing stage (filtered to relevant stages only).
    
    Args:
        collection: MongoDB tweets collection
        days: Number of days to look back
        
    Returns:
        DataFrame: Stage breakdown data
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Only include relevant processing stages
    relevant_stages = [
        "no_intent", "replied", "continuation", "tweet_generated", 
        "tweet_extracting", "ill_intentioned", "faq_request", 
        "no_araistotle_tag", "retry_failed", "permanently_failed"
    ]
    
    pipeline = [
        {
            "$match": {
                "created_at": {"$gte": cutoff_date},
                "stage": {"$in": relevant_stages}
            }
        },
        {
            "$group": {
                "_id": "$stage",
                "count": {"$sum": 1}
            }
        },
        {
            "$sort": {"count": -1}
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def create_response_time_chart(daily_data):
    """
    Create a response time trend chart.
    
    Args:
        daily_data: DataFrame with daily response time data
    """
    if daily_data.empty:
        st.warning("No daily response time data available.")
        return
    
    fig = go.Figure()
    
    # Add average response time line
    fig.add_trace(go.Scatter(
        x=daily_data["_id"],
        y=daily_data["avg_response_time"],
        mode="lines+markers",
        name="Average Response Time",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=8)
    ))
    
    # Add median response time line
    fig.add_trace(go.Scatter(
        x=daily_data["_id"],
        y=daily_data["median_response_time"],
        mode="lines+markers",
        name="Median Response Time",
        line=dict(color="#10b981", width=3, dash="dash"),
        marker=dict(size=8)
    ))
    
    # Add tweet count as bars (secondary y-axis)
    fig.add_trace(go.Bar(
        x=daily_data["_id"],
        y=daily_data["tweet_count"],
        name="Tweet Count",
        yaxis="y2",
        opacity=0.3,
        marker_color="#f59e0b"
    ))
    
    # Update layout
    fig.update_layout(
        title="Daily Tweet Response Times & Volume",
        xaxis_title="Date",
        yaxis_title="Response Time (minutes)",
        yaxis2=dict(
            title="Tweet Count",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_intent_pie_chart(intent_data):
    """
    Create a pie chart for user intents.
    
    Args:
        intent_data: DataFrame with intent breakdown data
    """
    if intent_data.empty:
        st.warning("No intent data available.")
        return
    
    fig = px.pie(
        intent_data,
        values="count",
        names="_id",
        title="Tweet Distribution by User Intent",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def create_stage_bar_chart(stage_data):
    """
    Create a bar chart for processing stages.
    
    Args:
        stage_data: DataFrame with stage breakdown data
    """
    if stage_data.empty:
        st.warning("No stage data available.")
        return
    
    fig = px.bar(
        stage_data,
        x="_id",
        y="count",
        title="Tweet Distribution by Processing Stage",
        color="_id",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        xaxis_title="Processing Stage",
        yaxis_title="Number of Tweets",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_twitter_bot_analytics():
    """
    Display the Twitter Bot Analytics view.
    """
    st.title("Twitter Bot Analytics")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Analytics Settings")
        days = st.slider("Analysis Period (days)", 1, 45, 15)
        
        st.subheader("Chart Options")
        show_daily_trends = st.checkbox("Show Daily Trends", value=True)
        show_intent_breakdown = st.checkbox("Show Intent Breakdown", value=True)
        show_stage_breakdown = st.checkbox("Show Stage Breakdown", value=True)
    
    # Get tweets collection
    collection = get_tweets_collection()
    if collection is None:
        st.error("Could not connect to tweets collection.")
        return
    
    # Calculate response time statistics
    with st.spinner("Calculating response time statistics..."):
        stats = calculate_response_time_stats(collection, days)
    
    if not stats:
        st.warning(f"No tweet data found for the last {days} days.")
        return
    
    # Display key metrics
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tweets Replied",
            f"{stats['total_tweets']:,}",
            help="Number of tweets replied to in the selected period.\nTotal tweets tagging the bot would be greater but not all get replies based on fact-check intent."
        )
    
    with col2:
        avg_time = stats['avg_response_time']
        st.metric(
            "Average Response Time",
            f"{avg_time:.1f} min",
            help="Average time from tweet creation to bot reply"
        )
    
    with col3:
        median_time = stats['median_response_time']
        st.metric(
            "Median Response Time",
            f"{median_time:.1f} min",
            help="Median time from tweet creation to bot reply"
        )
    
    with col4:
        min_time = stats['min_response_time']
        st.metric(
            "Fastest Response",
            f"{min_time:.1f} min",
            help="Fastest response time recorded"
        )
    
    # Response time distribution
    st.subheader("Response Time Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of response times (filtered to 120 minutes max, bin size capped at 20 minutes)
        filtered_data = stats['data'][stats['data']['response_time_minutes'] <= 120]
        
        # Calculate appropriate number of bins to keep bin size ≤ 20 minutes
        max_time = filtered_data['response_time_minutes'].max() if len(filtered_data) > 0 else 120
        bin_size = min(20, max_time / 10)  # Cap bin size at 20 minutes, minimum 10 bins
        nbins = max(10, int(max_time / bin_size))
        
        fig_hist = px.histogram(
            filtered_data,
            x="response_time_minutes",
            nbins=nbins,
            title="Response Time Distribution",
            labels={"response_time_minutes": "Response Time (minutes)", "count": "Number of Tweets"}
        )
        fig_hist.update_layout(showlegend=False)
        
        # Add outlines to bars for better visual separation
        fig_hist.update_traces(
            marker=dict(
                line=dict(width=1, color='white')
            )
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot of response times (filtered to 120 minutes max for better detail)
        filtered_data_box = stats['data'][stats['data']['response_time_minutes'] <= 120]
        
        if len(filtered_data_box) > 0:
            fig_box = px.box(
                filtered_data_box,
                y="response_time_minutes",
                title="Response Time (Box Plot)",
                labels={"response_time_minutes": "Response Time (minutes)"}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No response time data available for box plot")
    
    # Daily trends
    if show_daily_trends:
        st.subheader("Daily Response Time Trends")
        
        with st.spinner("Loading daily trends..."):
            daily_data = get_daily_response_times(collection, days)
        
        if not daily_data.empty:
            create_response_time_chart(daily_data)
        else:
            st.warning("No daily trend data available.")
    
    # Intent breakdown
    if show_intent_breakdown:
        st.subheader("User Intent Analysis")
        
        with st.spinner("Loading intent data..."):
            intent_data = get_intent_breakdown(collection, days)
        
        if not intent_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                create_intent_pie_chart(intent_data)
            
            with col2:
                # Intent table with response times
                st.subheader("Intent Details")
                intent_table = intent_data.copy()
                intent_table["avg_response_time"] = intent_table["avg_response_time"].round(1)
                intent_table.columns = ["Intent", "Count", "Avg Response Time (min)"]
                st.dataframe(intent_table, use_container_width=True)
        else:
            st.warning("No intent data available.")
    
    # Stage breakdown
    if show_stage_breakdown:
        st.subheader("Processing Stage Analysis")
        
        with st.spinner("Loading stage data..."):
            stage_data = get_stage_breakdown(collection, days)
        
        if not stage_data.empty:
            create_stage_bar_chart(stage_data)
        else:
            st.warning("No stage data available.")
    
    # Raw data section
    with st.expander("Raw Data"):
        st.subheader("Recent Tweet Data")
        
        # Show recent tweets with key fields
        recent_tweets = list(collection.find(
            {"created_at": {"$gte": datetime.now(timezone.utc) - timedelta(days=7)}},
            {
                "created_at": 1,
                "reply_timestamp": 1,
                "stage": 1,
                "user_intent.intent": 1,
                "tagger.username": 1,
                "parent_tweet": 1,
                "fact_check_summary": 1
            }
        ).sort("created_at", -1).limit(50))
        
        if recent_tweets:
            df_recent = pd.DataFrame(recent_tweets)
            
            # Calculate response time for display
            df_recent["response_time_minutes"] = df_recent.apply(
                lambda row: (row["reply_timestamp"] - row["created_at"]).total_seconds() / 60
                if row.get("reply_timestamp") and row.get("created_at") else None, axis=1
            )
            
            # Select columns to display - only include columns that actually exist
            available_cols = df_recent.columns.tolist()
            display_cols = []
            column_names = []
            
            # Map of desired columns to their display names
            col_mapping = {
                "created_at": "Created At",
                "stage": "Stage", 
                "user_intent.intent": "Intent",
                "tagger.username": "Tagger",
                "response_time_minutes": "Response Time (min)",
                "parent_tweet": "Parent Tweet"
            }
            
            # Only include columns that exist in the DataFrame
            for col, display_name in col_mapping.items():
                if col in available_cols:
                    display_cols.append(col)
                    column_names.append(display_name)
            
            if display_cols:
                df_display = df_recent[display_cols].copy()
                df_display.columns = column_names
            else:
                # Fallback: show all available columns
                df_display = df_recent.copy()
            
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No recent tweets found.")
