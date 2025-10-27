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

from dbutils.DocumentDB import document_db_web2, document_db_web3
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


def get_query_new_collection():
    """
    Get connection to the query_new collection.
    Now using web3 database for better feedback coverage.
    
    Returns:
        MongoDB collection: The query_new collection
    """
    try:
        client = document_db_web3.get_client()
        db = client["facticity"]
        return db["query_new"]
    except Exception as e:
        st.error(f"Failed to connect to query_new collection: {str(e)}")
        return None


def get_bonus_credit_tracker_collection():
    """
    Get connection to the bonus_credit_tracker collection.    
    Returns:
        MongoDB collection: The bonus_credit_tracker collection
    """
    try:
        client = document_db_web3.get_client()
        db = client["facticity"]
        return db["bonus_credit_tracker"]
    except Exception as e:
        st.error(f"Failed to connect to bonus_credit_tracker collection: {str(e)}")
        return None


def examine_feedbacks_structure():
    """
    Examine the structure of feedbacks array in query_new collection
    to understand the actual data format before creating visualizations.
    """
    try:
        query_collection = get_query_new_collection()
        if query_collection is None:
            st.error("Failed to connect to query_new collection")
            return
        
        # Get some sample documents with feedbacks
        sample_docs = list(query_collection.find(
            {"feedbacks": {"$exists": True, "$ne": []}},
            {"task_id": 1, "feedbacks": 1, "dislikes": 1}
        ).limit(5))
        
        if not sample_docs:
            st.info("No documents with feedbacks found")
            return
        
        st.subheader("📋 Sample Feedbacks Structure Analysis")
        
        for i, doc in enumerate(sample_docs, 1):
            st.write(f"**Sample Document {i}:**")
            st.write(f"Task ID: {doc.get('task_id', 'N/A')}")
            st.write(f"Dislikes: {doc.get('dislikes', 0)}")
            
            feedbacks = doc.get("feedbacks", [])
            st.write(f"Number of feedbacks: {len(feedbacks)}")
            
            if feedbacks:
                st.write("**Feedbacks Array Structure:**")
                for j, feedback in enumerate(feedbacks[:3], 1):  # Show first 3 feedbacks
                    st.write(f"  Feedback {j}:")
                    st.write(f"    - Keys: {list(feedback.keys())}")
                    
                    if "user_email" in feedback:
                        st.write(f"    - user_email: {feedback['user_email']}")
                    
                    if "comments" in feedback:
                        st.write(f"    - comments: {feedback['comments']}")
                    
                    if "reasons" in feedback:
                        st.write(f"    - reasons: {feedback['reasons']}")
                    
                    st.write("---")
            
            st.write("=" * 50)
        
        # Analyze all feedbacks to understand patterns
        st.subheader("📊 Feedbacks Analysis")
        
        all_feedbacks = []
        all_docs = list(query_collection.find(
            {"feedbacks": {"$exists": True, "$ne": []}},
            {"feedbacks": 1}
        ).limit(50))  # Analyze more docs
        
        for doc in all_docs:
            all_feedbacks.extend(doc.get("feedbacks", []))
        
        if all_feedbacks:
            st.write(f"**Total feedbacks analyzed: {len(all_feedbacks)}**")
            
            # Count unique reasons
            all_reasons = []
            all_comments = []
            users_with_feedback = set()
            
            for feedback in all_feedbacks:
                if "reasons" in feedback and isinstance(feedback["reasons"], list):
                    all_reasons.extend(feedback["reasons"])
                
                if "comments" in feedback and feedback["comments"]:
                    all_comments.append(feedback["comments"])
                
                if "user_email" in feedback and feedback["user_email"]:
                    users_with_feedback.add(feedback["user_email"])
            
            # Show reason frequency
            if all_reasons:
                reason_counts = {}
                for reason in all_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                st.write("**Most Common Reasons:**")
                sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
                for reason, count in sorted_reasons[:10]:
                    st.write(f"  - {reason}: {count} times")
            
            st.write(f"**Users who gave feedback: {len(users_with_feedback)}**")
            st.write(f"**Feedbacks with comments: {len(all_comments)}**")
            
            if all_comments:
                st.write("**Sample Comments:**")
                for comment in all_comments[:5]:
                    st.write(f"  - {comment}")
        
    except Exception as e:
        st.error(f"Error examining feedbacks structure: {str(e)}")


def get_moderator_feedback_mapping():
    """
    Maps moderator feedbacks across 3 collections:
    1. bonus_credit_tracker (type: "feedback") → task_ids
    2. tweets (stage: "replied") → queued_fact_check_tasks.task_id matching step 1
    3. query_new → feedbacks for matched task_ids
    
    Returns:
        list: List of tweet-feedback mappings with tweet details and feedback summaries
    """
    try:
        # Get all collections
        bonus_collection = get_bonus_credit_tracker_collection()
        tweets_collection = get_tweets_collection()
        query_collection = get_query_new_collection()
        
        if any(collection is None for collection in [bonus_collection, tweets_collection, query_collection]):
            st.error("Failed to connect to one or more collections")
            return []
        
        # Step 1: Get all task_ids from bonus_credit_tracker with type: "feedback"
        feedback_task_ids = []
        bonus_docs = bonus_collection.find({"type": "feedback"}, {"task_id": 1})
        for doc in bonus_docs:
            if doc and "task_id" in doc and doc["task_id"]:
                feedback_task_ids.append(doc["task_id"])
        
        if not feedback_task_ids:
            st.info("No feedback task_ids found in bonus_credit_tracker")
            return []
        
        # Step 2: Find replied tweets that have these task_ids in queued_fact_check_tasks
        replied_tweets = []
        tweets_cursor = tweets_collection.find(
            {"stage": "replied", "queued_fact_check_tasks": {"$exists": True, "$ne": None}},
            {
                "created_at": 1,
                "mention_tweet": 1,
                "parent_tweet": 1,
                "tagger.username": 1,
                "queued_fact_check_tasks": 1,
                "fact_check_summary": 1
            }
        )
        
        for tweet_doc in tweets_cursor:
            if not tweet_doc or "queued_fact_check_tasks" not in tweet_doc:
                continue
                
            tweet_task_ids = []
            queued_tasks = tweet_doc.get("queued_fact_check_tasks", [])
            
            # Check if queued_fact_check_tasks exists and is a list
            if isinstance(queued_tasks, list):
                for task in queued_tasks:
                    if (isinstance(task, dict) and 
                        "task_id" in task and 
                        task["task_id"] and 
                        task["task_id"] in feedback_task_ids):
                        tweet_task_ids.append(task["task_id"])
            
            if tweet_task_ids:
                replied_tweets.append({
                    "tweet_doc": tweet_doc,
                    "matching_task_ids": tweet_task_ids
                })
        
        # Step 3: Get feedbacks from query_new for all matching task_ids
        all_task_ids = []
        for tweet_data in replied_tweets:
            all_task_ids.extend(tweet_data["matching_task_ids"])
        
        # Remove duplicates
        unique_task_ids = list(set(all_task_ids))
        
        # Get feedbacks from query_new
        feedback_docs = {}
        if unique_task_ids:
            query_cursor = query_collection.find(
                {"task_id": {"$in": unique_task_ids}},
                {"task_id": 1, "feedbacks": 1, "dislikes": 1}
            )
            
            for doc in query_cursor:
                if doc and "task_id" in doc:
                    feedback_docs[doc["task_id"]] = doc
        
        # Step 4: Create final mapping
        tweet_feedback_mappings = []
        for tweet_data in replied_tweets:
            tweet_doc = tweet_data["tweet_doc"]
            tweet_feedbacks = []
            
            for task_id in tweet_data["matching_task_ids"]:
                if task_id in feedback_docs:
                    feedback_doc = feedback_docs[task_id]
                    tweet_feedbacks.append({
                        "task_id": task_id,
                        "feedbacks": feedback_doc.get("feedbacks", []),
                        "dislikes": feedback_doc.get("dislikes", 0)
                    })
            
            if tweet_feedbacks:
                tweet_feedback_mappings.append({
                    "tweet": {
                        "created_at": tweet_doc.get("created_at"),
                        "mention_tweet": tweet_doc.get("mention_tweet", ""),
                        "parent_tweet": tweet_doc.get("parent_tweet", ""),
                        "tagger_username": tweet_doc.get("tagger", {}).get("username", "Unknown"),
                        "fact_check_summary": tweet_doc.get("fact_check_summary", "")
                    },
                    "feedbacks": tweet_feedbacks
                })
        
        # Single concise status message
        st.success(f"Found {len(tweet_feedback_mappings)} replied tweets with {len(feedback_docs)} feedback tasks")
        return tweet_feedback_mappings
        
    except Exception as e:
        st.error(f"Error in moderator feedback mapping: {str(e)}")
        return []


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
        # Histogram of response times with fixed 5-minute intervals (0-60 minutes)
        filtered_data = stats['data'][stats['data']['response_time_minutes'] <= 60]
        
        # Create bins with 5-minute intervals (0-5, 5-10, 10-15, ..., 55-60)
        bin_size = 5
        max_time = 60
        bins = list(range(0, max_time + bin_size, bin_size))
        
        # Create histogram using manual bins
        hist_data, bin_edges = np.histogram(filtered_data['response_time_minutes'], bins=bins)
        
        # Create bin labels (e.g., "0-5", "5-10", etc.)
        bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
        
        # Create bar chart with fixed 5-minute intervals
        fig_hist = go.Figure(data=[
            go.Bar(
                x=bin_labels,
                y=hist_data,
                marker=dict(
                    color='#3b82f6',
                    line=dict(width=1, color='white')
                )
            )
        ])
        
        fig_hist.update_layout(
            title="Response Time Distribution (5-minute intervals)",
            xaxis_title="Response Time (minutes)",
            yaxis_title="Number of Tweets",
            showlegend=False,
            xaxis=dict(tickangle=-45)
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot of response times (filtered to 60 minutes max for better detail)
        filtered_data_box = stats['data'][stats['data']['response_time_minutes'] <= 60]
        
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
    
    # Top Bot Taggers section
    st.subheader("Most Frequent Bot Taggers")
    
    with st.spinner("Analyzing top taggers..."):
        # Get top users who tagged the bot most
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        pipeline = [
            {
                # Look out for tweets that are in the last n days (slider) and tweets that have either a username or id (taken from tagger object)
                "$match": {
                    "created_at": {"$gte": cutoff_date},
                    "$or": [
                        {"tagger.username": {"$exists": True, "$ne": ""}},
                        {"tagger.id": {"$exists": True, "$ne": ""}}
                    ]
                }
            },
            {
                "$project": {
                    "username": {
                        "$cond": [
                            {
                            "$and": [
                                {"$ne": ["$tagger.username", None]},  # Not null
                                {"$ne": ["$tagger.username", ""]},    # Not empty string
                                # {"$gt": [{"$strLenCP": {"$ifNull": ["$tagger.username", ""]}}, 0]}  # Has length > 0
                            ]
                            },
                            "$tagger.username",
                            {
                            "$cond": [
                                {
                                "$and": [
                                    {"$ne": ["$tagger.id", None]},
                                    {"$ne": ["$tagger.id", ""]}
                                ]
                                },
                                {"$toString": "$tagger.id"},   # Convert to string
                                "Unknown"                      # Last resort
                            ]
                            }
                        ]
                        },
                    "created_at": 1,
                    "reply_timestamp": 1,
                    "stage": 1,
                    "user_intent.intent": 1,
                    "parent_tweet": 1,
                    "mention_tweet": 1,
                    "replied": 1,
                    "response_time_minutes": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$reply_timestamp", None]},
                                    {"$ne": ["$created_at", None]}
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
            },
            {
                "$group": {
                    "_id": "$username",
                    "total_tags": {"$sum": 1},
                    "tweets": {
                        "$push": {
                            "created_at": "$created_at",
                            "stage": "$stage",
                            "intent": "$user_intent.intent",
                            "parent_tweet": "$parent_tweet",
                            "mention_tweet": "$mention_tweet",
                            "replied": "$replied",
                            "response_time": "$response_time_minutes" # previously copmuted response time used as response time field
                        }
                    }
                }
            },
            {
                "$sort": {"total_tags": -1}
            },
            {
                "$limit": 10
            }
        ]
        
        top_taggers = list(collection.aggregate(pipeline))
        
        if top_taggers:
            # Create summary table
            summary_data = []
            for tagger in top_taggers:
                summary_data.append({
                    "Username/User ID": tagger["_id"] if tagger["_id"] else "Unknown",
                    "Number of Tweets": tagger["total_tags"]
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            # Show detailed tweets for each top tagger
            st.subheader("Detailed Tweet History")
            
            # Tweets per page dropdown
            tweets_per_page = st.selectbox(
                "Tweets per page:",
                options=[1, 5, 10, 20],
                index=1,  # Default to 5
                key="tweets_per_page"
            )
            
            for idx, tagger in enumerate(top_taggers):
                username = tagger["_id"] if tagger["_id"] else "Unknown"
                total_tweets = tagger["total_tags"]
                
                with st.expander(f"#{idx+1} - {username} ({total_tweets} tweets)"):
                    # Get all tweets and sort by created_at descending
                    tweets = tagger["tweets"]
                    tweets_sorted = sorted(
                        tweets, 
                        key=lambda x: x.get("created_at") if x.get("created_at") else datetime.min.replace(tzinfo=timezone.utc),
                        reverse=True
                    )
                    
                    # Pagination
                    total_pages = (len(tweets_sorted) + tweets_per_page - 1) // tweets_per_page  # Ceiling division
                    
                    if total_pages > 0:
                        # Create unique key for each user's pagination
                        page_key = f"page_{idx}_{username}"
                        
                        # Initialize session state for this user if not exists
                        if page_key not in st.session_state:
                            st.session_state[page_key] = 1
                        
                        # Pagination controls
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col1:
                            if st.button("◀ Previous", key=f"prev_{idx}", disabled=st.session_state[page_key] <= 1):
                                st.session_state[page_key] -= 1
                                st.rerun()
                        
                        with col2:
                            st.write(f"Page {st.session_state[page_key]} of {total_pages}")
                        
                        with col3:
                            if st.button("Next ▶", key=f"next_{idx}", disabled=st.session_state[page_key] >= total_pages):
                                st.session_state[page_key] += 1
                                st.rerun()
                        
                        # Calculate slice for current page
                        start_idx = (st.session_state[page_key] - 1) * tweets_per_page
                        end_idx = start_idx + tweets_per_page
                        tweets_to_display = tweets_sorted[start_idx:end_idx]
                    else:
                        tweets_to_display = []
                    
                    if tweets_to_display:
                        # Calculate the actual tweet number based on page
                        start_num = (st.session_state[page_key] - 1) * tweets_per_page + 1
                        
                        for i, tweet in enumerate(tweets_to_display, start_num):
                            st.markdown(f"**Tweet {i}**")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                created_at = tweet.get("created_at")
                                if created_at:
                                    st.write(f"📅 **Date:** {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                                
                                intent = tweet.get("intent")
                                if intent:
                                    st.write(f"**Intent:** {intent}")
                                
                                stage = tweet.get("stage")
                                if stage:
                                    st.write(f"**Stage:** {stage}")
                            
                            with col2:
                                replied = tweet.get("replied")
                                if replied:
                                    st.write(f"**Replied:** Yes")
                                else:
                                    st.write(f"**Replied:** No")
                                
                                response_time = tweet.get("response_time")
                                if response_time is not None:
                                    st.write(f"⏱️ **Response Time:** {response_time:.1f} min")
                            
                            mention_tweet = tweet.get("mention_tweet")
                            if mention_tweet:
                                st.write(f"**Mention Tweet:** {mention_tweet[:500]}")
                            
                            parent_tweet = tweet.get("parent_tweet")
                            if parent_tweet:
                                st.write(f"**Parent Tweet:** {parent_tweet[:500]}")
                            
                            st.markdown("---")
                    else:
                        st.info("No tweet details available")
        else:
            st.info("No tagger data available for the selected period.")
    
    # Moderator Feedback Analytics Section
    st.subheader("Moderator Feedback from Replied Tweets")
    
    with st.spinner("Loading moderator feedback data..."):
        feedback_data = get_moderator_feedback_mapping()
    
    if feedback_data:
        # Calculate summary statistics
        total_tweets = len(feedback_data)
        total_feedbacks = sum(len(tweet_data["feedbacks"]) for tweet_data in feedback_data)
        total_task_ids = sum(len(tweet_data["feedbacks"]) for tweet_data in feedback_data)
        
        # Collect all feedbacks for analysis
        all_feedbacks = []
        all_reasons = []
        all_comments = []
        users_with_feedback = set()
        
        for tweet_data in feedback_data:
            for feedback_group in tweet_data["feedbacks"]:
                feedbacks_array = feedback_group.get("feedbacks", [])
                for feedback in feedbacks_array:
                    all_feedbacks.append(feedback)
                    
                    # Extract reasons (ignore "Share" as requested)
                    if "reasons" in feedback and isinstance(feedback["reasons"], list):
                        non_share_reasons = [r for r in feedback["reasons"] if r.lower() != "share"]
                        all_reasons.extend(non_share_reasons)
                    
                    # Extract comments
                    if "comments" in feedback and feedback["comments"]:
                        all_comments.append(feedback["comments"])
                    
                    # Extract user emails
                    if "user_email" in feedback and feedback["user_email"]:
                        users_with_feedback.add(feedback["user_email"])
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Tweets with Feedback",
                value=total_tweets,
                help="Number of replied tweets that received moderator feedback"
            )
        
        with col2:
            st.metric(
                label="Total Feedbacks",
                value=len(all_feedbacks),
                help="Total number of feedback entries received"
            )
        
        with col3:
            st.metric(
                label="Active Moderators",
                value=len(users_with_feedback),
                help="Number of unique users who provided feedback"
            )
        
        with col4:
            st.metric(
                label="Comments Provided",
                value=len(all_comments),
                help="Number of feedbacks that included written comments"
            )
        
        # Feedback Analysis Charts
        if all_reasons:            
            col1, col2 = st.columns(2)
            
            with col1:
                # Reasons distribution pie chart
                reason_counts = {}
                for reason in all_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                if reason_counts:
                    
                    reasons_df = pd.DataFrame(list(reason_counts.items()), columns=['Reason', 'Count'])
                    reasons_df = reasons_df.sort_values('Count', ascending=False)
                    
                    fig_reasons = px.pie(
                        reasons_df, 
                        values='Count', 
                        names='Reason',
                        title="Feedback Reasons Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_reasons.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_reasons, use_container_width=True)
            
            with col2:
                # Top reasons bar chart
                if reason_counts:
                    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    reasons_df = pd.DataFrame(top_reasons, columns=['Reason', 'Count'])
                    
                    fig_bar = px.bar(
                        reasons_df,
                        x='Count',
                        y='Reason',
                        orientation='h',
                        title="Top 10 Feedback Reasons",
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        # Top Discussed Tweets
        st.subheader("🔥 Most Discussed Tweets")
        
        # Calculate feedback count per tweet
        tweet_feedback_counts = []
        for tweet_data in feedback_data:
            tweet_info = tweet_data["tweet"]
            feedback_count = sum(len(feedback_group.get("feedbacks", [])) for feedback_group in tweet_data["feedbacks"])
            
            tweet_feedback_counts.append({
                "Tweet": tweet_info.get("mention_tweet", "N/A")[:50] + "..." if len(tweet_info.get("mention_tweet", "")) > 50 else tweet_info.get("mention_tweet", "N/A"),
                "Parent Tweet": tweet_info.get("parent_tweet", "N/A")[:50] + "..." if len(tweet_info.get("parent_tweet", "")) > 50 else tweet_info.get("parent_tweet", "N/A"),
                "Tagger": tweet_info.get("tagger_username", "Unknown"),
                "Feedback Count": feedback_count,
                "Created At": tweet_info.get("created_at", "N/A")
            })
        
        # Sort by feedback count and show top 10
        tweet_feedback_counts.sort(key=lambda x: x["Feedback Count"], reverse=True)
        top_tweets_df = pd.DataFrame(tweet_feedback_counts[:10])
        
        if not top_tweets_df.empty:
            st.dataframe(
                top_tweets_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No tweet feedback data available for display.")
        
        # Detailed Feedback Explorer
        st.subheader("🔍 Detailed Feedback Explorer")
        
        with st.expander("View All Tweet Feedbacks", expanded=False):
            if feedback_data:
                # Pagination
                items_per_page = 5
                total_pages = (len(feedback_data) + items_per_page - 1) // items_per_page
                
                if total_pages > 1:
                    page = st.selectbox("Select Page", range(1, total_pages + 1), key="feedback_page")
                else:
                    page = 1
                
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(feedback_data))
                
                st.write(f"Showing tweets {start_idx + 1}-{end_idx} of {len(feedback_data)}")
                
                for i in range(start_idx, end_idx):
                    tweet_data = feedback_data[i]
                    tweet_info = tweet_data["tweet"]
                    
                    st.write(f"**Tweet {i + 1}:**")
                    st.write(f"- **Mention Tweet:** {tweet_info.get('mention_tweet', 'N/A')}")
                    st.write(f"- **Parent Tweet:** {tweet_info.get('parent_tweet', 'N/A')}")
                    st.write(f"- **Tagger:** {tweet_info.get('tagger_username', 'Unknown')}")
                    st.write(f"- **Created:** {tweet_info.get('created_at', 'N/A')}")
                    
                    # Show feedbacks for this tweet
                    for j, feedback_group in enumerate(tweet_data["feedbacks"]):
                        st.write(f"  **Task ID {j + 1}:** {feedback_group.get('task_id', 'N/A')}")
                        st.write(f"  **Dislikes:** {feedback_group.get('dislikes', 0)}")
                        
                        feedbacks_array = feedback_group.get("feedbacks", [])
                        if feedbacks_array:
                            st.write("  **Feedbacks:**")
                            for k, feedback in enumerate(feedbacks_array):
                                st.write(f"    Feedback {k + 1}:")
                                if "user_email" in feedback:
                                    st.write(f"      - User: {feedback['user_email']}")
                                if "reasons" in feedback and feedback["reasons"]:
                                    non_share_reasons = [r for r in feedback["reasons"] if r.lower() != "share"]
                                    if non_share_reasons:
                                        st.write(f"      - Reasons: {', '.join(non_share_reasons)}")
                                if "comments" in feedback and feedback["comments"]:
                                    st.write(f"      - Comments: {feedback['comments']}")
                        else:
                            st.write("  No feedbacks available")
                    
                    st.write("---")
            else:
                st.info("No feedback data available.")
    else:
        st.info("No moderator feedback data available.")

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
