"""
Moderator Feedback Analytics view for the Facticity dashboard.
Comprehensive analysis of all moderator feedback across the platform.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
import numpy as np

from dbutils.DocumentDB import document_db_web2, document_db_web3
from utils.chart_utils import generate_chart


def get_query_new_collection():
    """
    Get connection to the query_new collection.
    
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


def get_comprehensive_feedback_data():
    """
    Get comprehensive feedback data by mapping:
    1. bonus_credit_tracker (type: "feedback") → task_ids
    2. query_new → feedbacks for all matching task_ids
    
    This covers ALL feedback in the system, not just Twitter-specific ones.
    
    Returns:
        dict: Comprehensive feedback data with statistics and raw feedbacks
    """
    try:
        # Get collections
        bonus_collection = get_bonus_credit_tracker_collection()
        query_collection = get_query_new_collection()
        
        if any(collection is None for collection in [bonus_collection, query_collection]):
            st.error("Failed to connect to one or more collections")
            return None
        
        # Get all task_ids from bonus_credit_tracker with type: "feedback" and fetch feedback data
        with st.spinner("🔍 Loading moderator feedback data..."):
            feedback_task_ids = []
            bonus_docs = bonus_collection.find({"type": "feedback"}, {"task_id": 1, "created_at": 1})
            
            for doc in bonus_docs:
                if doc and "task_id" in doc and doc["task_id"]:
                    feedback_task_ids.append({
                        "task_id": doc["task_id"],
                        "created_at": doc.get("created_at")
                    })
            
            if not feedback_task_ids:
                st.warning("No feedback task_ids found in bonus_credit_tracker")
                return None
            
            # Get feedbacks from query_new for all task_ids
            task_ids_list = [item["task_id"] for item in feedback_task_ids]
            
            feedback_docs = {}
            query_cursor = query_collection.find(
                {"task_id": {"$in": task_ids_list}},
                {
                    "task_id": 1, 
                    "feedbacks": 1,
                    "created_at": 1,
                    "query": 1,
                    "response": 1,
                    "result": 1,
                    "overall_assessment": 1
                }
            )
            
            for doc in query_cursor:
                if doc and "task_id" in doc:
                    feedback_docs[doc["task_id"]] = doc
            
            # Process and analyze all feedbacks
        
        all_feedbacks = []
        all_reasons = []
        all_comments = []
        users_with_feedback = set()
        task_feedback_mapping = []
        
        # Process each task_id
        for task_info in feedback_task_ids:
            task_id = task_info["task_id"]
            task_created_at = task_info.get("created_at")
            
            if task_id in feedback_docs:
                feedback_doc = feedback_docs[task_id]
                feedbacks_array = feedback_doc.get("feedbacks", [])
                
                # Process individual feedbacks
                for feedback in feedbacks_array:
                    all_feedbacks.append(feedback)
                    
                    # Extract reasons (excluding "Share" as per existing logic)
                    if "reasons" in feedback and isinstance(feedback["reasons"], list):
                        non_share_reasons = [r for r in feedback["reasons"] if r.lower() != "share"]
                        all_reasons.extend(non_share_reasons)
                    
                    # Extract comments
                    if "comments" in feedback and feedback["comments"]:
                        all_comments.append(feedback["comments"])
                    
                    # Extract user emails
                    if "user_email" in feedback and feedback["user_email"]:
                        users_with_feedback.add(feedback["user_email"])
                
                # Create task mapping
                task_feedback_mapping.append({
                    "task_id": task_id,
                    "task_created_at": task_created_at,
                    "query": feedback_doc.get("query", ""),
                    "response": feedback_doc.get("response", ""),
                    "result": feedback_doc.get("result", {}),
                    "overall_assessment": feedback_doc.get("overall_assessment", ""),
                    "feedback_count": len(feedbacks_array),
                    "feedbacks": feedbacks_array
                })
        
        # Calculate comprehensive statistics
        stats = {
            "total_fact_check_tasks": len(feedback_task_ids),
            "tasks_with_feedback": len(feedback_docs),
            "total_feedbacks": len(all_feedbacks),
            "unique_moderators": len(users_with_feedback),
            "total_comments": len(all_comments)
        }
        
        # Reason analysis
        reason_counts = Counter(all_reasons)
        
        # User activity analysis
        user_feedback_counts = Counter()
        for feedback in all_feedbacks:
            if "user_email" in feedback and feedback["user_email"]:
                user_feedback_counts[feedback["user_email"]] += 1
        
        # Time-based analysis
        task_feedback_mapping.sort(key=lambda x: x.get("task_created_at") or datetime.min.replace(tzinfo=timezone.utc))
        
        return {
            "stats": stats,
            "reason_counts": reason_counts,
            "user_feedback_counts": user_feedback_counts,
            "task_feedback_mapping": task_feedback_mapping,
            "all_feedbacks": all_feedbacks,
            "all_comments": all_comments,
            "users_with_feedback": users_with_feedback
        }
        
    except Exception as e:
        st.error(f"Error in comprehensive feedback analysis: {str(e)}")
        return None


def show_moderator_feedback_analytics():
    """
    Display the comprehensive Moderator Feedback Analytics view.
    """
    st.title("Moderator Feedback Analytics")
    
    # Set default values for all options
    show_raw_data = False
    show_detailed_tasks = False
    show_reason_analysis = True
    show_user_activity = True
    show_time_trends = True
    show_comment_analysis = True
    
    # Get comprehensive feedback data
    with st.spinner("🔄 Loading comprehensive feedback data..."):
        feedback_data = get_comprehensive_feedback_data()
    
    if not feedback_data:
        st.error("Failed to load feedback data. Please check database connections.")
        return
    
    stats = feedback_data["stats"]
    
    # Key Metrics Overview
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Fact-check Tasks",
            value=f"{stats['total_fact_check_tasks']:,}",
            help="Number of fact-check tasks with moderator feedback"
        )
    
    with col2:
        st.metric(
            label="Total Feedbacks",
            value=f"{stats['total_feedbacks']:,}",
            help="Total number of likes, dislikes and shares"
        )
    
    with col3:
        st.metric(
            label="👥 Active Moderators",
            value=f"{stats['unique_moderators']:,}",
            help="Unique users who provided feedback"
        )
    
    with col4:
        st.metric(
            label="💬 Comments Provided",
            value=f"{stats['total_comments']:,}",
            help="Number of feedbacks that included written comments"
        )
    
    # Reason Analysis Charts
    if show_reason_analysis:
        st.subheader("Feedback type analysis")
        
        # Define reason categories
        LIKE_REASONS = ['Balanced assessment', 'Helpful information', 'Clear explanation', 'Good sources']
        DISLIKE_REASONS = ['Harmful or offensive', 'Not helpful', 'Out of date', 'Not factually correct']
        
        # Process reasons data
        reason_counts = feedback_data["reason_counts"]
        
        if reason_counts:
            # Filter out "Share" and categorize reasons
            filtered_reasons = {reason: count for reason, count in reason_counts.items() if reason != "Share"}
            
            if filtered_reasons:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie Chart - Like vs Dislike proportions (count feedbacks, not reasons)
                    like_feedbacks = 0
                    dislike_feedbacks = 0
                    
                    for feedback in feedback_data["all_feedbacks"]:
                        if "reasons" in feedback and isinstance(feedback["reasons"], list):
                            feedback_reasons = [r for r in feedback["reasons"] if r.lower() != "share"]
                            if feedback_reasons:
                                # Check if any reason is positive or negative
                                has_positive = any(reason in LIKE_REASONS for reason in feedback_reasons)
                                has_negative = any(reason in DISLIKE_REASONS for reason in feedback_reasons)
                                
                                if has_positive and not has_negative:
                                    like_feedbacks += 1
                                elif has_negative and not has_positive:
                                    dislike_feedbacks += 1
                                # Mixed feedbacks are not counted in either category
                    
                    if like_feedbacks > 0 or dislike_feedbacks > 0:
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['👍 Like', '👎 Dislike'],
                            values=[like_feedbacks, dislike_feedbacks],
                            hole=0.3,
                            marker_colors=['#60a5fa', '#ef4444']
                        )])
                        
                        fig_pie.update_layout(
                            title="Likes and Dislikes",
                            showlegend=True,
                            annotations=[dict(text=f'Total<br>{like_feedbacks + dislike_feedbacks}', x=0.5, y=0.5, font_size=16, showarrow=False)],
                            font=dict(size=16, family='Arial Black')
                        )
                        
                        # Update pie chart text to be bold and show percentages
                        fig_pie.update_traces(
                            textinfo='percent+label',
                            textfont_size=16,
                            textfont_color='white',
                            textfont_family='Arial Black'
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Horizontal Bar Chart - Top reasons
                    sorted_reasons = sorted(filtered_reasons.items(), key=lambda x: x[1], reverse=True)
                    reasons_df = pd.DataFrame(sorted_reasons[:8], columns=['Reason', 'Count'])
                    
                    # Color coding based on like/dislike
                    colors = []
                    for reason in reasons_df['Reason']:
                        if reason in LIKE_REASONS:
                            colors.append('#60a5fa')  # Light blue for positive
                        elif reason in DISLIKE_REASONS:
                            colors.append('#ef4444')  # Red for negative
                        else:
                            colors.append('#6b7280')  # Gray for others
                    
                    fig_bar = go.Figure(data=[go.Bar(
                        x=reasons_df['Count'],
                        y=reasons_df['Reason'],
                        orientation='h',
                        marker_color=colors,
                        text=reasons_df['Count'],
                        textposition='auto',
                        textfont=dict(size=14, color='white', family='Arial Black')
                    )])
                    
                    fig_bar.update_layout(
                        title="Feedback Reasons",
                        xaxis_title="Count",
                        yaxis_title="Reason",
                        yaxis={'categoryorder': 'total ascending'},
                        height=400
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Summary statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    if like_feedbacks + dislike_feedbacks > 0:
                        positive_ratio = (like_feedbacks / (like_feedbacks + dislike_feedbacks)) * 100
                        st.metric("Like to Dislike Ratio", f"{positive_ratio:.1f}%")
                
                with col2:
                    most_common_reason = max(filtered_reasons.items(), key=lambda x: x[1])
                    st.metric("Most Common", f"{most_common_reason[0]}")
                
                # Reason Correlation Heatmap
                st.subheader("Reason Correlation Analysis")
                
                # Create correlation matrix for reasons that appear together
                reason_pairs = {}
                all_reason_names = list(filtered_reasons.keys())
                
                # Count co-occurrences of reasons in the same feedback
                for feedback in feedback_data["all_feedbacks"]:
                    if "reasons" in feedback and isinstance(feedback["reasons"], list):
                        feedback_reasons = [r for r in feedback["reasons"] if r.lower() != "share" and r in all_reason_names]
                        
                        # Count pairs of reasons that appear together
                        for i in range(len(feedback_reasons)):
                            for j in range(i + 1, len(feedback_reasons)):
                                reason1, reason2 = feedback_reasons[i], feedback_reasons[j]
                                # Create consistent pair key (alphabetical order)
                                pair_key = tuple(sorted([reason1, reason2]))
                                reason_pairs[pair_key] = reason_pairs.get(pair_key, 0) + 1
                
                if reason_pairs:
                    # Create correlation matrix
                    correlation_matrix = []
                    reason_labels = []
                    
                    for reason in all_reason_names:
                        correlation_row = []
                        for other_reason in all_reason_names:
                            if reason == other_reason:
                                correlation_row.append(1.0)  # Perfect correlation with self
                            else:
                                pair_key = tuple(sorted([reason, other_reason]))
                                count = reason_pairs.get(pair_key, 0)
                                # Normalize by the minimum of individual counts
                                min_count = min(filtered_reasons[reason], filtered_reasons[other_reason])
                                correlation = count / min_count if min_count > 0 else 0
                                correlation_row.append(correlation)
                        correlation_matrix.append(correlation_row)
                        reason_labels.append(reason)
                    
                    # Create heatmap
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=correlation_matrix,
                        x=reason_labels,
                        y=reason_labels,
                        colorscale='Blues',
                        showscale=True,
                        colorbar=dict(title="Correlation Strength")
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Reason Co-occurrence Heatmap",
                        xaxis_title="Reason",
                        yaxis_title="Reason",
                        height=500
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Show top reason pairs
                    st.subheader("Top Reason Combinations")
                    top_pairs = sorted(reason_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    for i, (pair, count) in enumerate(top_pairs, 1):
                        reason1, reason2 = pair
                        st.write(f"{i}. **{reason1}** + **{reason2}**: {count} times")
                else:
                    st.info("No reason correlations found (reasons don't appear together in feedbacks)")
            else:
                st.info("No feedback reasons found (excluding 'Share')")
        else:
            st.warning("No reason data available for analysis")
    
    # User Activity Analysis
    if show_user_activity:
        st.subheader("👥 User Activity Analysis")
        
        user_feedback_counts = feedback_data["user_feedback_counts"]
        
        if user_feedback_counts:
            # Horizontal Bar Chart - Top 20 most active moderators
            top_users = user_feedback_counts.most_common(20)
            users_df = pd.DataFrame(top_users, columns=['User Email', 'Feedback Count'])
            
            fig_users = px.bar(
                users_df,
                x='Feedback Count',
                y='User Email',
                orientation='h',
                title="Top 20 Most Active Moderators",
                color='Feedback Count',
                color_continuous_scale='Blues'
            )
            
            fig_users.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=600
            )
            
            st.plotly_chart(fig_users, use_container_width=True)
            
            # User activity summary metrics
            col1, col2 = st.columns(2)
            
            with col1:
                feedback_counts = list(user_feedback_counts.values())
                avg_feedback_per_user = sum(feedback_counts) / len(feedback_counts) if feedback_counts else 0
                st.metric("Avg Feedbacks per User", f"{avg_feedback_per_user:.1f}")
            
            with col2:
                if feedback_counts:
                    max_feedback = max(feedback_counts) if feedback_counts else 0
                    most_active_user = user_feedback_counts.most_common(1)[0][0] if user_feedback_counts else "N/A"
                    st.metric(
                        "Most Active User", 
                        f"{max_feedback:,} feedbacks",
                        help=f"Most feedbacks from one user"
                    )
                else:
                    st.metric("Most Active User", "0 feedbacks")
        else:
            st.warning("No user activity data available")
    
    # Comment Analysis
    if show_comment_analysis:
        st.subheader("💬 Comment Analysis")
        
        all_comments = feedback_data["all_comments"]
        
        if all_comments:
            col1, col2 = st.columns(2)
            
            with col1:
                # Word Cloud - Most frequent words in comments
                from collections import Counter
                import re
                
                # Combine all comments and extract words
                all_text = ' '.join(all_comments).lower()
                # Remove common words and extract meaningful words
                words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
                
                # Filter out common stop words
                stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use', 'very', 'when', 'with', 'this', 'that', 'they', 'have', 'been', 'from', 'will', 'your', 'good', 'nice', 'well', 'just', 'like', 'make', 'more', 'some', 'time', 'what', 'know', 'take', 'than', 'them', 'only', 'other', 'come', 'could', 'first', 'here', 'long', 'look', 'made', 'many', 'over', 'such', 'through', 'where', 'much', 'before', 'right', 'should', 'these', 'think', 'also', 'back', 'after', 'work', 'life', 'where', 'much', 'before', 'right', 'should', 'these', 'think', 'also', 'back', 'after', 'work', 'life'}
                
                filtered_words = [word for word in words if word not in stop_words]
                word_counts = Counter(filtered_words)
                
                if word_counts:
                    # Get top 20 words
                    top_words = word_counts.most_common(20)
                    words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
                    
                    fig_words = px.bar(
                        words_df,
                        x='Count',
                        y='Word',
                        orientation='h',
                        title="Most Frequent Words in Comments",
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    
                    fig_words.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400
                    )
                    
                    st.plotly_chart(fig_words, use_container_width=True)
                else:
                    st.info("No meaningful words found in comments")
            
            with col2:
                # Bar Chart - Comment length distribution
                comment_lengths = [len(comment) for comment in all_comments]
                
                fig_length = go.Figure(data=[go.Histogram(
                    x=comment_lengths,
                    nbinsx=20,
                    marker_color='#60a5fa',
                    opacity=0.7
                )])
                
                fig_length.update_layout(
                    title="Comment Length Distribution",
                    xaxis_title="Comment Length (characters)",
                    yaxis_title="Number of Comments",
                    height=400
                )
                
                st.plotly_chart(fig_length, use_container_width=True)
            
            # Comment analysis summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Comments", f"{len(all_comments):,}")
            
            with col2:
                avg_length = sum(comment_lengths) / len(comment_lengths) if comment_lengths else 0
                st.metric("Avg Comment Length", f"{avg_length:.0f} chars")
            
            with col3:
                longest_comment = max(comment_lengths) if comment_lengths else 0
                st.metric("Longest Comment", f"{longest_comment} chars")
            
            # Show longest comments with context
            st.subheader("Longest Comments")
            if all_comments:
                # Create a mapping of comments to their task data
                task_feedback_mapping = feedback_data["task_feedback_mapping"]
                comment_to_task = {}
                for task in task_feedback_mapping:
                    for feedback in task.get("feedbacks", []):
                        if "comments" in feedback and feedback["comments"]:
                            comment_text = feedback["comments"]
                            if comment_text not in comment_to_task:
                                comment_to_task[comment_text] = task
                
                # Sort comments by length and get the longest ones
                sorted_comments = sorted(all_comments, key=len, reverse=True)
                longest_comments = sorted_comments[:5]  # Show top 5 longest
                
                for i, comment in enumerate(longest_comments, 1):
                    # Get task data for this comment
                    task_data = comment_to_task.get(comment, {})
                    
                    with st.container():
                        st.markdown(f"### Comment {i} ({len(comment)} characters)")
                        
                        # Display query
                        if task_data.get("query"):
                            st.markdown(f"**Query:** {task_data['query']}")
                        
                        # Display classification with color coding
                        result = task_data.get("result", {})
                        classification = result.get("Classification", "") if isinstance(result, dict) else ""
                        overall_assessment = result.get("overall_assessment", "") if isinstance(result, dict) else ""
                        
                        if classification:
                            # Color code classification
                            classification_lower = classification.lower()
                            if classification_lower == "true":
                                st.markdown(f'**Classification:** <span style="color: green; font-weight: bold;">✅ {classification.upper()}</span>', unsafe_allow_html=True)
                            elif classification_lower == "false":
                                st.markdown(f'**Classification:** <span style="color: red; font-weight: bold;">❌ {classification.upper()}</span>', unsafe_allow_html=True)
                            elif classification_lower in ["unverifiable", "uncertain"]:
                                st.markdown(f'**Classification:** <span style="color: orange; font-weight: bold;">⚠️ {classification.upper()}</span>', unsafe_allow_html=True)
                            else:
                                st.markdown(f"**Classification:** {classification}")
                        
                        # Display overall assessment
                        if overall_assessment:
                            st.markdown(f"**Overall Assessment:** {overall_assessment}")
                        
                        # Display the actual comment
                        st.markdown(f"**💬 Moderator Comment:** {comment}")
                        st.markdown("---")
            else:
                st.info("No comments available")
        else:
            st.warning("No comment data available")
    
    # Raw data explorer
    if show_raw_data:
        st.subheader("🔍 Raw Data Explorer")
        
        with st.expander("📋 Task-Feedback Mapping", expanded=False):
            if feedback_data["task_feedback_mapping"]:
                # Create a summary table
                summary_data = []
                for task_data in feedback_data["task_feedback_mapping"][:50]:  # Show first 50
                    summary_data.append({
                        "Task ID": task_data["task_id"],
                        "Feedback Count": task_data["feedback_count"],
                        "Dislikes": task_data["dislikes"],
                        "Query Preview": task_data["query"][:100] + "..." if len(task_data["query"]) > 100 else task_data["query"],
                        "Created At": task_data.get("task_created_at", "N/A")
                    })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
                
                if len(feedback_data["task_feedback_mapping"]) > 50:
                    st.info(f"Showing first 50 of {len(feedback_data['task_feedback_mapping'])} tasks")
            else:
                st.info("No task feedback data available")
        
        with st.expander("👥 User Activity Summary", expanded=False):
            if feedback_data["user_feedback_counts"]:
                user_data = []
                for user, count in feedback_data["user_feedback_counts"].most_common(20):
                    user_data.append({
                        "User Email": user,
                        "Feedback Count": count
                    })
                
                df_users = pd.DataFrame(user_data)
                st.dataframe(df_users, use_container_width=True)
            else:
                st.info("No user activity data available")
        
        with st.expander("📝 Sample Comments", expanded=False):
            if feedback_data["all_comments"]:
                for i, comment in enumerate(feedback_data["all_comments"][:10], 1):
                    st.write(f"**Comment {i}:** {comment}")
                    st.write("---")
                
                if len(feedback_data["all_comments"]) > 10:
                    st.info(f"Showing first 10 of {len(feedback_data['all_comments'])} comments")
            else:
                st.info("No comments available")
    
    # Detailed task analysis
    if show_detailed_tasks:
        st.subheader("🔍 Detailed Task Analysis")
        
        if feedback_data["task_feedback_mapping"]:
            # Pagination for detailed view
            items_per_page = 5
            total_pages = (len(feedback_data["task_feedback_mapping"]) + items_per_page - 1) // items_per_page
            
            if total_pages > 1:
                page = st.selectbox("Select Page", range(1, total_pages + 1), key="detailed_page")
            else:
                page = 1
            
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(feedback_data["task_feedback_mapping"]))
            
            st.write(f"Showing tasks {start_idx + 1}-{end_idx} of {len(feedback_data['task_feedback_mapping'])}")
            
            for i in range(start_idx, end_idx):
                task_data = feedback_data["task_feedback_mapping"][i]
                
                with st.expander(f"Task {i + 1}: {task_data['task_id']}", expanded=False):
                    st.write(f"**Task ID:** {task_data['task_id']}")
                    st.write(f"**Created At:** {task_data.get('task_created_at', 'N/A')}")
                    st.write(f"**Feedback Count:** {task_data['feedback_count']}")
                    st.write(f"**Dislikes:** {task_data['dislikes']}")
                    
                    if task_data['query']:
                        st.write(f"**Query:** {task_data['query']}")
                    
                    if task_data['response']:
                        st.write(f"**Response:** {task_data['response'][:200]}...")
                    
                    # Show feedbacks
                    if task_data['feedbacks']:
                        st.write("**Feedbacks:**")
                        for j, feedback in enumerate(task_data['feedbacks'], 1):
                            st.write(f"  Feedback {j}:")
                            if "user_email" in feedback:
                                st.write(f"    - User: {feedback['user_email']}")
                            if "reasons" in feedback and feedback["reasons"]:
                                non_share_reasons = [r for r in feedback["reasons"] if r.lower() != "share"]
                                if non_share_reasons:
                                    st.write(f"    - Reasons: {', '.join(non_share_reasons)}")
                            if "comments" in feedback and feedback["comments"]:
                                st.write(f"    - Comments: {feedback['comments']}")
                            st.write("    ---")
        else:
            st.info("No detailed task data available")


if __name__ == "__main__":
    show_moderator_feedback_analytics()