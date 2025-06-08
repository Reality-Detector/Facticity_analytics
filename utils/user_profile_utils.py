"""
User profiling utilities for generating and managing user interest profiles
based on query embeddings and IAB categories.
"""
import random
import pandas as pd
import numpy as np
import json
import os
import boto3
import pickle
import re
import streamlit as st
from datetime import datetime, timezone
from typing import Tuple, Dict, List, Any, Optional
from config import BLACKLIST_DOMAINS, BLACKLIST_EMAILS
from config import AWS_SECRET_KEY, AWS_ACCESS_KEY


AWS_REGION = "us-west-2"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
IAB_CATEGORIES_PATH = "data/email_segments/claude-iab-descriptors.json"
IAB_EMBEDDINGS_PATH = "data/email_segments/category_embeddings.pkl"
QUERY_EMBEDDINGS_PATH = "data/email_segments/query_embeddings.pkl"


def create_bedrock_client():
    """Create AWS Bedrock client"""
    try:
        return boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
    except Exception as e:
        st.error(f"Failed to connect to Bedrock: {str(e)}")
        return None


def clean_query_text(query: str) -> str:
    """Clean and normalize search query text"""
    if not isinstance(query, str):
        return ""
    cleaned = re.sub(r'[^\w\s]', '', query)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def load_iab_categories() -> Dict[str, str]:
    """Load IAB categories from JSON file"""
    try:
        with open(IAB_CATEGORIES_PATH, 'r') as f:
            categories = json.load(f)
        return categories
    except Exception as e:
        st.warning(f"Error loading IAB categories: {str(e)}")
        return {}


def load_category_embeddings() -> Optional[Dict[str, List[float]]]:
    """Load pre-computed category embeddings if available"""
    try:
        if os.path.exists(IAB_EMBEDDINGS_PATH):
            with open(IAB_EMBEDDINGS_PATH, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        return None


def save_category_embeddings(embeddings: Dict[str, List[float]]) -> None:
    """Save category embeddings to disk"""
    try:
        os.makedirs(os.path.dirname(IAB_EMBEDDINGS_PATH), exist_ok=True)
        with open(IAB_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        st.error(f"Error saving category embeddings: {str(e)}")


def generate_category_embeddings(bedrock_client, categories: Dict[str, str]) -> Dict[str, List[float]]:
    """Generate embeddings for all IAB categories"""
    cached = load_category_embeddings()
    if cached:
        return cached

    if not bedrock_client:
        st.error("Bedrock client not available")
        return {}

    embeddings = {}
    total_categories = len(categories)

    for i, (cat, description) in enumerate(categories.items()):
        try:
            payload = json.dumps({"inputText": description})
            response = bedrock_client.invoke_model(
                modelId=EMBEDDING_MODEL,
                body=payload,
                contentType="application/json",
                accept="application/json"
            )
            result = json.loads(response['body'].read())
            embeddings[cat] = result['embedding']

        except Exception as e:
            st.warning(f"Error generating embedding for {cat}: {str(e)}")

    save_category_embeddings(embeddings)
    return embeddings


def load_query_embeddings() -> Dict[str, List[float]]:
    """Load pre-computed query embeddings if available"""
    try:
        if os.path.exists(QUERY_EMBEDDINGS_PATH):
            with open(QUERY_EMBEDDINGS_PATH, 'rb') as f:
                return pickle.load(f)
        return {}
    except Exception as e:
        st.warning(f"Error loading query embeddings: {str(e)}")
        return {}


def save_query_embeddings(embeddings: Dict[str, List[float]]) -> None:
    """Save query embeddings to disk"""
    try:
        os.makedirs(os.path.dirname(QUERY_EMBEDDINGS_PATH), exist_ok=True)
        with open(QUERY_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        st.error(f"Error saving query embeddings: {str(e)}")


def fetch_user_queries_with_date_range(get_db_connection, start_date: datetime, end_date: datetime,
                                       max_users: int = 1000, max_q: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Fetch user queries from MongoDB with configurable date range - EXACT copy from working profiler"""
    try:
        # Get the client, then get the collection - exactly like the original
        client = get_db_connection()
        coll = client.get_database("facticity").get_collection("query_new")

        print(
            f"Fetching queries from {start_date.date()} to {end_date.date()}")

        email_conditions = [
            {"userEmail": {"$exists": True, "$ne": ""}},
            {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}
        ] + [
            {"userEmail": {
                "$not": {"$regex": f"@{re.escape(domain)}$", "$options": "i"}}}
            for domain in BLACKLIST_DOMAINS
        ]

        q_filter = {
            "timestamp": {"$gte": start_date.isoformat(), "$lt": end_date.isoformat()},
            "query": {"$exists": True},
            "$and": email_conditions
        }

        user_count = len(coll.distinct("userEmail", q_filter))
        print(f"Found {user_count} unique users with queries")

        pipeline = [
            {"$match": q_filter},
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$userEmail",
                "queries": {"$push": {"query": "$query", "timestamp": "$timestamp"}}
            }},
            {"$project": {
                "email": "$_id",
                "queries": {"$slice": ["$queries", max_q]}
            }}
        ]

        sampled_data = []
        unique_queries = set()
        random.seed(42)

        user_batch = list(coll.aggregate(pipeline, allowDiskUse=True))
        print(f"Sampled {len(user_batch)} users")
        for user in user_batch:
            email = user.get("email")
            cleaned_queries = []
            seen_queries = set()

            for item in user.get("queries", []):
                clean_q = clean_query_text(item.get("query", ""))
                if clean_q and clean_q not in seen_queries:
                    seen_queries.add(clean_q)
                    cleaned_queries.append({
                        "query": clean_q,
                        "userEmail": email,
                        "timestamp": item.get("timestamp")
                    })

            if len(cleaned_queries) > max_q:
                cleaned_queries.sort(
                    key=lambda x: x["timestamp"], reverse=True)
                sampled_queries = cleaned_queries[:max_q]
            else:
                sampled_queries = cleaned_queries

            sampled_data.extend(sampled_queries)
            unique_queries.update([q["query"] for q in sampled_queries])

        sampled_df = pd.DataFrame(sampled_data)
        if sampled_df.empty:
            print("No valid queries found after processing")
            return pd.DataFrame(), pd.DataFrame(), {"user_count": user_count}

        sampled_df["timestamp"] = pd.to_datetime(
            sampled_df["timestamp"], utc=True)
        unique_df = pd.DataFrame({"query": list(unique_queries)})

        print(
            f"Processed {len(sampled_df)} queries from {len(user_batch)} users")
        print(f"Found {len(unique_df)} unique queries")
        return unique_df, sampled_df, {"user_count": user_count}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), {"error": str(e)}


def categorize_queries(full_df: pd.DataFrame, query_emb_map: Dict[str, List[float]],
                       cat_emb: Dict[str, List[float]]) -> pd.DataFrame:
    """Categorize queries using cosine similarity between query and category embeddings"""
    if full_df.empty:
        return full_df

    def top3_categories(query: str) -> List[Tuple[str, float]]:
        """Calculate top 3 matching categories for a query"""
        emb = query_emb_map.get(query)
        if not isinstance(emb, list):
            return []
        emb_arr = np.array(emb)
        emb_norm = np.linalg.norm(emb_arr)
        if emb_norm == 0:
            return []
        sims = []
        for cat, cat_emb_vec in cat_emb.items():
            cat_arr = np.array(cat_emb_vec)
            cat_norm = np.linalg.norm(cat_arr)
            if cat_norm == 0:
                continue
            score = np.dot(emb_arr, cat_arr) / (emb_norm * cat_norm)
            sims.append((cat, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:3]

    df = full_df.copy()
    df["categories"] = df["query"].map(top3_categories)
    df["top_category"] = df["categories"].apply(
        lambda x: x[0][0] if x else None)
    return df


def generate_user_profiles(categorized_df: pd.DataFrame) -> Dict[str, Dict]:
    """Generate user profiles from categorized queries"""
    if categorized_df.empty:
        return {}

    profiles = {}
    for email, group in categorized_df.groupby("userEmail"):
        cat_scores = {}
        analysed_queries = []

        group_len = len(group)
        for i, row in enumerate(group.itertuples()):
            if not row.categories:
                continue
            analysed_queries.append(row.query)
            weight = 1.0 - (i / (group_len * 1.5)) if group_len > 1 else 1.0
            for main_cat, score in row.categories:
                cat_scores[main_cat] = cat_scores.get(
                    main_cat, 0) + score * weight

        # Normalize scores to ensure they sum to 1
        total_score = sum(cat_scores.values())
        if total_score > 0:
            normalized_scores = {
                k: v / total_score for k, v in cat_scores.items()}
        else:
            normalized_scores = {}

        top_topics = [c for c, _ in sorted(normalized_scores.items(),
                                           key=lambda x: x[1], reverse=True)][:5]

        profiles[email] = {
            "email": email,
            "total_queries": group_len,
            "analysed_queries": len(analysed_queries),
            "category_scores": normalized_scores,
            "top_topics": top_topics,
            "recent_queries": analysed_queries
        }
    return profiles


def generate_query_embeddings_batch(bedrock_client, new_queries: List[str]) -> Dict[str, List[float]]:
    """Generate embeddings for a batch of new queries with progress tracking"""
    embeddings = {}

    if not new_queries:
        return embeddings

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, query in enumerate(new_queries):
        try:
            payload = json.dumps({"inputText": query})
            response = bedrock_client.invoke_model(
                modelId=EMBEDDING_MODEL,
                body=payload,
                contentType="application/json",
                accept="application/json"
            )
            result = json.loads(response['body'].read())
            embeddings[query] = result['embedding']

            progress_bar.progress((i + 1) / len(new_queries))
            status_text.text(f"Processing query {i + 1}/{len(new_queries)}")

        except Exception as e:
            st.warning(f"Error generating embedding for query: {str(e)}")

    progress_bar.empty()
    status_text.empty()

    return embeddings
