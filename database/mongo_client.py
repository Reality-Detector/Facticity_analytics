"""
MongoDB client module for the Facticity dashboard.
Handles connections and queries to MongoDB.
Main client database is represented by DB_CONNECTION_STRING. API database starts with API.
query_new collection is queried here for data on query activity. 
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import BLACKLIST_EMAILS, BLACKLIST_DOMAINS, DB_CONNECTION_STRING
from dbutils.DocumentDB import document_db_web2, document_db_api, document_db_web3
from utils import normalize_url_for_mongo, denormalize_url_from_mongo

@st.cache_resource
def get_db_connection(db_string=DB_CONNECTION_STRING, max_retries=3):
    """
    Creates and returns a cached MongoDB connection to the query_new collection with error handling.
    
    Args:
        db_string: MongoDB connection string
        max_retries: Maximum number of connection attempts
    
    Returns:
        MongoDB collection: The query_new collection or None if connection fails
    """
    if not db_string:
        st.error("Database connection string not provided")
        return None

    for attempt in range(max_retries):
        try:
            # Create client with timeouts and connection pooling
            if db_string == "api":
                client = document_db_api.get_client()
            else:
                client = document_db_web2.get_client()


            # Test the connection with a simple ping
            client.admin.command('ping')

            # Return the collection
            collection = client["facticity"]["query_new"]

            # Test collection access
            collection.find_one()

            # st.success(f"✅ Connected to MongoDB (attempt {attempt + 1})")
            return collection

        except Exception as e:
            error_msg = str(e)
            st.warning(
                f"Connection attempt {attempt + 1}/{max_retries} failed: {error_msg[:100]}...")

            # If it's a DNS error, provide specific guidance
            if "DNS" in error_msg or "resolution" in error_msg:
                st.error(
                    "🌐 DNS Resolution Error - Check your internet connection or try a different network")

            if attempt < max_retries - 1:  # Don't sleep on last attempt
                import time
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            continue

    # All attempts failed
    st.error("❌ Could not connect to MongoDB after multiple attempts")
    st.info("💡 Try: Check internet connection, verify MongoDB Atlas IP whitelist, or contact admin")
    return None


def is_valid_user(email):
    """
    Checks if an email is valid (not blacklisted).
    
    Args:
        email: Email address to check
        
    Returns:
        bool: True if the email is valid, False otherwise
    """
    # Check for None, empty string, or non-string values
    if email is None or not isinstance(email, str) or email.strip() == '':
        return False

    # Try to get the domain safely
    try:
        if '@' in email:
            domain = email.split('@')[-1]
            return (email not in BLACKLIST_EMAILS) and (domain not in BLACKLIST_DOMAINS)
        else:
            return False
    except Exception:
        # If any error occurs during processing, consider it invalid
        return False


@st.cache_data(ttl=10800)
def fetch_query_data(exclude_blacklisted=False):
    """
    Fetches all query data with userEmail and timestamp.
    
    Args:
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        DataFrame: DataFrame with userEmail and timestamp columns
    """
    collection = get_db_connection()

    query_filter = {"userEmail": {"$exists": True},
                    "timestamp": {"$exists": True}}

    # Add email/domain exclusion if requested
    if exclude_blacklisted:
        # Create email exclusion condition
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}

        # Create domain exclusion conditions
        domain_exclusions = []
        for domain in BLACKLIST_DOMAINS:
            domain_exclusions.append(
                {"userEmail": {"$not": {"$regex": f"@{domain}$"}}})

        # Combine all conditions
        query_filter = {"$and": [query_filter,
                                 email_exclusion, *domain_exclusions]}

    data = collection.find(
        query_filter,
        {"userEmail": 1, "timestamp": 1, "_id": 0}
    )
    return pd.DataFrame(list(data))


@st.cache_data(ttl=10800)
def aggregate_daily_with_users(start_date, end_date, exclude_blacklisted=False, db_string=DB_CONNECTION_STRING):
    """
    Aggregates queries and distinct user counts by day.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        list: List of aggregation results with _id (date), query_count, and user_count
    """
    collection = get_db_connection(db_string)

    match_condition = {"timestamp": {"$gte": start_date, "$lt": end_date}}

    # Add email/domain exclusion if requested
    if exclude_blacklisted:
        # Create email exclusion condition
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}

        # Create domain exclusion conditions
        domain_exclusions = []
        for domain in BLACKLIST_DOMAINS:
            domain_exclusions.append(
                {"userEmail": {"$not": {"$regex": f"@{domain}$"}}})

        # Combine all conditions
        match_condition = {"$and": [match_condition,
                                    email_exclusion, *domain_exclusions]}

    pipeline = [
        {"$match": match_condition},
        {"$addFields": {"date": {"$dateToString": {
            "format": "%Y-%m-%d", "date": {"$toDate": "$timestamp"}}}}},
        {"$group": {"_id": "$date",
                    "query_count": {"$sum": 1},
                    "users": {"$addToSet": "$userEmail"}}},
        {"$addFields": {"user_count": {"$size": "$users"}}},
        {"$sort": {"_id": 1}}
    ]
    return list(collection.aggregate(pipeline))


@st.cache_data(ttl=10800)
def aggregate_daily_by_url(start_date, end_date, exclude_blacklisted=False, db_string=DB_CONNECTION_STRING):
    """
    Aggregates query counts per requester_url by day.
    Missing or null requester_url values will be replaced with "writer".
    
    Returns:
        list: [
            {
                "_id": "2025-04-20",
                "urls": {
                    "https://app.facticity.ai/": 10,
                    "writer": 3,
                    ...
                }
            },
            ...
        ]
    """
    print("DB STRING:", db_string)
    collection = get_db_connection(db_string)

    # Ensure timestamp exists and convert missing requester_url to "writer"
    match_condition = {
        "timestamp": {"$gte": start_date, "$lt": end_date}
    }

    if exclude_blacklisted:
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}
        domain_exclusions = [
            {"userEmail": {"$not": {"$regex": f"@{domain}$"}}}
            for domain in BLACKLIST_DOMAINS
        ]
        match_condition = {"$and": [match_condition,
                                    email_exclusion, *domain_exclusions]}

    pipeline = [
        {"$match": match_condition},
        {"$addFields": {
            "date": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": {"$toDate": "$timestamp"}
                }
            },
            # Use $ifNull to substitute missing requester_url with "writer"
            "url": {
                "$ifNull": ["$requester_url", "writer"]
            }
        }},
        {"$group": {
            "_id": {"date": "$date", "url": "$url"},
            "query_count": {"$sum": 1}
        }},
        {"$group": {
            "_id": "$_id.date",
            "urls": {
                "$push": {
                    "k": "$_id.url",
                    "v": "$query_count"
                }
            }
        }},
        {"$sort": {"_id": 1}}
    ]

    # Get the raw results and process them in Python to avoid DocumentDB limitations
    raw_results = list(collection.aggregate(pipeline))
    
    # Process results to create safe field names for URLs
    processed_results = []
    for result in raw_results:
        processed_urls = {}
        for url_item in result["urls"]:
            url = url_item["k"]
            count = url_item["v"]
            # Create a safe key by replacing dots with underscores
            safe_key = url.replace(".", "_")
            processed_urls[safe_key] = count
        
        processed_results.append({
            "_id": result["_id"],
            "urls": processed_urls
        })
    
    return processed_results


@st.cache_data(ttl=10800)
def aggregate_daily_users_by_url(start_date, end_date, exclude_blacklisted=False, db_string=DB_CONNECTION_STRING):
    """
    Aggregates distinct userEmails per requester_url by day.
    Empty userEmails (non-logged in users) are counted as 'anonymous_user'.
    Missing or null requester_url values will be replaced with "writer".
    The same user accessing different URLs will be counted once per URL.
    
    Returns:
        list: [
            {
                "_id": "2025-04-20",
                "urls": {
                    "https://app.facticity.ai/": 5,  # 5 distinct users
                    "writer": 2,  # 2 distinct users
                    ...
                }
            },
            ...
        ]
    """
    collection = get_db_connection(db_string)

    # Ensure timestamp exists and include all userEmails (even empty ones for non-logged in users)
    match_condition = {
        "timestamp": {"$gte": start_date, "$lt": end_date},
        # Include empty userEmail (anonymous users)
        "userEmail": {"$exists": True}
    }

    if exclude_blacklisted:
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}
        domain_exclusions = [
            {"userEmail": {"$not": {"$regex": f"@{domain}$"}}}
            for domain in BLACKLIST_DOMAINS
        ]
        match_condition = {"$and": [match_condition,
                                    email_exclusion, *domain_exclusions]}

    pipeline = [
        {"$match": match_condition},
        {"$addFields": {
            "date": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": {"$toDate": "$timestamp"}
                }
            },
            # Use $ifNull to substitute missing requester_url with "writer"
            "url": {
                "$ifNull": ["$requester_url", "writer"]
            },
            # Handle empty userEmail by using a special identifier for anonymous users
            "userIdentifier": {
                "$cond": [
                    {"$eq": ["$userEmail", ""]},
                    # Use this for empty email (not logged in)
                    "anonymous_user",
                    "$userEmail"       # Use actual email for logged in users
                ]
            }
        }},
        # Group by date, url, and userIdentifier to get unique users per url per day
        {"$group": {
            "_id": {"date": "$date", "url": "$url", "user": "$userIdentifier"},
            # Just a placeholder, will count distinct users later
            "count": {"$sum": 1}
        }},
        # Now group by date and url to count unique users
        {"$group": {
            "_id": {"date": "$_id.date", "url": "$_id.url"},
            "user_count": {"$sum": 1}  # Count of unique users per url per day
        }},
        # Group by date to organize urls
        {"$group": {
            "_id": "$_id.date",
            "urls": {
                "$push": {
                    "k": "$_id.url",
                    "v": "$user_count"
                }
            }
        }},
        {"$sort": {"_id": 1}}
    ]

    # Get the raw results and process them in Python to avoid DocumentDB limitations
    raw_results = list(collection.aggregate(pipeline))
    
    # Process results to create safe field names for URLs
    processed_results = []
    for result in raw_results:
        processed_urls = {}
        for url_item in result["urls"]:
            url = url_item["k"]
            count = url_item["v"]
            # Create a safe key by replacing dots with underscores
            safe_key = url.replace(".", "_")
            processed_urls[safe_key] = count
        
        processed_results.append({
            "_id": result["_id"],
            "urls": processed_urls
        })
    
    return processed_results

@st.cache_data(ttl=10800)
def aggregate_weekly_with_users(start_date, end_date, exclude_blacklisted=False):
    """
    Aggregates queries and distinct user counts by week, grouped by the end of the week (Sunday).
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        list: List of aggregation results with _id (week end date), query_count, and user_count
    """
    collection = get_db_connection()

    match_condition = {"timestamp": {"$gte": start_date, "$lt": end_date}}

    # Add email/domain exclusion if requested
    if exclude_blacklisted:
        # Create email exclusion condition
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}

        # Create domain exclusion conditions
        domain_exclusions = []
        for domain in BLACKLIST_DOMAINS:
            domain_exclusions.append(
                {"userEmail": {"$not": {"$regex": f"@{domain}$"}}})

        # Combine all conditions
        match_condition = {"$and": [match_condition,
                                    email_exclusion, *domain_exclusions]}

    pipeline = [
        {"$match": match_condition},
        {"$addFields": {
            "date": {"$toDate": "$timestamp"},
            # 1=Sunday, 7=Saturday
            "dayOfWeek": {"$dayOfWeek": {"$toDate": "$timestamp"}}
        }},
        {"$addFields": {
            "daysToAdd": {
                "$subtract": [
                    7,
                    # daysToAdd = 7 - (dayOfWeek - 1)
                    {"$subtract": ["$dayOfWeek", 1]}
                ]
            }
        }},
        {"$addFields": {
            "endOfWeek": {
                "$dateAdd": {
                    "startDate": "$date",
                    "unit": "day",
                    "amount": "$daysToAdd"
                }
            }
        }},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$endOfWeek"}},
            "query_count": {"$sum": 1},
            "users": {"$addToSet": "$userEmail"}
        }},
        {"$addFields": {"user_count": {"$size": "$users"}}},
        {"$sort": {"_id": 1}}
    ]
    return list(collection.aggregate(pipeline))


@st.cache_data(ttl=10800)
def aggregate_monthly_with_users(start_date, end_date, exclude_blacklisted=False):
    """
    Aggregates queries and distinct user counts by month.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        list: List of aggregation results with _id (year/month), query_count, and user_count
    """
    collection = get_db_connection()

    match_condition = {"timestamp": {"$gte": start_date, "$lt": end_date}}

    # Add email/domain exclusion if requested
    if exclude_blacklisted:
        # Create email exclusion condition
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}

        # Create domain exclusion conditions
        domain_exclusions = []
        for domain in BLACKLIST_DOMAINS:
            domain_exclusions.append(
                {"userEmail": {"$not": {"$regex": f"@{domain}$"}}})

        # Combine all conditions
        match_condition = {"$and": [match_condition,
                                    email_exclusion, *domain_exclusions]}

    pipeline = [
        {"$match": match_condition},
        {"$addFields": {
            "year": {"$year": {"$toDate": "$timestamp"}},
            "month": {"$month": {"$toDate": "$timestamp"}}
        }},
        {"$group": {"_id": {"year": "$year", "month": "$month"},
                    "query_count": {"$sum": 1},
                    "users": {"$addToSet": "$userEmail"}}},
        {"$addFields": {"user_count": {"$size": "$users"}}},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]
    return list(collection.aggregate(pipeline))


@st.cache_data(ttl=10800)
def aggregate_quarterly_with_users(start_date, end_date, exclude_blacklisted=False):
    """
    Aggregates queries and distinct user counts by quarter with custom labels.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        list: List of aggregation results with _id (quarter label), query_count, and user_count
    """
    collection = get_db_connection()

    match_condition = {"timestamp": {"$gte": start_date, "$lt": end_date}}

    # Add email/domain exclusion if requested
    if exclude_blacklisted:
        # Create email exclusion condition
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}

        # Create domain exclusion conditions
        domain_exclusions = []
        for domain in BLACKLIST_DOMAINS:
            domain_exclusions.append(
                {"userEmail": {"$not": {"$regex": f"@{domain}$"}}})

        # Combine all conditions
        match_condition = {"$and": [match_condition,
                                    email_exclusion, *domain_exclusions]}

    pipeline = [
        {"$match": match_condition},
        {"$addFields": {
            "year": {"$year": {"$toDate": "$timestamp"}},
            "month": {"$month": {"$toDate": "$timestamp"}}
        }},
        {"$addFields": {
            "label": {
                "$switch": {
        "branches": [
            {
                "case": {"$and": [{"$gte": ["$month", 3]}, {"$lte": ["$month", 5]}]},
                "then": {"$concat": ["Mar-May ", {"$toString": "$year"}]}
            },
            {
                "case": {"$and": [{"$gte": ["$month", 6]}, {"$lte": ["$month", 8]}]},
                "then": {"$concat": ["June-Aug ", {"$toString": "$year"}]}
            },
            {
                "case": {"$and": [{"$gte": ["$month", 9]}, {"$lte": ["$month", 11]}]},
                "then": {"$concat": ["Sep-Nov ", {"$toString": "$year"}]}
            },
            {
                "case": {"$eq": ["$month", 12]},
                "then": {"$concat": ["Dec ", {"$toString": "$year"}, "-Feb ", {"$toString": {"$add": ["$year", 1]}}]}
            },
            {
                "case": {"$lte": ["$month", 2]},
                "then": {"$concat": ["Dec ", {"$toString": {"$subtract": ["$year", 1]}}, "-Feb ", {"$toString": "$year"}]}
            }
        ],
    "default": "Other"
}

            }
        }},
        {"$match": {"label": {"$nin": ["Other"]}}},
        {"$group": {
            "_id": "$label",
            "query_count": {"$sum": 1},
            "users": {"$addToSet": "$userEmail"}
        }},
        {"$addFields": {"user_count": {"$size": "$users"}}},
        {"$sort": {"_id": 1}}
    ]
    return list(collection.aggregate(pipeline))


def get_mongo_active_users(cutoff, exclude_blacklisted=False):
    """
    Gets active users from MongoDB after a specific cutoff date.
    
    Args:
        cutoff: Datetime cutoff
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        set: Set of active user emails
    """
    collection = get_db_connection()

    query_filter = {"timestamp": {"$gte": cutoff.isoformat()}, "userEmail": {
        "$exists": True}}

    # Add email/domain exclusion if requested
    if exclude_blacklisted:
        # Create email exclusion condition
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}

        # Create domain exclusion conditions
        domain_exclusions = []
        for domain in BLACKLIST_DOMAINS:
            domain_exclusions.append(
                {"userEmail": {"$not": {"$regex": f"@{domain}$"}}})

        # Combine all conditions
        query_filter = {"$and": [query_filter,
                                 email_exclusion, *domain_exclusions]}

    cursor = collection.find(query_filter, {"userEmail": 1})

    # Build a set of emails (normalized to lowercase)
    return set(doc["userEmail"].lower() for doc in cursor if "userEmail" in doc)

def fetch_all_emails_with_timestamps(collection, exclude_blacklisted=False):
    """
    Fetches all emails with their timestamps from MongoDB.
    
    Args:
        collection: MongoDB collection
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        DataFrame: DataFrame with userEmail and timestamp columns
    """
    import pandas as pd
    from config import BLACKLIST_EMAILS, BLACKLIST_DOMAINS
    
    query_filter = {"userEmail": {"$exists": True, "$ne": ""}}

    # Add email/domain exclusion if requested
    if exclude_blacklisted:
        # Create email exclusion condition
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}

        # Create domain exclusion conditions
        domain_exclusions = []
        for domain in BLACKLIST_DOMAINS:
            domain_exclusions.append(
                {"userEmail": {"$not": {"$regex": f"@{domain}$"}}})

        # Combine all conditions
        query_filter = {"$and": [query_filter,
                                email_exclusion, *domain_exclusions]}

    data = collection.find(
        query_filter, {"userEmail": 1, "timestamp": 1, "_id": 0})

    df = pd.DataFrame(list(data))
    if "userEmail" in df.columns:
        df["userEmail"] = df["userEmail"].astype(str).str.lower().str.strip()
    else:
        df["userEmail"] = None
    
    # Convert timestamps with uniform timezone handling
    if "timestamp" in df.columns and not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df["timestamp"] = None
        
    return df

@st.cache_data(ttl=10800)
def get_distribution_data(pipeline, exclude_blacklisted=False):
    """
    Gets user distribution data from MongoDB.
    
    Args:
        pipeline: MongoDB aggregation pipeline
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        list: Aggregation results
    """
    # If excluding blacklisted users, modify the pipeline's match stage if it exists
    if exclude_blacklisted and pipeline and len(pipeline) > 0 and "$match" in pipeline[0]:
        # Get existing match condition
        match_condition = pipeline[0]["$match"]

        # Create email exclusion condition
        email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}

        # Create domain exclusion conditions
        domain_exclusions = []
        for domain in BLACKLIST_DOMAINS:
            domain_exclusions.append(
                {"userEmail": {"$not": {"$regex": f"@{domain}$"}}})

        # Combine all conditions
        new_match = {"$and": [match_condition,
                              email_exclusion, *domain_exclusions]}

        # Replace the match stage
        pipeline[0]["$match"] = new_match

    return list(get_db_connection().aggregate(pipeline))


@st.cache_data(ttl=10800)
def fetch_recent_queries(days=7, exclude_blacklisted=True):
    """
    Fetch query data from MongoDB with caching
    Returns:
    DataFrame: DataFrame with query data from the specified time period
    """
    try:
        collection = get_db_connection()

        # Calculate the date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Format dates for MongoDB query
        start_date_iso = start_date.isoformat()
        end_date_iso = end_date.isoformat()

        # Create query filter to get queries from the specified time period with query text
        query_filter = {
            "timestamp": {"$gte": start_date_iso, "$lt": end_date_iso},
            "query": {"$exists": True},
            "userEmail": {"$exists": True}
        }

        # Add blacklist exclusion if requested
        if exclude_blacklisted:
            from config import BLACKLIST_EMAILS, BLACKLIST_DOMAINS

            # Exclude blacklisted emails
            email_exclusion = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}

            # Exclude blacklisted domains
            domain_exclusions = []
            for domain in BLACKLIST_DOMAINS:
                domain_exclusions.append(
                    {"userEmail": {"$not": {"$regex": f"@{domain}$"}}}
                )

            # Combine all conditions
            query_filter = {"$and": [query_filter,
                                     email_exclusion, *domain_exclusions]}

        # Fetch data from MongoDB
        data = collection.find(
            query_filter,
            {"query": 1, "userEmail": 1, "timestamp": 1, "_id": 0}
        )

        # Convert to DataFrame
        df = pd.DataFrame(list(data))

        if df.empty:
            st.warning(f"No queries found in the past {days} days.")
            return None

        return df

    except Exception as e:
        st.error(f"Error fetching recent queries: {e}")
        return None
