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
    Includes data from both web2 and web3 databases.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        list: List of aggregation results with _id (week end date), query_count, and user_count
    """
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
    
    # Query both web2 and web3 databases
    results_web2 = []
    results_web3 = []
    
    try:
        client_web2 = document_db_web2.get_client()
        collection_web2 = client_web2["facticity"]["query_new"]
        results_web2 = list(collection_web2.aggregate(pipeline))
    except Exception as e:
        st.warning(f"Error querying web2 database: {e}")
    
    try:
        client_web3 = document_db_web3.get_client()
        collection_web3 = client_web3["facticity"]["query_new"]
        results_web3 = list(collection_web3.aggregate(pipeline))
    except Exception as e:
        st.warning(f"Error querying web3 database: {e}")
    
    # Combine results by merging data for the same week end dates
    combined_results = {}
    
    # Process web2 results
    for result in results_web2:
        week_end = result["_id"]
        if week_end not in combined_results:
            combined_results[week_end] = {
                "_id": week_end,
                "query_count": 0,
                "users": set()
            }
        combined_results[week_end]["query_count"] += result["query_count"]
        # MongoDB $addToSet returns a list, convert to set and update
        users_list = result.get("users", [])
        combined_results[week_end]["users"].update(users_list)
    
    # Process web3 results
    for result in results_web3:
        week_end = result["_id"]
        if week_end not in combined_results:
            combined_results[week_end] = {
                "_id": week_end,
                "query_count": 0,
                "users": set()
            }
        combined_results[week_end]["query_count"] += result["query_count"]
        # MongoDB $addToSet returns a list, convert to set and update
        users_list = result.get("users", [])
        combined_results[week_end]["users"].update(users_list)
    
    # Convert sets to lists and calculate user_count
    final_results = []
    for week_end in sorted(combined_results.keys()):
        result = combined_results[week_end]
        final_results.append({
            "_id": result["_id"],
            "query_count": result["query_count"],
            "user_count": len(result["users"])
        })
    
    return final_results


@st.cache_data(ttl=10800)
def aggregate_monthly_with_users(start_date, end_date, exclude_blacklisted=False):
    """
    Aggregates queries and distinct user counts by month.
    Includes data from both web2 and web3 databases.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        list: List of aggregation results with _id (year/month), query_count, and user_count
    """
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
    
    # Query both web2 and web3 databases
    results_web2 = []
    results_web3 = []
    
    try:
        client_web2 = document_db_web2.get_client()
        collection_web2 = client_web2["facticity"]["query_new"]
        results_web2 = list(collection_web2.aggregate(pipeline))
    except Exception as e:
        st.warning(f"Error querying web2 database: {e}")
    
    try:
        client_web3 = document_db_web3.get_client()
        collection_web3 = client_web3["facticity"]["query_new"]
        results_web3 = list(collection_web3.aggregate(pipeline))
    except Exception as e:
        st.warning(f"Error querying web3 database: {e}")
    
    # Combine results by merging data for the same year/month
    combined_results = {}
    
    # Process web2 results
    for result in results_web2:
        year_month_key = (result["_id"]["year"], result["_id"]["month"])
        if year_month_key not in combined_results:
            combined_results[year_month_key] = {
                "_id": result["_id"],
                "query_count": 0,
                "users": set()
            }
        combined_results[year_month_key]["query_count"] += result["query_count"]
        # MongoDB $addToSet returns a list, convert to set and update
        users_list = result.get("users", [])
        combined_results[year_month_key]["users"].update(users_list)
    
    # Process web3 results
    for result in results_web3:
        year_month_key = (result["_id"]["year"], result["_id"]["month"])
        if year_month_key not in combined_results:
            combined_results[year_month_key] = {
                "_id": result["_id"],
                "query_count": 0,
                "users": set()
            }
        combined_results[year_month_key]["query_count"] += result["query_count"]
        # MongoDB $addToSet returns a list, convert to set and update
        users_list = result.get("users", [])
        combined_results[year_month_key]["users"].update(users_list)
    
    # Convert sets to lists and calculate user_count
    final_results = []
    for year_month_key in sorted(combined_results.keys()):
        result = combined_results[year_month_key]
        final_results.append({
            "_id": result["_id"],
            "query_count": result["query_count"],
            "user_count": len(result["users"])
        })
    
    return final_results


@st.cache_data(ttl=10800)
def aggregate_quarterly_with_users(start_date, end_date, exclude_blacklisted=False):
    """
    Aggregates queries and distinct user counts by quarter with custom labels.
    Includes data from both web2 and web3 databases.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        list: List of aggregation results with _id (quarter label), query_count, and user_count
    """
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
    
    # Query both web2 and web3 databases
    results_web2 = []
    results_web3 = []
    
    try:
        client_web2 = document_db_web2.get_client()
        collection_web2 = client_web2["facticity"]["query_new"]
        results_web2 = list(collection_web2.aggregate(pipeline))
    except Exception as e:
        st.warning(f"Error querying web2 database: {e}")
    
    try:
        client_web3 = document_db_web3.get_client()
        collection_web3 = client_web3["facticity"]["query_new"]
        results_web3 = list(collection_web3.aggregate(pipeline))
    except Exception as e:
        st.warning(f"Error querying web3 database: {e}")
    
    # Combine results by merging data for the same quarter label
    combined_results = {}
    
    # Process web2 results
    for result in results_web2:
        label = result["_id"]
        if label not in combined_results:
            combined_results[label] = {
                "_id": label,
                "query_count": 0,
                "users": set()
            }
        combined_results[label]["query_count"] += result["query_count"]
        # MongoDB $addToSet returns a list, convert to set and update
        users_list = result.get("users", [])
        combined_results[label]["users"].update(users_list)
    
    # Process web3 results
    for result in results_web3:
        label = result["_id"]
        if label not in combined_results:
            combined_results[label] = {
                "_id": label,
                "query_count": 0,
                "users": set()
            }
        combined_results[label]["query_count"] += result["query_count"]
        # MongoDB $addToSet returns a list, convert to set and update
        users_list = result.get("users", [])
        combined_results[label]["users"].update(users_list)
    
    # Convert sets to lists and calculate user_count
    final_results = []
    for label in sorted(combined_results.keys()):
        result = combined_results[label]
        final_results.append({
            "_id": result["_id"],
            "query_count": result["query_count"],
            "user_count": len(result["users"])
        })
    
    return final_results


def get_mongo_active_users(cutoff, exclude_blacklisted=False):
    """
    Gets active users from MongoDB after a specific cutoff date.
    Includes data from both web2 and web3 databases.
    
    Args:
        cutoff: Datetime cutoff
        exclude_blacklisted: Whether to exclude blacklisted emails/domains
        
    Returns:
        set: Set of active user emails
    """
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

    # Query both web2 and web3 databases
    active_users = set()
    
    try:
        client_web2 = document_db_web2.get_client()
        collection_web2 = client_web2["facticity"]["query_new"]
        cursor_web2 = collection_web2.find(query_filter, {"userEmail": 1})
        # Build a set of emails (normalized to lowercase) from web2
        active_users.update(doc["userEmail"].lower() for doc in cursor_web2 if "userEmail" in doc)
    except Exception as e:
        st.warning(f"Error querying web2 database: {e}")
    
    try:
        client_web3 = document_db_web3.get_client()
        collection_web3 = client_web3["facticity"]["query_new"]
        cursor_web3 = collection_web3.find(query_filter, {"userEmail": 1})
        # Build a set of emails (normalized to lowercase) from web3
        active_users.update(doc["userEmail"].lower() for doc in cursor_web3 if "userEmail" in doc)
    except Exception as e:
        st.warning(f"Error querying web3 database: {e}")

    return active_users

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


@st.cache_data(ttl=10800)
def aggregate_daily_tweets(start_date, end_date):
    """
    Aggregates tweet counts by day from the tweets collection.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        
    Returns:
        list: List of aggregation results with _id (date) and tweet_count
    """
    from dbutils.DocumentDB import document_db_web2
    
    try:
        client = document_db_web2.get_client()
        tweets_collection = client["facticity"]["tweets"]
        
        # Convert ISO strings to datetime objects
        start_dt = pd.to_datetime(start_date).to_pydatetime()
        end_dt = pd.to_datetime(end_date).to_pydatetime()
        
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_dt, "$lt": end_dt}
                }
            },
            {
                "$project": {
                    "date": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$created_at"
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "tweet_count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        results = list(tweets_collection.aggregate(pipeline))
        return results
        
    except Exception as e:
        st.warning(f"Error fetching Twitter data: {e}")
        return []


@st.cache_data(ttl=10800)
def aggregate_daily_twitter_users(start_date, end_date):
    """
    Aggregates unique Twitter users (taggers) by day from the tweets collection.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        
    Returns:
        list: List of aggregation results with _id (date) and user_count
    """
    from dbutils.DocumentDB import document_db_web2
    
    try:
        client = document_db_web2.get_client()
        tweets_collection = client["facticity"]["tweets"]
        
        # Convert ISO strings to datetime objects
        start_dt = pd.to_datetime(start_date).to_pydatetime()
        end_dt = pd.to_datetime(end_date).to_pydatetime()
        
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_dt, "$lt": end_dt},
                    "$or": [
                        {"tagger.username": {"$exists": True, "$ne": ""}},
                        {"tagger.id": {"$exists": True, "$ne": ""}}
                    ]
                }
            },
            {
                "$project": {
                    "date": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$created_at"
                        }
                    },
                    "user_identifier": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$tagger.username", None]},
                                    {"$ne": ["$tagger.username", ""]}
                                ]
                            },
                            "$tagger.username",
                            {"$toString": "$tagger.id"}
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": {"date": "$date", "user": "$user_identifier"}
                }
            },
            {
                "$group": {
                    "_id": "$_id.date",
                    "user_count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        results = list(tweets_collection.aggregate(pipeline))
        return results
        
    except Exception as e:
        st.warning(f"Error fetching Twitter user data: {e}")
        return []
