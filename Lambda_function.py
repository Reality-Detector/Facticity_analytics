import json
from datetime import datetime, timedelta, timezone
import re
import requests
import os
import pandas as pd
from database.mongo_client import get_db_connection, is_valid_user
from utils.user_profile_utils import load_iab_categories
from pymongo import MongoClient

DATA_FOLDER = "data/email_segments"
USER_PROFILES_FILE = os.path.join(DATA_FOLDER, "user_profiles.json")


def is_url(string):
    return bool(re.match(r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$', string))


def is_video_transcript(text: str) -> bool:
    return bool(re.search(r"\d{1,2}:\d{2}", text)) or bool(re.search(r"(Speaker \d+|Narrator|Host):", text, re.IGNORECASE))


def load_user_profiles():
    try:
        with open(USER_PROFILES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}



profile_data = load_user_profiles()

def categorize_user_activity(df):
    now = datetime.now(timezone.utc)
    user_categories = []

    for email, group in df.groupby("userEmail"):
        timestamps = sorted(group["timestamp"].tolist())
        days_ago = [(now - ts).days for ts in timestamps]
        total = len(timestamps)
        latest = min(days_ago)

        query = ""
        if email in profile_data:
            query = profile_data[email].get("recent_queries", [""])[0]

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

    return user_categories


def lambda_handler(event, context):
    """
    AWS Lambda handler to return user engagement categories.
    Accepts optional parameters:
    - `days`: how many days to look back (default 7)
    - `exclude_blacklisted`: True/False (default True)
    """
    

    # Parse input
    days = int(event.get("days", 7))
    exclude_blacklisted = event.get("exclude_blacklisted", True)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    # Get DB data
    collection = get_db_connection()
    pipeline = [
        {"$match": {"timestamp": {"$gte": start_date.isoformat()}}},
        {"$project": {"userEmail": 1, "timestamp": 1}},
    ]

    data = list(collection.aggregate(pipeline))
    if not data:
        return {"statusCode": 404, "body": json.dumps({"message": "No data found"})}

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=["timestamp"])

    if exclude_blacklisted:
        df = df[df["userEmail"].apply(is_valid_user)]

    categorized_users = categorize_user_activity(df)

    # Insert each categorized user into MongoDB
    mongo_uri = os.environ["MONGODB_URI"]
    client = MongoClient(mongo_uri)
    db = client["facticity"]
    collection = db["weekly_emails"]
    collection.delete_many({})
    for user in categorized_users:
        collection.insert_one({
            "userEmail": user["userEmail"],
            "categories": user["Categories"]
        })
    zapier_url = "https://hooks.zapier.com/hooks/catch/23606341/ubeyngh/"  # replace with your URL

    for user in categorized_users:
        payload = {
            "userEmail": user["userEmail"],
            "categories": user["Categories"]
        }
        requests.post(zapier_url, json=payload)
    return {
        "statusCode": 200,
        "body": json.dumps({"message": f"{len(categorized_users)} records inserted"})
    }


