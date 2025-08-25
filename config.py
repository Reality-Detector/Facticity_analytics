"""
Configuration module for the Facticity dashboard.
Centralizes all configuration settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Auth0 Configuration
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
API_AUDIENCE = f"https://{AUTH0_DOMAIN}/api/v2/" if AUTH0_DOMAIN else ""
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")

# MongoDB Configuration
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
# API_DB_CONNECTION_STRING = os.getenv("API_CUSTOMER_DB_CONNECTION_STRING")

# PostHog Configuration
POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")
POSTHOG_PROJECT_ID = os.getenv("POSTHOG_PROJECT_ID", "127673")

# User filtering settings - load blacklisted emails from env
blacklist_emails_str = os.getenv("BLACKLIST_EMAILS", "")
BLACKLIST_EMAILS = set(email.strip()
                       for email in blacklist_emails_str.split(",") if email.strip())

# Table with user info (email, location)
AUTH0_IP_LOOKUP_FILEPATH = "data/auth0/auth0_iplookup.csv"

# Add default blacklisted emails if not already in environment
default_emails = [
    "dennis.ye@gmail.com",
    "sriramrsk111@gmail.com",
    "jacobkozhipatt10@gmail.com",
    "n.jeunw@gmail.com",
    "amshahrujrashid@gmail.com"
]
for email in default_emails:
    BLACKLIST_EMAILS.add(email)

# Domain blacklist remains static as requested
BLACKLIST_DOMAINS = set(["aiseer.co"])

# # Chart colors
PRIMARY_BLUE = " #1a73e8"
LIGHT_BLUE = "#8ab4f8"
MODERN_ORANGE = "rgba(255,109,0,0.95)"

# Blues (dark to light)
blue_1 = "#174ea6"
blue_2 = "#1a73e8"
blue_3 = "#4285f4"
blue_4 = "#8ab4f8"
blue_5 = "#d2e3fc"

# Oranges (dark to light)
orange_1 = "#cc4400"
orange_2 = "#ff6d00"
orange_3 = "#ff7043"
orange_4 = "#ffc7a6"

# Neutrals (dark to light)
gray_1 = "#212121"
gray_2 = "#424242"
gray_3 = "#9e9e9e"
gray_4 = "#e0e0e0"
gray_5 = "#f5f5f5"

# Accents
green_1 = "#00c853"
purple_1 = "#9b59b6"
red_1 = "#e53935"
yellow_1 = "#f9a825"

# Combined palette for charts (ordered)
CUSTOM_PALETTE = [
    blue_1, blue_2, blue_3, orange_2, orange_3,
    green_1, purple_1, red_1, yellow_1, orange_4, blue_4
]
