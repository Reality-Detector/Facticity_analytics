"""
Date handling utilities for the Facticity dashboard.
"""
import re
from datetime import datetime, timedelta


def quarter_sort_key(label):
    """
    Sort key for quarterly labels like:
    - "Mar-May 2025"
    - "June-Aug 2025"
    - "Sep-Nov 2025"
    - "Dec 2024-Feb 2025"
    """
    if label.startswith("Mar-May"):
        quarter_index = 1
        year_str = label.split()[1]
    elif label.startswith("June-Aug"):
        quarter_index = 2
        year_str = label.split()[1]
    elif label.startswith("Sep-Nov"):
        quarter_index = 3
        year_str = label.split()[1]
    elif label.startswith("Dec") and "-Feb" in label:
        quarter_index = 4
        # Extract the first year from "Dec 2024-Feb 2025"
        year_str = label.split()[1].split("-")[0]
    else:
        # Fallback for unrecognized formats
        quarter_index = 999
        year_str = "9999"

    try:
        year = int(year_str)
    except ValueError:
        year = 9999

    return (year, quarter_index)



def get_date_ranges():
    """
    Returns standard date ranges for analytics.
    
    Returns:
        dict: Dictionary of date ranges
    """
    now = datetime.now()
    
    # Daily range (last 14 days)
    daily_start = (now - timedelta(days=14)).isoformat()
    daily_end = now.isoformat()
    
    # Weekly/Monthly/Quarterly ranges
    standard_start = "2024-06-01T00:00:00Z"
    standard_end = now.isoformat()
    
    return {
        "daily": (daily_start, daily_end),
        "weekly": (standard_start, standard_end),
        "monthly": (standard_start, standard_end),
        "quarterly": (standard_start, standard_end)
    }
