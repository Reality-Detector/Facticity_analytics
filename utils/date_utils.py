"""
Date handling utilities for the Facticity dashboard.
"""
import re
from datetime import datetime, timedelta


def quarter_sort_key(label):
    """
    Assign a numeric index based on the quarter portion
    and parse out the 'starting year' for that quarter.
    
    Args:
        label: Quarter label string (e.g., 'June-Aug 2024')
        
    Returns:
        tuple: Sortable tuple of (year, quarter_number)
    """
    # Quarter order:
    #   "June-Aug" -> 1
    #   "Sep-Nov"  -> 2
    #   "Dec"      -> 3

    parts = label.split()
    # Examples:
    #   "June-Aug 2024"        -> ["June-Aug", "2024"]
    #   "Sep-Nov 2024"         -> ["Sep-Nov", "2024"]
    #   "Dec 2024-Feb 2025"    -> ["Dec", "2024-Feb", "2025"]

    # Identify which quarter it is:
    if label.startswith("June-Aug"):
        quarter_index = 1
        # The year is simply parts[1], e.g. "2024"
        year_str = parts[1]
    elif label.startswith("Sep-Nov"):
        quarter_index = 2
        year_str = parts[1]
    elif label.startswith("Dec"):
        quarter_index = 3
        # "Dec 2024-Feb 2025" -> parts[1] is "2024-Feb"
        # We only need the first year, e.g. "2024"
        # so we can split by "-" or by " " or just take the numeric portion:
        year_str = parts[1].split("-")[0]  # "2024"
    else:
        # If for some reason it's not recognized, throw it at the end
        quarter_index = 999
        year_str = "9999"

    # Convert year_str to int safely
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
