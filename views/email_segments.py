"""
Enhanced Email segments view for the Facticity dashboard with visualizations, IP geolocation analysis,
and field-specific analysis options.
"""
import streamlit as st
import pandas as pd
import gzip
import io
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import ipaddress
import time
import re
import numpy as np
import pycountry
import os
from pathlib import Path
from streamlit_plotly_events import plotly_events
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from database.mongo_client import get_db_connection, fetch_all_emails_with_timestamps
from database.auth0_client import export_all_users
from services.analytics import filter_emails_exclusively, classify_engagement_with_pandas
from utils.chart_utils import get_table_download_link
from utils.user_profile_utils import (
    create_bedrock_client, load_iab_categories, generate_category_embeddings,
    load_query_embeddings, save_query_embeddings, fetch_user_queries_with_date_range,
    categorize_queries, generate_user_profiles, generate_query_embeddings_batch
)
from config import AUTH0_IP_LOOKUP_FILEPATH

# Constants
DATA_FOLDER = "data/email_segments"
AUTH0_DATA_FILE = os.path.join(DATA_FOLDER, "auth0_export.csv")
TIME_BASED_DATA_FILE = os.path.join(DATA_FOLDER, "time_based_segments.csv")
ENGAGEMENT_DATA_FILE = os.path.join(DATA_FOLDER, "engagement_segments.csv")
USER_PROFILES_FILE =  os.path.join(DATA_FOLDER, "user_profiles.json") 

@st.cache_data
def load_user_profiles():
    """Load user profiles data from JSON file."""
    try:
        # Try current directory first, then parent directory
        with open(USER_PROFILES_FILE, 'r') as f:
            data = json.load(f)
        st.success(f"Loaded {len(data)} user profiles from {USER_PROFILES_FILE}")
        return data
    except Exception as e:
        st.error(f"Error loading user profiles: {e}")
        return {}
    
def ensure_data_folder():
    """Ensure the data folder exists."""
    Path(DATA_FOLDER).mkdir(exist_ok=True)


def save_data_to_csv(df, filename):
    """Save DataFrame to CSV in the data folder."""
    ensure_data_folder()
    df.to_csv(filename, index=False)
    st.success(f"Data saved to {filename}")


def load_data_from_csv(filename):
    """Load DataFrame from CSV if file exists."""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None


def get_last_fetched_time(filename):
    """Get the last modified time of a file."""
    if os.path.exists(filename):
        timestamp = os.path.getmtime(filename)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return None

def standardize_country(country_str):
    """
    Convert country codes to full country names using pycountry library.
    Simplified to avoid data contamination.
    
    Args:
        country_str: String containing country code or name
        
    Returns:
        Standardized country name
    """
    if not country_str or pd.isna(country_str):
        return None
        
    # Clean the input string
    if isinstance(country_str, str):
        cleaned = country_str.strip().strip('"\'').strip()
    else:
        return country_str
    
    # Handle empty strings
    if not cleaned:
        return None
    
    # Direct mapping for common codes and edge cases
    country_mapping = {
        'US': 'United States',
        'USA': 'United States',
        'United States': 'United States',
        'GB': 'United Kingdom',
        'UK': 'United Kingdom',
        'DE': 'Germany',
        'CA': 'Canada',
        'AU': 'Australia',
        'SG': 'Singapore',
        'SE': 'Sweden',
        'CH': 'Switzerland',
        'FR': 'France',
        'JP': 'Japan',
        'IN': 'India',
        'BR': 'Brazil',
        'IT': 'Italy',
        'NL': 'Netherlands',
        'ES': 'Spain',
        'KR': 'South Korea',
        'MX': 'Mexico',
        'RU': 'Russia',
        'CN': 'China',
        'DK': 'Denmark',
        'NO': 'Norway',
        'NZ': 'New Zealand',
        'BE': 'Belgium',
        'ID': 'Indonesia',
        'CO': 'Colombia',
        'KH': 'Cambodia',
        'CR': 'Costa Rica',
        'IE': 'Ireland',
    }
    
    # Check if it's in our direct mapping
    if cleaned.upper() in country_mapping:
        return country_mapping[cleaned.upper()]
    
    # Try as 2-letter country code (ISO 3166-1 alpha-2)
    if len(cleaned) == 2:
        try:
            country = pycountry.countries.get(alpha_2=cleaned.upper())
            if country:
                return country.name
        except (AttributeError, KeyError):
            pass
    
    return cleaned


# IP Geolocation functions
def is_valid_ip(ip_str):
    """Check if the string is a valid IP address."""
    try:
        # Handle IPv6 addresses that start with numbers like 2406:3003:...
        if ':' in ip_str:
            # IPv6 address
            ipaddress.IPv6Address(ip_str)
            return True
        else:
            # IPv4 address
            ipaddress.IPv4Address(ip_str)
            return True
    except (ValueError, ipaddress.AddressValueError):
        return False

def clean_ip(ip_str):
    """Clean IP string by removing quotes and whitespace."""
    if pd.isna(ip_str) or not isinstance(ip_str, str):
        return None
    # Remove quotes and whitespace
    clean_ip = ip_str.strip("'\" \t")
    # Check if it's a valid IP after cleaning
    if is_valid_ip(clean_ip):
        return clean_ip
    return None

def geolocate_ip_batch(ips):
    """
    Batch geolocate IP addresses using ip-api.com (free tier, 45 requests per minute)
    
    Args:
        ips: List of IP addresses to geolocate
        
    Returns:
        Dictionary mapping IPs to country codes
    """
    # Remove duplicates and None values
    unique_ips = list(set([ip for ip in ips if ip]))

    # Split into chunks of 100 (API limit)
    chunk_size = 100
    results = {}

    for i in range(0, len(unique_ips), chunk_size):
        chunk = unique_ips[i:i+chunk_size]

        try:
            response = requests.post(
                'http://ip-api.com/batch',
                json=[{"query": ip} for ip in chunk],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                for item in data:
                    if item.get('status') == 'success':
                        results[item['query']] = {
                            'country': item.get('country'),
                            'country_code': item.get('countryCode'),
                            'region': item.get('regionName'),
                            'city': item.get('city')
                        }

            # Respect rate limits (45 requests per minute for free tier)
            if i + chunk_size < len(unique_ips):
                time.sleep(1.5)

        except Exception as e:
            st.warning(f"Error geolocating IP batch: {e}")

    return results


def load_existing_ip_data():
    """
    Load existing IP lookup data from CSV file.
    
    Returns:
        Dictionary mapping emails to IP data
    """
    try:
        ip_df = pd.read_csv(AUTH0_IP_LOOKUP_FILEPATH)

        # Clean column names
        ip_df.columns = [col.strip() for col in ip_df.columns]

        # Clean email and ip values by removing quotes
        for col in ['email', 'last_ip']:
            if col in ip_df.columns:
                ip_df[col] = ip_df[col].astype(
                    str).str.replace("'", "").str.strip()

        # Create mapping dictionary for easier lookup
        ip_lookup = {}
        for _, row in ip_df.iterrows():
            if 'email' in row and 'asn_country_code' in row and pd.notna(row['asn_country_code']):
                ip_lookup[row['email']] = {
                    'country_code': row['asn_country_code'],
                    'country': row['asn_description'].split(',')[-1].strip() if pd.notna(row['asn_description']) else None,
                    'provider': row['asn_description'].split(',')[0].strip().replace('"', '') if pd.notna(row['asn_description']) else None,
                    'ip': row['last_ip'],
                    'ip_version': row['ip_version']
                }

        st.success(f"Loaded {len(ip_lookup)} existing IP lookup records")
        return ip_lookup
    except Exception as e:
        st.warning(f"Could not load existing IP data: {e}")
        return {}


def save_new_ip_data(new_ip_data):
    """
    Save newly found IP geolocation data to the auth0_iplookup.csv file.
    
    Args:
        new_ip_data: Dictionary of new IP data keyed by email
    """
    try:
        # Try to load existing file
        try:
            existing_df = pd.read_csv(AUTH0_IP_LOOKUP_FILEPATH)
        except:
            # Create a new DataFrame if file doesn't exist
            existing_df = pd.DataFrame(columns=[
                'name', 'email', 'last_ip', 'ip_version', 'asn',
                'asn_country_code', 'asn_description'
            ])

        # Create DataFrame from new data
        new_rows = []
        for email, data in new_ip_data.items():
            # Find name from email if possible
            name = email.split('@')[0].replace('.', ' ').title()

            new_rows.append({
                'name': f"'{name}",  # Format like existing data
                'email': f"'{email}",
                'last_ip': data['last_ip'],
                'ip_version': data['ip_version'],
                'asn': 'N/A',  # We don't have ASN from geolocation API
                'asn_country_code': data['country_code'],
                'asn_description': f'"{data.get("region", "")}, {data["country"]}"'
            })

        if not new_rows:
            st.info("No new IP data to save to file")
            return

        # Convert to DataFrame and concatenate with existing
        new_df = pd.DataFrame(new_rows)

        # Remove duplicates based on email before appending
        existing_emails = existing_df['email'].apply(
            lambda x: x.replace("'", "").strip() if isinstance(x, str) else x)
        new_emails = new_df['email'].apply(lambda x: x.replace(
            "'", "").strip() if isinstance(x, str) else x)

        # Only keep new emails that don't exist in the existing data
        new_df = new_df[~new_emails.isin(existing_emails)]

        if not new_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(AUTH0_IP_LOOKUP_FILEPATH, index=False)
            st.success(
                f"Saved {len(new_df)} new IP lookup records to auth0_iplookup.csv")
        else:
            st.info("No new IP data to save to file")

    except Exception as e:
        st.error(f"Error saving IP data to CSV: {e}")
        st.exception(e)


def analyze_auth0_export(df):
    """
    Analyze Auth0 export data and add IP geolocation data,
    merging with existing IP lookup data where available.
    Simplified to avoid country standardization issues.
    
    Args:
        df: DataFrame with Auth0 export data
        
    Returns:
        Enhanced DataFrame with analysis
    """
    # Make a copy to avoid modifying the original
    analysis_df = df.copy()

    # Clean column names for consistency
    analysis_df.columns = [col.strip() for col in analysis_df.columns]

    # 1. Extract and clean IP addresses
    if 'last_ip' in analysis_df.columns:
        analysis_df['clean_ip'] = analysis_df['last_ip'].apply(clean_ip)

    # 2. Load existing IP data
    existing_ip_data = load_existing_ip_data()

    # 3. Prepare empty columns for geolocation data
    for col in ['ip_country', 'ip_country_code', 'ip_provider', 'ip_region', 'ip_city', 'ip_version']:
        analysis_df[col] = None

    # 4. Apply existing IP data where available
    if 'email' in analysis_df.columns and existing_ip_data:
        for i, row in analysis_df.iterrows():
            email = row['email']
            if isinstance(email, str):
                email = email.replace("'", "").strip()
                if email in existing_ip_data:
                    ip_data = existing_ip_data[email]
                    analysis_df.at[i, 'ip_country'] = ip_data['country']
                    analysis_df.at[i,
                                   'ip_country_code'] = ip_data['country_code']
                    analysis_df.at[i, 'ip_provider'] = ip_data['provider']
                    analysis_df.at[i, 'ip_version'] = ip_data['ip_version']

    # 5. Identify emails without country data that need geolocation
    emails_without_country = set()
    for i, row in analysis_df.iterrows():
        if pd.isna(row.get('ip_country')) and not pd.isna(row.get('clean_ip')):
            emails_without_country.add(i)

    # 6. Geolocate only IPs without existing data
    new_ip_data = {}
    if emails_without_country:
        st.write(
            f"Looking up {len(emails_without_country)} IP addresses without existing data...")

        # Filter for IPs that need geolocation
        ips_to_lookup = [
            analysis_df.at[i, 'clean_ip']
            for i in emails_without_country
            if not pd.isna(analysis_df.at[i, 'clean_ip'])
        ]

        if ips_to_lookup:
            with st.spinner(f'Geolocating {len(ips_to_lookup)} IP addresses...'):
                geo_results = geolocate_ip_batch(ips_to_lookup)

                # Add geolocation data only for rows without existing data
                for i in emails_without_country:
                    ip = analysis_df.at[i, 'clean_ip']
                    if ip in geo_results:
                        analysis_df.at[i, 'ip_country'] = geo_results[ip].get(
                            'country')
                        analysis_df.at[i, 'ip_country_code'] = geo_results[ip].get(
                            'country_code')
                        analysis_df.at[i, 'ip_region'] = geo_results[ip].get(
                            'region')
                        analysis_df.at[i, 'ip_city'] = geo_results[ip].get(
                            'city')
                        analysis_df.at[i,
                                       'ip_version'] = 'IPv6' if ':' in ip else 'IPv4'

                        # Store new IP data for saving to CSV
                        if 'email' in analysis_df.columns:
                            email = analysis_df.at[i, 'email']
                            if isinstance(email, str) and email.strip():
                                new_ip_data[email.replace("'", "").strip()] = {
                                    'last_ip': ip,
                                    'ip_version': 'IPv6' if ':' in ip else 'IPv4',
                                    'country': geo_results[ip].get('country'),
                                    'country_code': geo_results[ip].get('country_code'),
                                    'region': geo_results[ip].get('region'),
                                    'city': geo_results[ip].get('city')
                                }
    else:
        st.success(
            "All users already have IP country data from existing lookup file")

    # 7. Save new IP data to CSV if any was found
    if new_ip_data:
        save_new_ip_data(new_ip_data)

    # 8. Extract declared country from user_metadata.country if it exists
    country_col = [col for col in analysis_df.columns if 'country' in col.lower(
    ) and 'user_metadata' in col.lower()]
    if country_col:
        metadata_country_col = country_col[0]
        # Clean metadata country column (remove quotes, etc.)
        analysis_df['declared_country'] = analysis_df[metadata_country_col].apply(
            lambda x: x.strip("'\" \t") if isinstance(x, str) else x
        )

    # 9. Standardize country names using pycountry
    # Apply standardization to both declared and inferred countries
    if 'declared_country' in analysis_df.columns:
        analysis_df['standardized_declared_country'] = analysis_df['declared_country'].apply(
            standardize_country)

    if 'ip_country' in analysis_df.columns:
        analysis_df['standardized_inferred_country'] = analysis_df['ip_country'].apply(
            standardize_country)

    # 10. Create simple match flag based on standardized values
    if 'standardized_declared_country' in analysis_df.columns and 'standardized_inferred_country' in analysis_df.columns:
        # Simple exact match for reliability
        analysis_df['country_match'] = analysis_df.apply(
            lambda row: (pd.notna(row['standardized_declared_country']) and
                         pd.notna(row['standardized_inferred_country']) and
                         row['standardized_declared_country'] == row['standardized_inferred_country']),
            axis=1
        )

        # Create standardized_country column with the best available value
        analysis_df['standardized_country'] = analysis_df.apply(
            lambda row: row['standardized_declared_country'] if pd.notna(row['standardized_declared_country'])
            else row['standardized_inferred_country'],
            axis=1
        )

    # 11. Set country status
    analysis_df['country_status'] = analysis_df.apply(
        lambda row:
            'Match' if row.get('country_match', False)
            else 'Mismatch' if (pd.notna(row.get('standardized_declared_country')) and
                                pd.notna(row.get('standardized_inferred_country')))
            else 'Only Declared' if pd.notna(row.get('standardized_declared_country'))
            else 'Only Inferred' if pd.notna(row.get('standardized_inferred_country'))
            else 'Both Missing',
        axis=1
    )

    # 12. Convert timestamps to datetime
    date_columns = ['created_at', 'updated_at', 'last_login']
    for col in date_columns:
        if col in analysis_df.columns:
            analysis_df[col] = pd.to_datetime(
                analysis_df[col], errors='coerce')

    # 13. Calculate account age
    if 'created_at' in analysis_df.columns:
        # Convert created_at to datetime with UTC timezone
        created_at_dt = pd.to_datetime(
            analysis_df['created_at'], errors='coerce')

        # Ensure 'now' has the same timezone
        now = datetime.now(created_at_dt.dt.tz)

        # If created_at doesn't have timezone info, make now timezone-naive
        if created_at_dt.dt.tz is None:
            now = datetime.now()

        # Calculate the difference in days
        analysis_df['account_age_days'] = (
            now - created_at_dt).dt.total_seconds() / (24 * 60 * 60)

    # 14. Calculate engagement metrics
    if 'logins_count' in analysis_df.columns:
        # Convert to numeric, handling errors
        analysis_df['logins_count'] = pd.to_numeric(
            analysis_df['logins_count'], errors='coerce').fillna(0)

        # Calculate logins per day
        if 'account_age_days' in analysis_df.columns:
            analysis_df['logins_per_day'] = analysis_df['logins_count'] / \
                analysis_df['account_age_days'].clip(lower=1)

    return analysis_df


def detect_field_type(series):
    """
    Detect the data type of a pandas Series to determine appropriate analysis methods.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        str: One of 'numeric', 'categorical', 'datetime', 'text', 'boolean', 'ip_address'
    """
    # Check for nulls
    if series.isna().all():
        return 'empty'

    # Clean the series, removing quotes for string detection
    if series.dtype == object:
        cleaned_series = series.dropna().apply(
            lambda x: x.strip("'\" \t") if isinstance(x, str) else x
        )
    else:
        cleaned_series = series.dropna()

    # Check if empty after cleaning
    if len(cleaned_series) == 0:
        return 'empty'

    # Check if boolean
    if pd.api.types.is_bool_dtype(series) or (
            series.dropna().isin([True, False, 0, 1, '0', '1', 'True', 'False', 'true', 'false']).all() and
            len(series.dropna().unique()) <= 2):
        return 'boolean'

    # Check if datetime
    try:
        if pd.api.types.is_datetime64_dtype(series):
            return 'datetime'
        # Try to convert sample to datetime
        sample = cleaned_series.iloc[0] if not pd.isna(
            cleaned_series.iloc[0]) else cleaned_series.dropna().iloc[0]
        if isinstance(sample, str) and (
            re.match(r'\d{4}-\d{2}-\d{2}', sample) or  # ISO date
            re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', sample)  # ISO datetime
        ):
            # Try converting to datetime
            pd.to_datetime(cleaned_series, errors='raise')
            return 'datetime'
    except (ValueError, TypeError, IndexError):
        pass

    # Check if IP address
    if all(is_valid_ip(str(x)) for x in cleaned_series if not pd.isna(x) and str(x) != ''):
        return 'ip_address'

    # Check if numeric
    try:
        pd.to_numeric(cleaned_series, errors='raise')
        if len(cleaned_series.unique()) < 10 or len(cleaned_series) / len(cleaned_series.unique()) > 5:
            return 'categorical'  # Numeric but acts as categorical
        return 'numeric'
    except (ValueError, TypeError):
        pass

    # Check categorical vs text
    if isinstance(cleaned_series.iloc[0], str):
        # If ratio of unique values to total values is low, likely categorical
        if len(cleaned_series.unique()) < 15 or len(cleaned_series) / len(cleaned_series.unique()) > 3:
            return 'categorical'
        # If average string length is high, likely text
        avg_len = cleaned_series.str.len().mean()
        if avg_len > 20:
            return 'text'

    # Default to categorical
    return 'categorical'


def analyze_field(df, field_name):
    """
    Analyze a specific field and provide appropriate visualizations and statistics.
    
    Args:
        df: DataFrame containing the field
        field_name: Name of the field to analyze
        
    Returns:
        None (displays analysis directly in Streamlit)
    """
    # Get the series
    series = df[field_name]

    # Detect field type
    field_type = detect_field_type(series)

    # Display basic info
    st.write(f"Field: **{field_name}**")
    st.write(f"Type: **{field_type}**")

    # Count non-null values
    non_null_count = series.notna().sum()
    null_count = series.isna().sum()
    st.write(
        f"Non-null values: {non_null_count} ({non_null_count/len(series):.1%})")

    if field_type == 'empty':
        st.write("This field is empty or contains only null values.")
        return

    # Analysis based on field type
    if field_type in ['categorical', 'boolean']:
        value_counts = series.value_counts().reset_index()
        value_counts.columns = [field_name, 'Count']

        st.write("Value counts:")
        st.dataframe(value_counts)

        if len(value_counts) > 20:
            plot_data = value_counts.head(20)
            st.write("(Showing top 20 values)")
        else:
            plot_data = value_counts

        if not plot_data.empty:
            fig = px.bar(
                plot_data,
                x=field_name,
                y='Count',
                title=f"Distribution of {field_name}"
            )

            # Instead of st.plotly_chart, capture clicks
            selected_points = plotly_events(
                fig, click_event=True, select_event=True)

            if selected_points:
                clicked_value = plot_data.iloc[selected_points[0]
                                               ['pointIndex']][field_name]
                st.success(f"You clicked on: {clicked_value}")

                # Filter emails matching clicked value
                matching_df = df[df[field_name] == clicked_value]
                if 'email' in matching_df.columns:
                    emails = matching_df['email'].dropna().unique()
                    st.dataframe(emails)
                    st.download_button(
                        label="Download Selected Emails",
                        data="\n".join(emails),
                        file_name=f"selected_{field_name}_{clicked_value}.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No 'email' column found in the data.")
            else:
                st.plotly_chart(fig)

            # PIE chart also clickable if needed
            if len(plot_data) <= 10:
                fig_pie = px.pie(
                    plot_data,
                    names=field_name,
                    values='Count',
                    title=f"Distribution of {field_name}"
                )
                st.plotly_chart(fig_pie)

    elif field_type == 'numeric':
        # Convert to numeric to ensure analysis works
        numeric_series = pd.to_numeric(series, errors='coerce')

        # Calculate statistics
        stats = {
            'Mean': numeric_series.mean(),
            'Median': numeric_series.median(),
            'Std Dev': numeric_series.std(),
            'Min': numeric_series.min(),
            'Max': numeric_series.max(),
            '25%': numeric_series.quantile(0.25),
            '75%': numeric_series.quantile(0.75)
        }

        # Display statistics
        st.write("Summary statistics:")
        st.dataframe(pd.DataFrame([stats]))

        # Histogram
        fig = px.histogram(
            df,
            x=field_name,
            title=f"Distribution of {field_name}",
            marginal='box'  # Add box plot on top
        )
        st.plotly_chart(fig)

        # Check if there are outliers and offer to exclude them
        q1 = numeric_series.quantile(0.25)
        q3 = numeric_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = numeric_series[(numeric_series < lower_bound) | (
            numeric_series > upper_bound)]

        if len(outliers) > 0:
            st.write(f"Detected {len(outliers)} potential outliers.")
            if st.checkbox(f"Exclude outliers from {field_name} visualization"):
                filtered_series = numeric_series[(numeric_series >= lower_bound) & (
                    numeric_series <= upper_bound)]
                fig = px.histogram(
                    filtered_series,
                    title=f"Distribution of {field_name} (excluding outliers)",
                    marginal='box'
                )
                st.plotly_chart(fig)

    elif field_type == 'datetime':
        # Convert to datetime
        date_series = pd.to_datetime(series, errors='coerce')

        # Display range
        min_date = date_series.min()
        max_date = date_series.max()
        st.write(f"Date range: {min_date} to {max_date}")

        # Group by year-month and count
        try:
            date_counts = date_series.dt.to_period(
                'M').value_counts().sort_index()
            date_counts = date_counts.reset_index()
            date_counts.columns = ['Month', 'Count']
            date_counts['Month'] = date_counts['Month'].astype(str)

            # Plot time series
            fig = px.line(
                date_counts,
                x='Month',
                y='Count',
                title=f"Time series of {field_name}"
            )
            st.plotly_chart(fig)

            # Plot by day of week if appropriate
            if (max_date - min_date).days > 7:
                dow_counts = date_series.dt.day_name().value_counts().reindex([
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                ])

                fig = px.bar(
                    x=dow_counts.index,
                    y=dow_counts.values,
                    title=f"{field_name} by Day of Week"
                )
                st.plotly_chart(fig)

            # Plot by hour of day
            hour_counts = date_series.dt.hour.value_counts().sort_index()

            fig = px.bar(
                x=hour_counts.index,
                y=hour_counts.values,
                title=f"{field_name} by Hour of Day",
                labels={'x': 'Hour', 'y': 'Count'}
            )
            st.plotly_chart(fig)

        except Exception as e:
            st.warning(f"Could not generate time-based visualizations: {e}")

    elif field_type == 'text':
        # Text analysis
        # Get string lengths
        if series.dtype == object:
            string_lengths = series.dropna().astype(str).str.len()

            # Display length statistics
            stats = {
                'Mean Length': string_lengths.mean(),
                'Max Length': string_lengths.max(),
                'Min Length': string_lengths.min()
            }
            st.write("Text length statistics:")
            st.dataframe(pd.DataFrame([stats]))

            # Histogram of text lengths
            fig = px.histogram(
                string_lengths,
                title=f"Distribution of Text Lengths for {field_name}"
            )
            st.plotly_chart(fig)

            # Show sample values
            st.write("Sample values:")
            sample_size = min(5, len(series.dropna()))
            for i, sample in enumerate(series.dropna().sample(sample_size).values):
                st.write(f"{i+1}. {sample}")

    elif field_type == 'ip_address':
        # IP address analysis
        # Clean IPs
        clean_ips = series.apply(clean_ip).dropna()

        # Geolocate IPs
        if len(clean_ips) > 0:
            if st.button("Geolocate IPs"):
                with st.spinner("Geolocating IP addresses..."):
                    geo_results = geolocate_ip_batch(clean_ips.tolist())

                    # Create country distribution
                    countries = [geo_results.get(ip, {}).get(
                        'country') for ip in clean_ips]
                    country_counts = pd.Series(
                        countries).value_counts().reset_index()
                    country_counts.columns = ['Country', 'Count']

                    st.write("Country distribution based on IP geolocation:")
                    st.dataframe(country_counts)

                    # Plot country distribution
                    if not country_counts.empty:
                        fig = px.bar(
                            country_counts.head(15),
                            x='Country',
                            y='Count',
                            title=f"Top Countries for {field_name}"
                        )
                        st.plotly_chart(fig)

        # Show IPv4 vs IPv6 distribution
        ipv4_count = sum(':' not in str(ip) for ip in clean_ips)
        ipv6_count = sum(':' in str(ip) for ip in clean_ips)

        ip_type_data = pd.DataFrame({
            'IP Type': ['IPv4', 'IPv6'],
            'Count': [ipv4_count, ipv6_count]
        })

        fig = px.pie(
            ip_type_data,
            names='IP Type',
            values='Count',
            title=f"IPv4 vs IPv6 Distribution for {field_name}"
        )
        st.plotly_chart(fig)



def extract_categories(data):
    """Collect all unique categories across all users."""
    all_categories = set()
    for email, profile in data.items():
        if 'category_scores' in profile:
            all_categories.update(profile['category_scores'].keys())
    return sorted(all_categories)

def aggregate_category_scores(data):
    """Calculate average category scores across all users."""
    categories = extract_categories(data)
    category_totals = {category: 0 for category in categories}
    category_counts = {category: 0 for category in categories}

    for email, profile in data.items():
        if 'category_scores' in profile:
            for category, score in profile['category_scores'].items():
                category_totals[category] += score
                category_counts[category] += 1

    avg_scores = {}
    for category in categories:
        if category_counts[category] > 0:
            avg_scores[category] = category_totals[category] / category_counts[category]
        else:
            avg_scores[category] = 0

    return avg_scores


def generate_user_profiles_integrated(start_date, end_date, max_users=500, max_queries_per_user=30):
    """Generate user profiles with date range selection"""
    st.subheader("Generate User Profiles")

    # Show date range info
    days_diff = (end_date - start_date).days
    st.info(
        f"Analyzing {days_diff} days of data from {start_date.date()} to {end_date.date()}")

    # Load IAB categories
    categories = load_iab_categories()
    if not categories:
        st.error(
            "IAB categories not found. Please ensure claude-iab-descriptors.json exists in data/iab/")
        return None

    st.success(f"Loaded {len(categories)} IAB categories")

    # Connect to Bedrock
    bedrock_client = create_bedrock_client()
    if not bedrock_client:
        return None

    # Generate/load category embeddings
    with st.spinner("Loading category embeddings..."):
        cat_embeddings = generate_category_embeddings(
            bedrock_client, categories)

    if not cat_embeddings:
        st.error("Failed to generate category embeddings")
        return None

    # Fetch user queries
    with st.spinner(f"Fetching user queries (max {max_users} users, {max_queries_per_user} queries each)..."):
        # Import the mongo client function that returns the client, not collection
        from database.mongo_client import MongoClient
        from config import DB_CONNECTION_STRING

        def get_mongo_client():
            return MongoClient(DB_CONNECTION_STRING)

        unique_df, full_df, meta = fetch_user_queries_with_date_range(
            get_mongo_client, start_date, end_date, max_users, max_queries_per_user
        )

    if full_df.empty:
        st.warning("No queries found for the selected date range")
        if 'error' in meta:
            st.error(f"Database error: {meta['error']}")
        return None

    st.success(
        f"Found {len(full_df)} queries from {meta.get('user_count', 0)} users")

    # Show data summary
    if meta.get('total_queries'):
        st.info(f"Total queries in date range: {meta['total_queries']}")
        if meta.get('filtered_queries'):
            st.info(f"After filtering: {meta['filtered_queries']}")

    # Generate query embeddings
    query_embeddings = load_query_embeddings()
    new_queries = [q for q in unique_df['query'].unique()
                   if q not in query_embeddings]

    if new_queries:
        st.info(f"Generating embeddings for {len(new_queries)} new queries...")
        new_embeddings = generate_query_embeddings_batch(
            bedrock_client, new_queries)
        query_embeddings.update(new_embeddings)
        save_query_embeddings(query_embeddings)

    # Categorize queries
    with st.spinner("Categorizing queries..."):
        categorized_df = categorize_queries(
            full_df, query_embeddings, cat_embeddings)

    # Generate profiles
    profiles = generate_user_profiles(categorized_df)

    if profiles:
    # Save profiles to the same file path
        USER_PROFILES_FILE = "data/email_segments/user_profiles.json"
        os.makedirs(os.path.dirname(USER_PROFILES_FILE), exist_ok=True)
        with open(USER_PROFILES_FILE, 'w') as f:
            json.dump(profiles, f, indent=2)
        st.success(f"Generated and saved {len(profiles)} user profiles")
        # Show top categories
        if 'top_category' in categorized_df.columns:
            st.subheader("Top Categories Distribution")
            top_cats = categorized_df['top_category'].value_counts().head(10)
            fig = px.bar(x=top_cats.values, y=top_cats.index, orientation='h',
                         title="Top 10 Query Categories")
            st.plotly_chart(fig)

    return profiles

def show_email_segments_view():
    """
    Unified Email Segments view: fetches all data and runs all analysis on demand.
    Displays all results with interactive features, updating only on fetch.
    """
    st.title("Email Segments Dashboard")
    
    with st.sidebar:
        exclude_blacklisted = st.checkbox(
            "Exclude Internal/Blacklist Users",
            value=True,
            help="Filter out emails in the blacklist and from blacklisted domains"
        )

    # File paths
    paths = {
        "time_based": TIME_BASED_DATA_FILE,
        "engagement": ENGAGEMENT_DATA_FILE,
        "auth0": AUTH0_DATA_FILE
    }

    # Load previous analysis if present
    for key, path in paths.items():
        if key not in st.session_state:
            st.session_state[key] = load_data_from_csv(path)

    # Unified fetch button
    if st.button("Fetch All Data and Analyze", key="fetch_all"):
        with st.spinner("Fetching and analyzing all data..."):
            # MongoDB user activity segments
            collection = get_db_connection()
            now = pd.Timestamp.now(tz='UTC')
            email_data = fetch_all_emails_with_timestamps(collection)

            time_based_df = filter_emails_exclusively(
                email_data, now, exclude_blacklisted=exclude_blacklisted)
            engagement_df = classify_engagement_with_pandas(
                email_data, now, exclude_blacklisted=exclude_blacklisted)
            time_based_df['email'] = time_based_df['userEmail']
            engagement_df['email'] = engagement_df['userEmail'] 
            st.session_state['time_based'] = time_based_df
            st.session_state['engagement'] = engagement_df
            save_data_to_csv(time_based_df, TIME_BASED_DATA_FILE)
            save_data_to_csv(engagement_df, ENGAGEMENT_DATA_FILE)

            # Auth0 export
            file_url = export_all_users()
            if file_url:
                response = requests.get(file_url)
                if response.ok:
                    try:
                        with gzip.open(io.BytesIO(response.content), 'rb') as f:
                            csv_content = f.read().decode('utf-8')
                        df = pd.read_csv(io.StringIO(csv_content))
                        analysis_df = analyze_auth0_export(df)
                                        
                        analysis_df['email'] = analysis_df['email'].str.replace("'", "")
                        st.session_state['auth0'] = analysis_df

                        save_data_to_csv(analysis_df, AUTH0_DATA_FILE)
                    except Exception as e:
                        st.error(f"Error processing Auth0 export: {e}")
                else:
                    st.error("Failed to retrieve Auth0 export file.")
            else:
                st.error("Failed to export users from Auth0.")

    # Last fetched time
    for key, label in zip(
        ["time_based", "engagement", "auth0"],
        ["Time-Based Segments", "Engagement Segments", "Auth0 Export"]
    ):
        last_fetched = get_last_fetched_time(paths[key])
        if last_fetched:
            st.caption(
                f"{label} last fetched: {last_fetched.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # --- DISPLAY ANALYSIS ---
    col1, col2 = st.columns(2)

    # Time-Based Filter
    with col1:
        st.subheader("Time-Based Filter")
        st.markdown("User activity within specific time ranges.")
        time_based_df = st.session_state['time_based']
        if time_based_df is not None:
            try:
                segment_counts = time_based_df['Time_Based_Segment'].value_counts()
            except:
                segment_counts = time_based_df['Category'].value_counts(
                )
            fig = px.pie(
                names=segment_counts.index,
                values=segment_counts.values,
                title="User Distribution by Time Segment"
            )
            st.plotly_chart(fig)
            download_link = get_table_download_link(
                time_based_df,
                "time_based_filtered_users.csv",
                "Download Time-Based Users as CSV"
            )
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.info("No time-based segment data. Click 'Fetch All Data and Analyze'.")

    # Engagement Levels
    with col2:
        st.subheader("Engagement Level Segments")
        st.markdown("Classify user engagement based on query activity.")
        engagement_df = st.session_state['engagement']
        if engagement_df is not None:
            try:
                engagement_counts = engagement_df['Engagement_Segment'].value_counts(
            )
            except:
                engagement_counts = engagement_df['Engagement'].value_counts()
            fig = px.pie(
                names=engagement_counts.index,
                values=engagement_counts.values,
                title="User Distribution by Engagement Level",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig)
            download_link = get_table_download_link(
                engagement_df,
                "engagement_level_users.csv",
                "Download Engagement-Level Users as CSV"
            )
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.info("No engagement segment data. Click 'Fetch All Data and Analyze'.")

    # --- AUTH0 EXPORT / ENRICHED USER DATA ---
    st.subheader("Auth0 User Export and Analysis")


    auth0_df = st.session_state.get('auth0')
    time_df = st.session_state.get('time_based')
    eng_df = st.session_state.get('engagement')

    # 1) Make sure you actually have the dataframes
    if time_df is None or eng_df is None or auth0_df is None:
        st.info("No data yet: click Fetch All Data and Analyze.")
        return

    # 2) Normalize column names *before* merge guard
    if 'Category' in time_df.columns:
        time_df = time_df.rename(columns={'Category': 'Time_Based_Segment'})
    if 'Engagement' in eng_df.columns:
        eng_df = eng_df.rename(columns={'Engagement': 'Engagement_Segment'})

    # # 3) Debug prints in UI
    # st.write("→ time_df cols:", time_df.columns.tolist())
    # st.write("→ eng_df  cols:", eng_df.columns.tolist())


    # 4) Now do your one-time mergegment columns, do the merge now:
    if ('Time_Based_Segment' not in auth0_df.columns
            or 'Engagement_Segment' not in auth0_df.columns):

        merged = auth0_df.copy()
        # standardize the email key
        for df in (merged, time_df, eng_df):
            df['email'] = df['email'].str.lower().str.strip()

        # now safe to subset and merge
        merged = merged.merge(
            time_df[['email', 'Time_Based_Segment']],
            on='email', how='left'
        )
        merged = merged.merge(
            eng_df[['email', 'Engagement_Segment']],
            on='email', how='left'
        )

        # drop any accidental dupes
        merged = merged.loc[:, ~merged.columns.duplicated()]
        st.session_state['auth0'] = merged
        auth0_df = merged
    else:
        # already merged, just pull from state
        auth0_df = st.session_state['auth0']


    # 5) Finally, display the merged result
    auth0_merged = st.session_state['auth0']
    
    # Display the dataframe with unique columns (avoid showing duplicates)
    display_cols = list(dict.fromkeys(auth0_merged.columns))
    # st.dataframe(auth0_merged[display_cols])
    
    # Provide download option
    csv_data = auth0_merged.to_csv(index=False)
    st.download_button(
        label="Download Enhanced Data CSV",
        data=csv_data,
        file_name=f"auth0_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    ran = True
    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
        "Enhanced Data", "Country Analysis", "Account Activity", "User Profiles"
    ])
    
    with analysis_tab1:
        st.dataframe(auth0_merged)
        
    with analysis_tab2:
        if 'declared_country' in auth0_merged.columns and 'ip_country' in auth0_merged.columns:
            st.subheader("Declared vs. Inferred Country")
            status_counts = auth0_merged['country_status'].value_counts().reindex(
                ['Match', 'Mismatch', 'Only Declared', 'Only Inferred', 'Both Missing'],
                fill_value=0
            )
            status_df = pd.DataFrame({
                'Status': status_counts.index,
                'Count': status_counts.values
            })
            fig = px.pie(
                status_df,
                names='Status',
                values='Count',
                title="Declared vs. IP-Inferred Country Status",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig)
            comparison_df = auth0_merged[
                ['email', 'standardized_declared_country', 'standardized_inferred_country', 'country_status']
            ].dropna(subset=['standardized_declared_country', 'standardized_inferred_country'], how='all')
            if not comparison_df.empty:
                st.write("Detailed Country Comparison:")
                st.dataframe(comparison_df.rename(columns={
                    'standardized_declared_country': 'Declared Country',
                    'standardized_inferred_country': 'Inferred Country'
                }))
            st.subheader("User Distribution by Country")
            country_counts = auth0_merged['standardized_country'].value_counts().reset_index()
            country_counts.columns = ['Country', 'Count']
            country_counts = country_counts.sort_values('Count', ascending=False)
            fig = px.bar(
                country_counts.head(15),
                x='Country',
                y='Count',
                title="Top 15 Countries by User Count"
            )
            st.plotly_chart(fig)
            st.dataframe(country_counts)
            
    with analysis_tab3:
        if 'created_at' in auth0_merged.columns:
            auth0_merged['created_at'] = pd.to_datetime(auth0_merged['created_at'], errors='coerce')
            created_counts = auth0_merged.set_index('created_at').resample('M').size().reset_index()
            created_counts.columns = ['Date', 'New Users']
            fig = px.line(
                created_counts,
                x='Date',
                y='New Users',
                title="User Account Creation Over Time"
            )
            st.plotly_chart(fig)
        if 'logins_count' in auth0_merged.columns:
            login_stats = {
                "Avg. Logins": auth0_merged['logins_count'].mean(),
                "Max Logins": auth0_merged['logins_count'].max(),
                "Users with 1 Login": (auth0_merged['logins_count'] == 1).sum(),
                "Users with >5 Logins": (auth0_merged['logins_count'] > 5).sum(),
            }
            st.subheader("Login Activity")
            st.write(login_stats)
            fig = px.histogram(
                auth0_merged,
                x='logins_count',
                nbins=20,
                title="Distribution of Login Counts"
            )
            st.plotly_chart(fig)
            
    with analysis_tab4:
        st.subheader("User Profiles Analysis")

        # Add profile generation section with performance controls
        st.subheader("Generate New Profiles")

        col1, col2 = st.columns(2)
        with col1:
            profile_start_date = st.date_input(
                "Profile Start Date",
                # Default to 7 days for better performance
                value=datetime.now().date() - timedelta(days=7),
                max_value=datetime.now().date()
            )
        with col2:
            profile_end_date = st.date_input(
                "Profile End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )

        # Performance controls
        with st.expander("Advanced Settings"):
            col3, col4 = st.columns(2)
            with col3:
                max_users = st.number_input(
                    "Max Users",
                    min_value=10,
                    max_value=2000,
                    value=500,
                    help="Reduce for faster processing"
                )
            with col4:
                max_queries = st.number_input(
                    "Max Queries per User",
                    min_value=5,
                    max_value=100,
                    value=30,
                    help="Reduce for faster processing"
                )

        # Show estimated date range
        days_selected = (profile_end_date - profile_start_date).days
        if days_selected > 7:
            st.warning(
                f"may take a while. consider shorter range for faster performance")

        if st.button("Generate User Profiles", key="generate_profiles"):
            start_dt = datetime.combine(
                profile_start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_dt = datetime.combine(
                profile_end_date, datetime.min.time()).replace(tzinfo=timezone.utc)

            new_profiles = generate_user_profiles_integrated(
                start_dt, end_dt, max_users, max_queries)
            if new_profiles:
                # Clear the cached data so it reloads fresh data
                load_user_profiles.clear()
                st.rerun()

        # Load user profiles data (existing code continues...)
        profiles_data = load_user_profiles()
        if ('total_queries' not in auth0_df.columns and profiles_data):
            auth0_df = merge_with_user_profiles(auth0_df, profiles_data)
            st.session_state['auth0'] = auth0_df

        if not profiles_data:
            st.info(
                "No user profiles data available. Ensure user_profiles.json exists in the current or parent directory.")
        else:
            # Profile analysis options
            profile_analysis_type = st.selectbox(
                "Select Profile Analysis Type",
                ["Overview", "Category Analysis", "User Details"]
            )

            if profile_analysis_type == "Overview":
                col1, col2, col3 = st.columns(3)

                total_users = len(profiles_data)
                total_queries = sum(profile.get('total_queries', 0)
                                    for profile in profiles_data.values())
                total_analyzed = sum(profile.get('analysed_queries', 0)
                                     for profile in profiles_data.values())

                col1.metric("Total Profiled Users", total_users)
                col2.metric("Total Queries", total_queries)
                col3.metric("Total Analyzed Queries", total_analyzed)

                # Add CSV download section
                st.subheader("Download User Profiles Data")

                if st.button("Generate CSV Download", key="generate_profiles_csv"):
                    with st.spinner("Preparing user profiles CSV..."):
                        profiles_csv_df = create_user_profiles_csv(
                            profiles_data)

                        # Show preview
                        st.write("Preview of CSV data:")
                        st.dataframe(profiles_csv_df.head(10))

                        # Create download button
                        csv_data = profiles_csv_df.to_csv(index=False)
                        st.download_button(
                            label="Download User Profiles CSV",
                            data=csv_data,
                            file_name=f"user_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_profiles_csv"
                        )

                        st.success(
                            f"CSV ready for download with {len(profiles_csv_df)} user profiles!")

                # User activity distribution
                user_activity = pd.DataFrame({
                    'Email': list(profiles_data.keys()),
                    'Queries': [profile.get('total_queries', 0) for profile in profiles_data.values()]
                })

                fig = px.histogram(
                    user_activity,
                    x='Queries',
                    nbins=20,
                    title="User Query Distribution"
                )
                st.plotly_chart(fig)

            elif profile_analysis_type == "Category Analysis":
                avg_scores = aggregate_category_scores(profiles_data)
                sorted_categories = sorted(
                    avg_scores.items(), key=lambda x: x[1], reverse=True)

                categories_df = pd.DataFrame(sorted_categories, columns=[
                    'Category', 'Average Score'])

                fig = px.bar(
                    categories_df,
                    x='Average Score',
                    y='Category',
                    orientation='h',
                    title='Average Category Interest Scores'
                )
                st.plotly_chart(fig)

            elif profile_analysis_type == "User Details":
                user_emails = list(profiles_data.keys())
                selected_user = st.selectbox("Select a user", user_emails)

                if selected_user and selected_user in profiles_data:
                    user_data = profiles_data[selected_user]

                    st.subheader(f"Profile for: {selected_user}")

                    col1, col2 = st.columns(2)
                    col1.metric("Total Queries",
                                user_data.get('total_queries', 0))
                    col2.metric("Analyzed Queries",
                                user_data.get('analysed_queries', 0))

                    # Category scores
                    if 'category_scores' in user_data:
                        st.subheader("Category Scores")
                        sorted_categories = sorted(
                            user_data['category_scores'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )

                        categories_df = pd.DataFrame(
                            sorted_categories, columns=['Category', 'Score'])
                        fig = px.bar(
                            categories_df,
                            x='Score',
                            y='Category',
                            orientation='h',
                            title='Category Interest Scores'
                        )
                        st.plotly_chart(fig)

                    # Top topics
                    if 'top_topics' in user_data:
                        st.subheader("Top Topics")
                        for topic in user_data['top_topics']:
                            st.write(f"• {topic}")

                    # Recent queries
                    if 'recent_queries' in user_data:
                        st.subheader("Recent Queries")
                        for query in user_data['recent_queries'][:10]:  # Show first 10
                            st.write(f"• {query}")


def create_user_profiles_csv(profiles_data):
    """Create a CSV-ready DataFrame from user profiles data with top topics and probabilities."""
    csv_rows = []
    
    for email, profile in profiles_data.items():
        row = {'email': email}
        
        # Get top topics and their scores/probabilities
        if 'top_topics' in profile and profile['top_topics']:
            # Add up to 3 top topics
            for i, topic in enumerate(profile['top_topics'][:3]):
                row[f'topic{i+1}'] = topic
                
                # Try to get probability/score for this topic from category_scores
                if 'category_scores' in profile:
                    # Find matching category score (topics might match category names)
                    topic_score = profile['category_scores'].get(topic, 0)
                    row[f'prob{i+1}'] = topic_score
                else:
                    row[f'prob{i+1}'] = 0
        
        # Fill empty topic slots
        for i in range(len(profile.get('top_topics', [])), 3):
            row[f'topic{i+1}'] = ''
            row[f'prob{i+1}'] = 0
        
        # Add query examples (first 3 recent queries)
        if 'recent_queries' in profile and profile['recent_queries']:
            query_examples = profile['recent_queries'][:3]
            row['query_examples'] = ' | '.join(query_examples)
        else:
            row['query_examples'] = ''
        
        # Add other useful metrics
        row['total_queries'] = profile.get('total_queries', 0)
        row['analysed_queries'] = profile.get('analysed_queries', 0)
        
        csv_rows.append(row)
    
    # Create DataFrame with consistent column order
    columns = ['email', 'total_queries', 'analysed_queries', 
               'topic1', 'prob1', 'topic2', 'prob2', 'topic3', 'prob3', 
               'query_examples']
    
    df = pd.DataFrame(csv_rows)
    
    # Ensure all columns exist (fill missing with empty/0)
    for col in columns:
        if col not in df.columns:
            if 'prob' in col or col in ['total_queries', 'analysed_queries']:
                df[col] = 0
            else:
                df[col] = ''
    
    # Reorder columns
    df = df[columns]
    
    return df


def merge_with_user_profiles(auth0_df, profiles_data):
    """Merge Auth0 data with user profiles data."""
    if not profiles_data:
        return auth0_df

    # Create profiles DataFrame
    profiles_rows = []
    for email, profile in profiles_data.items():
        profiles_rows.append({
            'email': email.lower().strip(),
            'total_queries': profile.get('total_queries', 0),
            'analysed_queries': profile.get('analysed_queries', 0),
            'top_categories': ', '.join(sorted(profile.get('category_scores', {}).keys())[:5]),
            'top_topics': ', '.join(profile.get('top_topics', [])[:3])
        })

    profiles_df = pd.DataFrame(profiles_rows)

    # Merge with auth0 data
    auth0_df['email'] = auth0_df['email'].str.lower().str.strip()
    merged_df = auth0_df.merge(
        profiles_df, on='email', how='left', suffixes=('', '_profile'))

    return merged_df

if __name__ == "__main__":
    show_email_segments_view()