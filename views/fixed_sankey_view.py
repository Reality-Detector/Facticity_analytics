"""
Fixed Sankey diagram view that includes both original and time-bound analyses.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone

from utils.sankey_utils import (
    create_combined_visualization,
    create_sankey_diagram,
    get_mongodb_data
)
from utils.fixed_sankey_utils import (
    create_time_bound_sankey,
    generate_time_bound_analysis
)
from database.auth0_client import get_auth0_user_list
from database.mongo_client import is_valid_user
from utils.posthog_utils import get_posthog_data


# Cache Auth0 export to avoid multiple API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_auth0_data():
    """Get cached Auth0 user data to avoid multiple API calls."""
    return get_auth0_user_list()


def filter_auth0_data_by_paid_status(auth0_df, paid_emails):
    """Filter Auth0 users by paid status."""
    paid_emails_set = {email.lower().strip() for email in paid_emails}

    # Add 'is_paid' column to auth0_users
    auth0_df['is_paid'] = auth0_df['email'].apply(
        lambda email: str(email).lower().strip() in paid_emails_set
    )

    # Also add userEmail column for consistency with other functions
    auth0_df['userEmail'] = auth0_df['email']

    return auth0_df


def create_paid_nonpaid_time_bound_sankey(start_date, end_date, auth0_df, paid_emails, exclude_blacklisted):
    """Create a time-bound Sankey diagram with paid/non-paid user distinction."""
    # Use ALL Auth0 users regardless of creation/login date
    auth0_filtered = auth0_df.copy()

    # Add paid status
    auth0_filtered = filter_auth0_data_by_paid_status(
        auth0_filtered, paid_emails)

    if exclude_blacklisted:
        # Filter valid Auth0 users (email validation only)
        auth0_valid_users = auth0_filtered[auth0_filtered.apply(
            lambda row: is_valid_user(row['email']), axis=1)].copy()
    else:
        auth0_valid_users = auth0_filtered.copy()

    # Split auth0_valid_users into paid and non-paid
    auth0_valid_paid = auth0_valid_users[auth0_valid_users['is_paid']].copy()
    auth0_valid_non_paid = auth0_valid_users[~auth0_valid_users['is_paid']].copy(
    )

    # Get query data for the period (this already has date filtering)
    mongodb_df = get_mongodb_data(start_date, end_date)

    # Process paid users
    active_paid_users = mongodb_df[mongodb_df['userEmail'].isin(
        auth0_valid_paid['email'])]
    # st.write(active_paid_users)
    # st.write(auth0_valid_paid)
    active_paid_user_count = len(
        active_paid_users['userEmail'].unique()) if not active_paid_users.empty else 0
    inactive_paid_user_count = len(auth0_valid_paid) - active_paid_user_count  # there may be three emails gievn but four shown in the graph bc for example william cardwell has one account with auth0 and one with google auth

    # Process non-paid users
    active_non_paid_users = mongodb_df[mongodb_df['userEmail'].isin(
        auth0_valid_non_paid['email'])]
    active_non_paid_user_count = len(active_non_paid_users['userEmail'].unique(
    )) if not active_non_paid_users.empty else 0
    inactive_non_paid_user_count = len(
        auth0_valid_non_paid) - active_non_paid_user_count

    # Count anonymous queries (queries without a valid userEmail)
    anonymous_queries = mongodb_df[~mongodb_df['userEmail'].isin(
        auth0_valid_users['email'])]
    anonymous_query_count = anonymous_queries['count'].sum(
    ) if not anonymous_queries.empty else 0

    # Query distribution for paid users
    def categorize_queries(count):
        if count == 1:
            return '1 Query'
        elif count <= 5:
            return '2-5 Queries'
        else:
            return '6+ Queries'

    # Process paid active users
    paid_user_data = pd.merge(
        active_paid_users,
        auth0_valid_paid[['email', 'logins_count']],
        left_on='userEmail',
        right_on='email',
        how='left'
    ) if not active_paid_users.empty else pd.DataFrame()

    if not paid_user_data.empty:
        paid_user_data['query_category'] = paid_user_data['count'].apply(
            categorize_queries)
        paid_query_distribution = paid_user_data['query_category'].value_counts(
        ).to_dict()
        paid_single_query_users = paid_query_distribution.get('1 Query', 0)
        paid_few_query_users = paid_query_distribution.get('2-5 Queries', 0)
        paid_many_query_users = paid_query_distribution.get('6+ Queries', 0)
        paid_registered_query_count = paid_user_data['count'].sum()
    else:
        paid_single_query_users = 0
        paid_few_query_users = 0
        paid_many_query_users = 0
        paid_registered_query_count = 0

    # Process non-paid active users
    non_paid_user_data = pd.merge(
        active_non_paid_users,
        auth0_valid_non_paid[['email', 'logins_count']],
        left_on='userEmail',
        right_on='email',
        how='left'
    ) if not active_non_paid_users.empty else pd.DataFrame()

    if not non_paid_user_data.empty:
        non_paid_user_data['query_category'] = non_paid_user_data['count'].apply(
            categorize_queries)
        non_paid_query_distribution = non_paid_user_data['query_category'].value_counts(
        ).to_dict()
        non_paid_single_query_users = non_paid_query_distribution.get(
            '1 Query', 0)
        non_paid_few_query_users = non_paid_query_distribution.get(
            '2-5 Queries', 0)
        non_paid_many_query_users = non_paid_query_distribution.get(
            '6+ Queries', 0)
        non_paid_registered_query_count = non_paid_user_data['count'].sum()
    else:
        non_paid_single_query_users = 0
        non_paid_few_query_users = 0
        non_paid_many_query_users = 0
        non_paid_registered_query_count = 0

    # Create metrics dictionary for paid vs non-paid breakdown
    metrics = {
        'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'total_users': len(auth0_valid_users),
        'paid_users': len(auth0_valid_paid),
        'non_paid_users': len(auth0_valid_non_paid),
        'active_paid_users': active_paid_user_count,
        'inactive_paid_users': inactive_paid_user_count,
        'active_non_paid_users': active_non_paid_user_count,
        'inactive_non_paid_users': inactive_non_paid_user_count,
        'paid_single_query_users': paid_single_query_users,
        'paid_few_query_users': paid_few_query_users,
        'paid_many_query_users': paid_many_query_users,
        'non_paid_single_query_users': non_paid_single_query_users,
        'non_paid_few_query_users': non_paid_few_query_users,
        'non_paid_many_query_users': non_paid_many_query_users,
        'paid_query_count': paid_registered_query_count,
        'non_paid_query_count': non_paid_registered_query_count,
        'anonymous_query_count': anonymous_query_count,
        'total_query_count': paid_registered_query_count + non_paid_registered_query_count + anonymous_query_count
    }

    # Create Sankey diagram
    import plotly.graph_objects as go

    node_labels = [
        "All Users",               # 0
        "Paid Users",              # 1
        "Non-Paid Users",          # 2
        "Active Paid",             # 3
        "Inactive Paid",           # 4
        "Active Non-Paid",         # 5
        "Inactive Non-Paid",       # 6
        "Paid 1 Query",            # 7
        "Paid 2-5 Queries",        # 8
        "Paid 6+ Queries",         # 9
        "Non-Paid 1 Query",        # 10
        "Non-Paid 2-5 Queries",    # 11
        "Non-Paid 6+ Queries",     # 12
        "Paid Queries",            # 13
        "Non-Paid Queries",        # 14
        "Anonymous Queries",       # 15
        "Total Queries"            # 16
    ]

    source = [
        0, 0,                   # All Users -> Paid, Non-Paid
        1, 1,                   # Paid -> Active, Inactive
        2, 2,                   # Non-Paid -> Active, Inactive
        3, 3, 3,                # Active Paid -> Query Categories
        5, 5, 5,                # Active Non-Paid -> Query Categories
        7, 8, 9,                # Paid Query Categories -> Paid Queries
        10, 11, 12,             # Non-Paid Query Categories -> Non-Paid Queries
        13, 14, 15              # All Query Types -> Total Queries
    ]

    target = [
        1, 2,                   # All Users -> Paid, Non-Paid
        3, 4,                   # Paid -> Active, Inactive
        5, 6,                   # Non-Paid -> Active, Inactive
        7, 8, 9,                # Active Paid -> Query Categories
        10, 11, 12,             # Active Non-Paid -> Query Categories
        13, 13, 13,             # Paid Query Categories -> Paid Queries
        14, 14, 14,             # Non-Paid Query Categories -> Non-Paid Queries
        16, 16, 16              # All Query Types -> Total Queries
    ]

    values = [
        metrics['paid_users'],                  # All Users -> Paid
        metrics['non_paid_users'],              # All Users -> Non-Paid
        metrics['active_paid_users'],           # Paid -> Active
        metrics['inactive_paid_users'],         # Paid -> Inactive
        metrics['active_non_paid_users'],       # Non-Paid -> Active
        metrics['inactive_non_paid_users'],     # Non-Paid -> Inactive
        metrics['paid_single_query_users'],     # Active Paid -> 1 Query
        metrics['paid_few_query_users'],        # Active Paid -> 2-5 Queries
        metrics['paid_many_query_users'],       # Active Paid -> 6+ Queries
        metrics['non_paid_single_query_users'],  # Active Non-Paid -> 1 Query
        # Active Non-Paid -> 2-5 Queries
        metrics['non_paid_few_query_users'],
        metrics['non_paid_many_query_users'],   # Active Non-Paid -> 6+ Queries
        metrics['paid_single_query_users'],     # Paid 1 Query -> Paid Queries
        # Paid 2-5 Queries -> Paid Queries
        metrics['paid_few_query_users'],
        # Paid 6+ Queries -> Paid Queries
        metrics['paid_many_query_users'],
        # Non-Paid 1 Query -> Non-Paid Queries
        metrics['non_paid_single_query_users'],
        # Non-Paid 2-5 Queries -> Non-Paid Queries
        metrics['non_paid_few_query_users'],
        # Non-Paid 6+ Queries -> Non-Paid Queries
        metrics['non_paid_many_query_users'],
        metrics['paid_query_count'],            # Paid Queries -> Total Queries
        # Non-Paid Queries -> Total Queries
        metrics['non_paid_query_count'],
        # Anonymous Queries -> Total Queries
        metrics['anonymous_query_count']
    ]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=values
        )
    )])

    # Update layout
    fig.update_layout(
        title_text=f"Paid vs Non-Paid User Activity: {metrics['period']}",
        font_size=12
    )

    return fig, metrics


def process_week_data_with_auth0(start_date, end_date, auth0_df, exclude_blacklisted):
    """Process all data sources for a specific week using pre-loaded Auth0 data."""
    # Filter Auth0 data for this period
    auth0_filtered = auth0_df.copy()
    if "last_login" in auth0_filtered.columns:
        auth0_filtered["last_login"] = pd.to_datetime(
            auth0_filtered["last_login"], errors="coerce", utc=True)
    if "created_at" in auth0_filtered.columns:
        auth0_filtered["created_at"] = pd.to_datetime(
            auth0_filtered["created_at"], errors="coerce", utc=True)

    # Filter by date range
    mask = ((auth0_filtered["created_at"] >= start_date) & (auth0_filtered["created_at"] <= end_date)) | \
           ((auth0_filtered["last_login"] >= start_date)
            & (auth0_filtered["last_login"] <= end_date))
    auth0_users = auth0_filtered[mask].copy()

    # Prepare userEmail field if needed
    if 'userEmail' not in auth0_users.columns:
        auth0_users['userEmail'] = auth0_users['email']

    # Get PostHog sessions data
    posthog_df = get_posthog_data(start_date, end_date)
    if posthog_df.empty:
        return None

    # Get MongoDB query data
    mongodb_df = get_mongodb_data(start_date, end_date)

    # Check if mongodb_df has expected columns
    if mongodb_df.empty or 'userEmail' not in mongodb_df.columns or 'count' not in mongodb_df.columns:
        # Create a default empty dataframe with expected columns
        mongodb_df = pd.DataFrame(columns=["userEmail", "count"])

    if exclude_blacklisted:
        # Filter valid Auth0 users (exclude blacklisted domains/emails)
        auth0_valid_users = auth0_users[auth0_users.apply(
            lambda row: is_valid_user(row['userEmail']), axis=1)].copy()
    else:
        auth0_valid_users = auth0_users.copy()

    # Process session data
    session_df = posthog_df.copy()
    session_df["ip_address"] = session_df["ip_addresses"].apply(
        lambda ips: ips[0] if isinstance(ips, list) and len(ips) > 0 else None
    )


    # For PostHog IP matching, we need to handle shared IPs
    # Identify shared IP addresses
    if 'last_ip' in auth0_valid_users.columns:
        # Extract ip_address from last_ip if not already present
        if 'ip_address' not in auth0_valid_users.columns:
            # Handle different formats of last_ip (quoted strings, plain IPs, etc.)
            auth0_valid_users['ip_address'] = auth0_valid_users['last_ip'].astype(str).apply(
                lambda x: x.strip("'\"") if isinstance(x, str) else x
            )

        # Only proceed if we successfully created ip_address column
        if 'ip_address' in auth0_valid_users.columns:
            auth0_shared_ip_addresses = auth0_valid_users[
                auth0_valid_users.groupby('ip_address')[
                    'ip_address'].transform('count') > 1
            ]

            # Mark sessions that come from a shared IP address
            session_df["is_shared_ip"] = session_df["ip_address"].isin(
                auth0_shared_ip_addresses["ip_address"])

            # Sort sessions by latest session_start
            session_df = session_df.sort_values(
                by="session_start", ascending=False)

            # For sessions from shared IPs, keep only a limited number
            shared_ips_df = session_df[session_df["is_shared_ip"] == True].copy()
            filtered_shared_ips = pd.DataFrame()
            if not shared_ips_df.empty:
                filtered_shared_ips = shared_ips_df.groupby("ip_address", group_keys=False).apply(
                    lambda x: x.head(
                        auth0_shared_ip_addresses["ip_address"].value_counts().get(x.name, 1))
                )

            # For non-shared IPs, keep the first occurrence per IP
            non_shared_ips_df = session_df[~session_df["is_shared_ip"]].drop_duplicates(
                subset=["ip_address"], keep="first")

            # Combine both subsets to form final_df
            final_df = pd.concat([filtered_shared_ips, non_shared_ips_df])
            final_df = final_df.drop(
                columns=["is_shared_ip"]).reset_index(drop=True)

            # Flag each session as logged in if its IP is found in the Auth0 user list
            # FIX: Use auth0_valid_users instead of auth0_users
            auth0_ip_set = set(auth0_valid_users['ip_address'].dropna().unique())
            final_df['logged_in'] = final_df['ip_address'].apply(
                lambda ip: ip in auth0_ip_set)
        else:
            # If ip_address column creation failed, create default final_df
            final_df = session_df.copy()
            final_df['logged_in'] = False
    else:
        # If last_ip not available, create a default final_df
        final_df = session_df.copy()
        final_df['logged_in'] = False

    # Merge Auth0 users with MongoDB query counts
    merged_df = pd.merge(auth0_valid_users, mongodb_df,
                         on="userEmail", how="left")
    merged_df["count"] = merged_df["count"].fillna(0).astype(int)

    # Define bins based on query count
    def assign_bin(count):
        if count == 0:
            return "0"
        elif count == 1:
            return "1"
        else:
            return ">1"

    merged_df["bin"] = merged_df["count"].apply(assign_bin)

    # Count new vs. returning users
    new_users_count = auth0_valid_users[auth0_valid_users['logins_count'] == 1].shape[0]
    returning_users_count = auth0_valid_users[auth0_valid_users['logins_count'] > 1].shape[0]

    # Get counts for event bins
    bin_1 = merged_df['bin'].value_counts().get('1', 0)
    bin_more = merged_df['bin'].value_counts().get('>1', 0)
    bin_0 = merged_df['bin'].value_counts().get('0', 0)

    # Safely calculate query counts
    not_logged_in_total = 0
    logged_in_total = 0

    # Check if we have valid data to perform calculations
    if not mongodb_df.empty and 'userEmail' in mongodb_df.columns and 'count' in mongodb_df.columns:
        # Filter to get non-logged in queries
        not_logged_in_mask = ~mongodb_df['userEmail'].isin(
            auth0_valid_users['userEmail'])
        not_logged_in_df = mongodb_df[not_logged_in_mask]
        if 'count' in not_logged_in_df.columns:
            not_logged_in_total = not_logged_in_df['count'].sum()

        # Calculate logged in total
        logged_in_total = merged_df['count'].sum()

    # Return all metrics needed for the Sankey diagram
    return {
        'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'logged_in_count': final_df['logged_in'].value_counts().get(True, 0),
        'not_logged_in_count': final_df['logged_in'].value_counts().get(False, 0),
        'new_users_count': new_users_count,
        'returning_users_count': returning_users_count,
        'bin_1': bin_1,
        'bin_more': bin_more,
        'bin_0': bin_0,
        'not_logged_in_total': not_logged_in_total,
        'logged_in_total': logged_in_total
    }


def generate_week_over_week_analysis_with_auth0(num_weeks, auth0_df, exclude_blacklisted):
    """
    Generate Sankey diagrams for the specified number of past weeks using cached Auth0 data.
    
    Args:
        num_weeks: Number of weeks to analyze
        auth0_df: Pre-loaded Auth0 DataFrame
        
    Returns:
        tuple: (Figures, Summary DataFrame)
    """
    now = datetime.now(timezone.utc)

    # Create a list to store weekly metrics
    weekly_metrics = []
    weekly_figures = []

    # Process data for each week
    for i in range(num_weeks):
        end_date = now - timedelta(days=7*i)
        start_date = end_date - timedelta(days=7)

        metrics = process_week_data_with_auth0(start_date, end_date, auth0_df, exclude_blacklisted)
        if metrics:
            weekly_metrics.append(metrics)
            weekly_figures.append(create_sankey_diagram(metrics))

    # Create a summary DataFrame for easy comparison
    if weekly_metrics:
        summary_df = pd.DataFrame(weekly_metrics)

        # Calculate week-over-week changes
        for col in ['logged_in_count', 'not_logged_in_count', 'new_users_count',
                    'returning_users_count', 'bin_1', 'bin_more', 'bin_0',
                    'not_logged_in_total', 'logged_in_total']:
            summary_df[f'{col}_wow_change'] = summary_df[col].pct_change(
                -1) * 100

        return weekly_figures, summary_df

    return [], pd.DataFrame()


def show_fixed_sankey_view():
    """
    Display both original and time-bound Sankey diagrams.
    """
    st.title("User Flow Analytics")

    st.markdown("""
    1. Time bound method removes the ip matching 
    2. Original Posthog flow uses last_login field in auth0 to determine if user was logged in or not, hence older weeks may misclassify more logged in users as not logged in (method is best for last __ time period)
                
    - queries from @aiseer.co emails and blacklist emails are removed from calculations
    """)

    # Create sidebar for common settings
    with st.sidebar:
        exclude_blacklisted = st.checkbox("Exclude Internal/Blacklist Users", value=True,
                                          help="Filter out emails in the blacklist and from blacklisted domains")
        st.subheader("Analysis Settings")
        

        # Number of weeks to analyze
        num_weeks = st.slider(
            "Number of weeks to analyze",
            min_value=2,
            max_value=4,
            value=3
        )
        # File path for saved paid emails
    PAID_EMAILS_FILE = "paid_users.txt"

    # Try to load existing paid emails from file
    try:
        with open(PAID_EMAILS_FILE, "r") as f:
            saved_emails = f.read()
    except FileNotFoundError:
        saved_emails = ""

    # Paid user configuration with previously saved emails pre-filled
    paid_users_file = st.text_area(
        "Enter paid user emails (one per line)",
        value=saved_emails,
        height=100,
        help="Enter email addresses of paid users, one per line. These will be saved for future use."
    )

    # Save button for paid emails
    if st.button("Save Paid User List"):
        try:
            with open(PAID_EMAILS_FILE, "w") as f:
                f.write(paid_users_file)
            st.success(
                f"Saved {len(paid_users_file.splitlines())} email addresses to {PAID_EMAILS_FILE}")
        except Exception as e:
            st.error(f"Error saving emails: {e}")

    # Convert text area to list of emails
    paid_emails = [email.strip()
                   for email in paid_users_file.splitlines() if email.strip()]

    # Create tabs for different visualizations
    tabs = st.tabs(
        ["Time-Bound Auth0 Flow", "Original PostHog Flow", "Paid vs Non-Paid Flow"])

    # Cache Auth0 data for all tabs to use
    auth0_data_placeholder = st.empty()
    auth0_df = None

    # Tab 1: Time-Bound Auth0 Flow
    with tabs[0]:
        st.subheader("Time-Bound Auth0 Flow")
        st.markdown("""
        This visualization shows user flow based on time-bound Auth0 data.
        """)

        if st.button("Generate Time-Bound Sankey Diagrams"):
            with st.spinner("Generating time-bound Sankey diagrams..."):
                # Generate the time-bound analysis
                fixed_figures, fixed_summary = generate_time_bound_analysis(
                    num_weeks, exclude_blacklisted=exclude_blacklisted)

                if not fixed_figures:
                    st.error("No data available for time-bound analysis.")
                else:
                    for i, fig in enumerate(fixed_figures):
                        st.subheader(
                            f"Week {i+1}: {fixed_summary.iloc[i]['period']}")
                        st.plotly_chart(fig, use_container_width=True)

                    # Display summary metrics
                    st.subheader("Summary Metrics")

                    # Select columns to display
                    display_cols = [
                        'period', 'active_users', 'inactive_users', 'new_users',
                        'returning_users', 'single_query_users', 'few_query_users',
                        'many_query_users', 'registered_query_count', 'anonymous_query_count'
                    ]

                    st.dataframe(
                        fixed_summary[display_cols], use_container_width=True)

                    # Week-over-week changes
                    st.subheader("Week-over-Week Changes")

                    # Format percentage columns for display
                    display_df = fixed_summary.copy()
                    pct_cols = [
                        col for col in display_df.columns if col.endswith('_wow_change')]
                    for col in pct_cols:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")

                    change_cols = [
                        'period', 'active_users_wow_change', 'new_users_wow_change',
                        'returning_users_wow_change', 'registered_query_count_wow_change',
                        'anonymous_query_count_wow_change'
                    ]

                    st.dataframe(display_df[change_cols],
                                 use_container_width=True)

    # Tab 2: Original PostHog Flow
    with tabs[1]:
        st.subheader("Original PostHog Flow")
        st.markdown("""
        This visualization shows the original user flow based on PostHog data.
        """)

        if st.button("Generate Original PostHog Flow"):
            # Load Auth0 data if it's not already loaded
            with st.spinner("Loading Auth0 user data (cached)..."):
                auth0_df = get_cached_auth0_data()

                if auth0_df.empty:
                    st.error(
                        "Failed to retrieve Auth0 user data. Please try again later.")
                else:
                    st.success(
                        f"Successfully loaded {len(auth0_df)} Auth0 users.")

            with st.spinner("Generating original PostHog flow Sankey diagrams..."):
                # Use the method with cached Auth0 data
                original_figures, original_summary = generate_week_over_week_analysis_with_auth0(
                    num_weeks, auth0_df, exclude_blacklisted=exclude_blacklisted)

                if not original_figures:
                    st.error("No data available for original analysis.")
                else:
                    combined_fig = create_combined_visualization(
                        original_figures, original_summary)
                    st.plotly_chart(combined_fig, use_container_width=True)

                    # Display individual weekly diagrams
                    for i, fig in enumerate(original_figures):
                        st.subheader(
                            f"Week {i+1}: {original_summary.iloc[i]['period']}")
                        st.plotly_chart(fig, use_container_width=True)

                    # Display summary metrics
                    st.subheader("Summary Metrics")

                    # Select columns to display
                    display_cols = [
                        'period', 'logged_in_count', 'not_logged_in_count',
                        'new_users_count', 'returning_users_count',
                        'logged_in_total', 'not_logged_in_total'
                    ]

                    st.dataframe(
                        original_summary[display_cols], use_container_width=True)

                    # Week-over-week changes
                    st.subheader("Week-over-Week Changes")

                    # Format percentage columns for display
                    display_df = original_summary.copy()
                    pct_cols = [
                        col for col in display_df.columns if col.endswith('_wow_change')]
                    for col in pct_cols:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")

                    change_cols = [
                        'period', 'logged_in_count_wow_change', 'not_logged_in_count_wow_change',
                        'new_users_count_wow_change', 'returning_users_count_wow_change',
                        'logged_in_total_wow_change', 'not_logged_in_total_wow_change'
                    ]

                    st.dataframe(display_df[change_cols],
                                 use_container_width=True)

    # Tab 3: Paid vs Non-Paid Flow
    with tabs[2]:
        st.subheader("Paid vs Non-Paid Flow")
        st.markdown("""
        This visualization breaks down user activity by paid status.
        """)

        if len(paid_emails) == 0:
            st.warning(
                "Please enter at least one paid user email to generate this analysis.")

        if st.button("Generate Paid vs Non-Paid Flow"):
            # Check if we have paid email addresses
            if len(paid_emails) == 0:
                st.error(
                    "Please enter at least one paid user email to generate this analysis.")
            else:
                # Load Auth0 data if it's not already loaded
                with st.spinner("Loading Auth0 user data (cached)..."):
                    auth0_df = get_cached_auth0_data()

                    if auth0_df.empty:
                        st.error(
                            "Failed to retrieve Auth0 user data. Please try again later.")
                    else:
                        st.success(
                            f"Successfully loaded {len(auth0_df)} Auth0 users.")

                        # Display info about paid users
                        st.info(f"Found {len(paid_emails)} paid users")

                        paid_figures = []
                        paid_metrics = []
                        now = datetime.now(timezone.utc)

                        for i in range(num_weeks):
                            end_date = now - timedelta(days=7*i)
                            start_date = end_date - timedelta(days=7)

                            try:
                                fig, metrics = create_paid_nonpaid_time_bound_sankey(
                                    start_date, end_date, auth0_df, paid_emails, exclude_blacklisted=exclude_blacklisted)
                                if fig and metrics:
                                    paid_figures.append(fig)
                                    paid_metrics.append(metrics)
                            except Exception as e:
                                st.error(f"Error processing week {i+1}: {e}")

                        # Display paid vs non-paid Sankey diagrams
                        if paid_figures:
                            # Create summary DataFrame
                            paid_summary_df = pd.DataFrame(paid_metrics)

                            for i, fig in enumerate(paid_figures):
                                st.subheader(
                                    f"Week {i+1}: {paid_summary_df.iloc[i]['period']}")
                                st.plotly_chart(fig, use_container_width=True)

                            # Calculate week-over-week changes
                            for col in [
                                'paid_users', 'non_paid_users', 'active_paid_users',
                                'inactive_paid_users', 'active_non_paid_users',
                                'paid_query_count', 'non_paid_query_count', 'anonymous_query_count'
                            ]:
                                if col in paid_summary_df.columns:
                                    paid_summary_df[f'{col}_wow_change'] = paid_summary_df[col].pct_change(
                                        -1) * 100

                            # Display summary metrics
                            st.subheader("Summary Metrics")

                            # Select columns to display
                            display_cols = [
                                'period', 'total_users', 'paid_users', 'non_paid_users',
                                'active_paid_users', 'inactive_paid_users',
                                'active_non_paid_users', 'inactive_non_paid_users',
                                'paid_query_count', 'non_paid_query_count',
                                'anonymous_query_count', 'total_query_count'
                            ]

                            # Filter to only include columns that exist in the dataframe
                            display_cols = [
                                col for col in display_cols if col in paid_summary_df.columns]

                            st.dataframe(
                                paid_summary_df[display_cols], use_container_width=True)

                            # Week-over-week changes
                            st.subheader("Week-over-Week Changes")

                            # Format percentage columns for display
                            display_df = paid_summary_df.copy()
                            pct_cols = [
                                col for col in display_df.columns if col.endswith('_wow_change')]
                            for col in pct_cols:
                                display_df[col] = display_df[col].apply(
                                    lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")

                            change_cols = [
                                'period', 'paid_users_wow_change', 'non_paid_users_wow_change',
                                'active_paid_users_wow_change', 'paid_query_count_wow_change',
                                'non_paid_query_count_wow_change'
                            ]

                            # Filter to only include columns that exist in the dataframe
                            change_cols = [
                                col for col in change_cols if col in display_df.columns]

                            st.dataframe(
                                display_df[change_cols], use_container_width=True)
                        else:
                            st.error(
                                "No data available for paid vs non-paid visualization.")
