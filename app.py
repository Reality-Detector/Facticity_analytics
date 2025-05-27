"""
Main Streamlit application file for Facticity dashboard.
"""
import streamlit as st
import os
import re

# Page configuration
st.set_page_config(
    page_title="Facticity Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_dotenv(dotenv_path='.env'):
    """
    Load environment variables from a .env file into the environment.
    Simple replacement for python-dotenv.
    
    Args:
        dotenv_path (str): Path to the .env file, defaults to '.env'
    """
    if not os.path.exists(dotenv_path):
        print(f"Warning: {dotenv_path} file not found")
        return

    with open(dotenv_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse key-value pairs
            match = re.match(r'^([A-Za-z_0-9]+)=(.*)$', line)
            if match:
                key, value = match.groups()
                # Remove quotes if present
                value = value.strip('\'"')
                # Set environment variable
                os.environ[key] = value


from views import (
    show_metrics_view,
    show_user_activity_view,
    show_interactive_engagement_view,
    show_email_segments_view,
    show_fixed_sankey_view,
    show_blacklist_view,
    show_query_explorer_view,
    show_overview_view,
)

# Load environment variables
load_dotenv()


st.sidebar.title("Facticity Analytics")

# View selection
view_options = {
    "Overview Dashboard": show_overview_view,
    "Metrics Dashboard": show_metrics_view,
    "User Activity": show_user_activity_view,
    "Usage Distribution": show_interactive_engagement_view,
    "Email Segments": show_email_segments_view,
    "User Flow": show_fixed_sankey_view,
    "Query Explorer": show_query_explorer_view,
    "Email Blacklist": show_blacklist_view,
}
selected_view = st.sidebar.selectbox(
    "Select View",
    list(view_options.keys()),
    index=1, 
)

# Display the selected view
view_options[selected_view]()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Fax")