import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
import pycountry
from datetime import datetime, timezone, timedelta
import json
import os
import pickle
from pathlib import Path
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import boto3

from config import AUTH0_IP_LOOKUP_FILEPATH, AWS_SECRET_KEY, AWS_ACCESS_KEY

# Import database functions
from database.mongo_client import get_db_connection, fetch_recent_queries

# Constants
DATA_DIR = "data/explorer"
os.makedirs(DATA_DIR, exist_ok=True)
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
COST_PER_TOKEN = 0.0004  # Update with current pricing
EMBEDDING_CACHE_FILE = f"{DATA_DIR}/iab_embeddings_cache.pkl"

AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY
AWS_SECRET_ACCESS_KEY = AWS_SECRET_KEY
AWS_REGION = 'us-west-2'

# Create Bedrock client
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Load IAB categories from JSON files


def load_iab_categories():
    try:
        # Load main categories
        with open(f"{DATA_DIR}/maincategories.json", "r") as f:
            main_categories = json.load(f)

        # Load subcategories
        with open(f"{DATA_DIR}/subcategories.json", "r") as f:
            subcategories = json.load(f)

        return main_categories, subcategories
    except Exception as e:
        st.error(f"Error loading IAB categories: {str(e)}")
        # Fallback to minimal categories if files can't be loaded
        return ["Technology & Computing", "Other"], {"Technology & Computing": ["Software", "Hardware"]}


# Global variables for categories
MAIN_CATEGORIES, SUBCATEGORIES = load_iab_categories()


def generate_iab_embeddings():
    """Generate embeddings for IAB categories using the working approach"""
    # Check if embeddings already exist
    try:
        with open(EMBEDDING_CACHE_FILE, 'rb') as f:
            st.success("Loading cached IAB embeddings")
            return pickle.load(f)
    except FileNotFoundError:
        pass

    st.info("Generating embeddings for IAB categories. This will be done only once.")

    # Prepare data for embedding
    iab_data = []

    # For each main category, use its subcategories as examples
    for main_cat in MAIN_CATEGORIES:
        if main_cat in SUBCATEGORIES:
            # Use subcategories as examples of the main category
            subcats = SUBCATEGORIES[main_cat]
            for subcat in subcats:
                iab_data.append({
                    "text": subcat,
                    "main_category": main_cat
                })
        else:
            # If no subcategories, use the main category itself
            iab_data.append({
                "text": main_cat,
                "main_category": main_cat
            })

    # Convert to DataFrame
    iab_df = pd.DataFrame(iab_data)

    # Get texts and prepare for embedding
    texts = iab_df['text'].tolist()
    total_texts = len(texts)

    # Create progress display
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store embeddings
    embeddings = []

    # Generate embeddings in batches
    batch_size = 10
    total_batches = (total_texts + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_texts)
        batch_texts = texts[start_idx:end_idx]
        status_text.text(
            f"Processing batch {batch_idx+1}/{total_batches} ({start_idx+1}-{end_idx} of {total_texts})")

        batch_embeddings = []

        for text in batch_texts:
            try:
                # Call Bedrock
                payload = json.dumps({"inputText": text})
                response = bedrock_client.invoke_model(
                    modelId="amazon.titan-embed-text-v2:0",
                    body=payload,
                    contentType="application/json",
                    accept="application/json"
                )
                result = json.loads(response['body'].read())
                batch_embeddings.append(result["embedding"])
                time.sleep(0.1)  # Avoid rate limits
            except Exception as e:
                st.error(f"Error generating IAB embedding: {str(e)[:100]}")
                # Add placeholder embedding
                batch_embeddings.append([0.0] * 1536)

        # Add results
        embeddings.extend(batch_embeddings)

        # Update progress
        progress_bar.progress((batch_idx + 1) / total_batches)

    # Add to dataframe
    iab_df['embedding'] = embeddings

    # Group by main category and average the embeddings for each
    category_embeddings = {}
    for main_cat in MAIN_CATEGORIES:
        cat_df = iab_df[iab_df['main_category'] == main_cat]
        if not cat_df.empty:
            # Average the embeddings of all examples for this category
            avg_embedding = np.mean(cat_df['embedding'].tolist(), axis=0)
            category_embeddings[main_cat] = avg_embedding

    # Save embeddings to disk
    with open(EMBEDDING_CACHE_FILE, 'wb') as f:
        pickle.dump(category_embeddings, f)

    status_text.text("IAB embeddings generation complete!")
    return category_embeddings


def categorize_with_titan(query_embedding, category_embeddings):
    """Find closest IAB category using cosine similarity with Titan embeddings"""
    similarities = {}

    for category, cat_embedding in category_embeddings.items():
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, cat_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(cat_embedding)
        )
        similarities[category] = similarity

    # Return category with highest similarity
    return max(similarities, key=similarities.get)


def generate_query_embeddings(df):
    """Generate embeddings for queries using AWS Bedrock Titan"""
    # Make sure we have queries
    if df is None or df.empty or 'query' not in df.columns:
        st.error("No queries available for embedding generation")
        return None

    # Get queries and prepare for embedding
    queries = df['query'].tolist()
    total_queries = len(queries)

    # Create progress display
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store embeddings and token counts
    embeddings = []
    token_counts = []

    # Generate embeddings in batches
    batch_size = 10
    total_batches = (total_queries + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_queries)
        batch_queries = queries[start_idx:end_idx]
        status_text.text(
            f"Processing batch {batch_idx+1}/{total_batches} ({start_idx+1}-{end_idx} of {total_queries})")

        batch_embeddings = []
        batch_token_counts = []

        for query in batch_queries:
            try:
                # Call Bedrock
                payload = json.dumps({"inputText": query})
                response = bedrock_client.invoke_model(
                    modelId="amazon.titan-embed-text-v2:0",
                    body=payload,
                    contentType="application/json",
                    accept="application/json"
                )
                result = json.loads(response['body'].read())
                batch_embeddings.append(result["embedding"])
                batch_token_counts.append(result["inputTextTokenCount"])
                time.sleep(0.1)  # Avoid rate limits
            except Exception as e:
                st.error(f"Error generating embedding: {str(e)[:100]}")
                # Add placeholder embedding
                batch_embeddings.append([0.0] * 1536)
                batch_token_counts.append(0)

        # Add results
        embeddings.extend(batch_embeddings)
        token_counts.extend(batch_token_counts)

        # Update progress
        progress_bar.progress((batch_idx + 1) / total_batches)

    # Add to dataframe
    df_with_embeddings = df.copy()
    df_with_embeddings['embedding'] = embeddings
    df_with_embeddings['token_count'] = token_counts

    # Calculate cost
    total_tokens = sum(token_counts)
    cost_per_1000_tokens = 0.0004
    estimated_cost = (total_tokens / 1000) * cost_per_1000_tokens
    status_text.text(f"Completed! Cost: ${estimated_cost:.4f}")

    return df_with_embeddings


def process_data_with_titan_embeddings(df):
    """Process data using Titan embeddings for IAB categorization"""
    # Load or generate IAB category embeddings
    if 'iab_embeddings' not in st.session_state:
        st.session_state.iab_embeddings = generate_iab_embeddings()

    category_embeddings = st.session_state.iab_embeddings

    # Generate embeddings for queries
    df_with_embeddings = generate_query_embeddings(df)

    if df_with_embeddings is None:
        return None

    # Categorize queries based on embeddings
    df_with_embeddings['category'] = df_with_embeddings['embedding'].apply(
        lambda x: categorize_with_titan(x, category_embeddings)
    )

    # Compress embeddings before saving to save space
    df_compressed = df_with_embeddings.copy()
    df_compressed['embedding'] = df_compressed['embedding'].apply(
        lambda x: np.array(x, dtype=np.float16).tobytes()
    )

    # Add to session state and save to disk
    processed_data = {
        'df': df_compressed,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'embedding_model': EMBEDDING_MODEL,
        'processed': True
    }

    # Save processed data
    save_data_to_disk(processed_data)

    # Return uncompressed data for immediate use
    return {
        'df': df_with_embeddings,
        'vectorizer': None,  # Not using TF-IDF vectorizer anymore
        'query_vectors': None  # Using embeddings directly
    }


def save_data_to_disk(data):
    """Save data with embeddings to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{DATA_DIR}/query_data_{timestamp}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    # Maintain only last 5 files
    pickle_files = list(Path(DATA_DIR).glob("query_data_*.pkl"))
    if len(pickle_files) > 5:
        old_files = sorted(pickle_files, key=os.path.getmtime)[:-5]
        for file in old_files:
            os.remove(file)

    return filename


def load_data_from_disk(filename=None):
    """Load data with embeddings from disk"""
    if not filename:
        pickle_files = list(Path(DATA_DIR).glob("query_data_*.pkl"))
        if not pickle_files:
            return None
        filename = max(pickle_files, key=os.path.getmtime)

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Decompress embeddings
    if 'df' in data and 'embedding' in data['df']:
        data['df']['embedding'] = data['df']['embedding'].apply(
            lambda x: np.frombuffer(
                x, dtype=np.float16).tolist() if isinstance(x, bytes) else x
        )

    return data


def show_query_explorer_view():
    """Main entry point for the enhanced query explorer dashboard."""
    st.title("Query Explorer Dashboard")

    # Sidebar for controls
    with st.sidebar:
        st.header("Data Controls")
        days_to_fetch = st.slider(
            "Days of Data", min_value=1, max_value=90, value=14)
        exclude_blacklisted = st.checkbox("Exclude Internal Users", value=True)

        # Data loading options
        st.subheader("Data Operations")
        data_operation = st.radio(
            "Choose operation:",
            ["Load saved data", "Fetch new data", "Refresh data"]
        )

    # Main data operations
    if data_operation == "Load saved data":
        if st.sidebar.button("Load Data"):
            with st.spinner("Loading saved data..."):
                load_saved_data()

    elif data_operation == "Fetch new data":
        if st.sidebar.button("Fetch Data"):
            with st.spinner("Fetching new data..."):
                fetch_new_data(days_to_fetch, exclude_blacklisted)

    elif data_operation == "Refresh data":
        if st.sidebar.button("Refresh Data"):
            with st.spinner("Refreshing data..."):
                refresh_data(days_to_fetch, exclude_blacklisted)

    # Main tabs for analysis
    tabs = st.tabs(
        ["IAB Category Analysis", "Geographic Analysis", "Query Search"])

    # Tab 1: IAB Category Analysis
    with tabs[0]:
        display_topic_analysis()

    # Tab 2: Geographic Analysis
    with tabs[1]:
        display_geographic_analysis()

    # Tab 3: Query Search
    with tabs[2]:
        display_query_search()


def load_saved_data():
    """Load the most recent saved data"""
    try:
        # Find the most recent pickle file
        pickle_files = list(Path(DATA_DIR).glob("query_data_*.pkl"))
        if not pickle_files:
            st.error("No saved data files found")
            return

        latest_file = max(pickle_files, key=os.path.getmtime)

        # Load data from disk
        data = load_data_from_disk(latest_file)

        # Store in session state
        st.session_state['query_data'] = data

        # Success message with details
        st.success(f"Loaded {len(data['df'])} queries from {latest_file.name}")
        if 'timestamp' in data:
            st.info(f"Data last updated: {data['timestamp']}")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


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


def fetch_new_data(days=14, exclude_blacklisted=True):
    """Fetch new data from database and process it"""
    try:
        # Fetch recent queries using the cached function
        recent_queries = fetch_recent_queries(days, exclude_blacklisted)

        if recent_queries is None or len(recent_queries) == 0:
            st.error("No queries found for the selected period")
            return

        # Add country information
        queries_with_country = get_country_from_data(recent_queries)

        # Generate embeddings
        embedded_data = generate_query_embeddings(queries_with_country)
        if embedded_data is None:
            st.error("Failed to generate embeddings")
            return

        # Store in session state
        st.session_state['query_data'] = {
            'df': embedded_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'embedding_model': EMBEDDING_MODEL,
            'processed': False  # Mark as not yet processed for categorization
        }

        # Save to disk for persistence
        save_data_to_disk(st.session_state['query_data'])

        st.success(f"Fetched and embedded {len(embedded_data)} queries")

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")


def refresh_data(days=14, exclude_blacklisted=True):
    """Refresh data by fetching only new queries since last update"""
    try:
        # Check if we have existing data
        if 'query_data' not in st.session_state:
            fetch_new_data(days, exclude_blacklisted)
            return

        existing_data = st.session_state['query_data']

        # Find the most recent timestamp in existing data
        if 'df' in existing_data and 'timestamp' in existing_data['df'].columns:
            existing_df = existing_data['df']
            max_timestamp = pd.to_datetime(existing_df['timestamp']).max()

            # Fetch only queries after this timestamp
            collection = get_db_connection()

            query_filter = {
                "timestamp": {"$gt": max_timestamp.isoformat()},
                "query": {"$exists": True},
                "userEmail": {"$exists": True}
            }

            # Add blacklist filter if needed
            if exclude_blacklisted:
                from config import BLACKLIST_EMAILS, BLACKLIST_DOMAINS
                email_filter = {"userEmail": {"$nin": list(BLACKLIST_EMAILS)}}
                domain_filters = []
                for domain in BLACKLIST_DOMAINS:
                    domain_filters.append(
                        {"userEmail": {"$not": {"$regex": f"@{domain}$"}}}
                    )
                query_filter = {
                    "$and": [query_filter, email_filter, *domain_filters]}

            # Execute query
            new_data = list(collection.find(
                query_filter,
                {"query": 1, "userEmail": 1, "timestamp": 1, "_id": 0}
            ))

            if not new_data:
                st.info("No new queries found since last update")
                return

            # Convert to dataframe and add country info
            new_df = pd.DataFrame(new_data)
            new_df_with_country = get_country_from_data(new_df)

            # Generate embeddings for new data
            new_df_with_embeddings = generate_query_embeddings(
                new_df_with_country)
            if new_df_with_embeddings is None:
                st.error("Failed to generate embeddings for new data")
                return

            # Combine with existing data
            # First, ensure existing embeddings are decompressed
            if isinstance(existing_df['embedding'].iloc[0], bytes):
                existing_df['embedding'] = existing_df['embedding'].apply(
                    lambda x: np.frombuffer(x, dtype=np.float16).tolist()
                )

            combined_df = pd.concat([existing_df, new_df_with_embeddings])

            # Update session state
            st.session_state['query_data'] = {
                'df': combined_df,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'embedding_model': EMBEDDING_MODEL,
                'processed': False  # Reset processed flag since we have new data
            }

            # Save to disk
            save_data_to_disk(st.session_state['query_data'])

            st.success(
                f"Added {len(new_df)} new queries. Total: {len(combined_df)}")
        else:
            # If no timestamp data available, just fetch new data
            fetch_new_data(days, exclude_blacklisted)

    except Exception as e:
        st.error(f"Error refreshing data: {str(e)}")


def get_country_from_data(df):
    """Extract country information from user data"""
    # Create a copy with normalized emails
    df_copy = df.copy()
    if 'userEmail' in df_copy.columns:
        df_copy['email'] = df_copy['userEmail'].str.lower().str.strip()

    # Extract domain and guess country from TLD
    df_copy['domain'] = df_copy['email'].apply(
        lambda x: x.split('@')[-1] if isinstance(x, str) and '@' in x else ''
    )

    # Common TLD to country mapping
    tld_to_country = {
        'us': 'US', 'uk': 'GB', 'ca': 'CA', 'au': 'AU', 'de': 'DE',
        'fr': 'FR', 'jp': 'JP', 'cn': 'CN', 'in': 'IN', 'sg': 'SG'
    }

    # Map domain TLD to country
    df_copy['country_code'] = df_copy['domain'].apply(
        lambda x: tld_to_country.get(x.split('.')[-1].lower())
        if isinstance(x, str) and '.' in x else None
    )

    # Try to load IP geolocation data if available
    try:
        ip_df = pd.read_csv(AUTH0_IP_LOOKUP_FILEPATH)
        # Clean column names
        ip_df.columns = [col.replace('*', '').strip() for col in ip_df.columns]

        # Clean values
        for col in ip_df.columns:
            if ip_df[col].dtype == 'object':
                ip_df[col] = ip_df[col].str.replace("'", "").str.strip()

        # Create email to country mapping
        ip_lookup = {}
        for _, row in ip_df.iterrows():
            if 'email' in row and 'asn_country_code' in row and pd.notna(row['asn_country_code']):
                email = row['email'].lower().strip()
                ip_lookup[email] = row['asn_country_code']

        # Add IP-based country (prioritized over TLD)
        df_copy['ip_country'] = df_copy['email'].map(ip_lookup)

        # Use IP country first, then fall back to TLD country
        df_copy['country_code'] = df_copy.apply(
            lambda row: row['ip_country'] if pd.notna(
                row['ip_country']) else row['country_code'],
            axis=1
        )
    except FileNotFoundError:
        # IP data not available, continue with TLD-based countries
        pass

    # Convert country codes to full names
    def get_country_name(code):
        if pd.isna(code):
            return 'Unknown'

        try:
            # Handle special cases
            if code == 'UK':
                code = 'GB'

            # Look up country name
            country = pycountry.countries.get(alpha_2=code)
            if country:
                return country.name
            return code
        except:
            return code

    df_copy['country'] = df_copy['country_code'].apply(get_country_name)

    return df_copy


def display_topic_analysis():
    st.header("Topic Analysis")

    if 'query_data' not in st.session_state:
        st.info("Please load or fetch data first")
        return

    data = st.session_state['query_data']
    df = data['df']

    # Check if we need to process data with IAB categories
    if data.get('processed') is not True:
        with st.spinner("Categorizing queries with Titan embeddings... This may take a moment"):
            # Process data with IAB categories
            processed_result = process_data_with_titan_embeddings(df)

            if processed_result is None:
                st.error("Categorization failed")
                return

            # Update session state with processed data
            st.session_state['query_data']['processed'] = True
            st.session_state['query_data']['categorized'] = processed_result

            # Save updated data
            save_data_to_disk(st.session_state['query_data'])

    # Display results
    categorized_data = st.session_state['query_data']['categorized']
    df_categorized = categorized_data['df']

    # Display category distribution
    category_counts = df_categorized['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    # Add percentage
    total_queries = len(df_categorized)
    category_counts['percentage'] = (
        category_counts['count'] / total_queries * 100).round(1)

    # Create bar chart of category distribution
    fig = px.bar(
        category_counts.sort_values('count', ascending=False),
        x='category',
        y='count',
        text='percentage',
        color='category',
        title="Query Distribution by Topic",
        labels={'category': 'Category',
                'count': 'Number of Queries', 'percentage': '%'}
    )

    # Improve layout
    fig.update_traces(
        texttemplate='%{text}%',
        textposition='outside'
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        xaxis_title="IAB Category",
        yaxis_title="Number of Queries"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show sample queries for selected category
    st.subheader("Sample Queries by Category")

    # Category selector
    selected_category = st.selectbox(
        "Select a category to see sample queries",
        options=sorted(df_categorized['category'].unique())
    )

    if selected_category:
        # Filter data for selected category
        category_df = df_categorized[df_categorized['category']
                                     == selected_category]

        # Calculate confidence scores using normalized similarity if available
        if 'embedding' in category_df.columns and 'iab_embeddings' in st.session_state:
            category_embedding = st.session_state.iab_embeddings[selected_category]

            # Calculate similarity scores
            category_df['confidence'] = category_df['embedding'].apply(
                lambda x: np.dot(x, category_embedding) / (
                    np.linalg.norm(x) * np.linalg.norm(category_embedding)
                ) * 100  # Convert to percentage
            )

            # Sort by confidence score
            category_df = category_df.sort_values(
                'confidence', ascending=False)

            # Display samples with confidence scores
            st.write(
                f"Showing top {min(10, len(category_df))} queries by confidence score:")

            for i, (_, row) in enumerate(category_df.head(10).iterrows(), 1):
                st.write(
                    f"{i}. {row['query']} (Confidence: {row['confidence']:.1f}%)")
        else:
            # Just show random samples
            samples = category_df.sample(min(10, len(category_df)))
            for i, query in enumerate(samples['query'].values, 1):
                st.write(f"{i}. {query}")


def display_geographic_analysis():
    """Display geographic analysis with country-specific IAB categories"""
    st.header("Geographic Analysis")

    if 'query_data' not in st.session_state:
        st.info("Please load or fetch data first")
        return

    if 'categorized' not in st.session_state['query_data']:
        st.warning("Please run category analysis first")
        return

    df = st.session_state['query_data']['categorized']['df']

    if 'country' not in df.columns:
        st.error("Country data not available")
        return

    # Country distribution
    country_counts = df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']

    # Display top countries bar chart
    st.subheader("Top Countries by Query Volume")
    top_countries = country_counts.sort_values(
        'count', ascending=False).head(10)

    fig = px.bar(
        top_countries,
        x='country',
        y='count',
        title="Top 10 Countries by Query Count",
        color='country',
        labels={'country': 'Country', 'count': 'Number of Queries'}
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Number of Queries",
        xaxis_tickangle=-45,
        showlegend=False  # Hide legend to save space
    )

    st.plotly_chart(fig, use_container_width=True)

    # Country selector for detailed analysis
    st.subheader("Country-Specific Category Analysis")
    selected_country = st.selectbox(
        "Select a country to analyze",
        options=sorted(df['country'].unique())
    )

    if selected_country:
        # Filter data for selected country
        country_data = df[df['country'] == selected_country]

        # Show total queries for this country
        st.metric("Total Queries", len(country_data))

        # Get IAB category distribution for this country
        country_categories = country_data['category'].value_counts(
        ).reset_index()
        country_categories.columns = ['category', 'count']
        country_categories['percentage'] = (
            country_categories['count'] / len(country_data) * 100).round(1)

        # Sort by count descending
        country_categories = country_categories.sort_values(
            'count', ascending=False)

        # Create horizontal bar chart for categories in this country
        cat_fig = px.bar(
            country_categories.head(10),  # Show top 10 categories
            y='category',
            x='count',
            orientation='h',
            title=f"Top Query Categories in {selected_country}",
            text='percentage',
            color='count',
            color_continuous_scale='Blues',
            labels={'category': 'IAB Category',
                    'count': 'Number of Queries', 'percentage': 'Percentage (%)'}
        )

        # Add percentage labels
        cat_fig.update_traces(
            texttemplate='%{text}%',
            textposition='outside'
        )

        # Better layout
        cat_fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Number of Queries",
            yaxis_title=""
        )

        st.plotly_chart(cat_fig, use_container_width=True)

        # Show sample queries for top categories
        st.subheader("Sample Queries by Category")

        # Get top 3 categories for this country
        top_categories = country_categories.head(3)

        for _, row in top_categories.iterrows():
            category = row['category']
            count = row['count']
            percentage = row['percentage']

            with st.expander(f"{category} ({count} queries, {percentage}%)"):
                # Get sample queries for this category in this country
                sample_df = country_data[country_data['category'] == category]

                # Show top queries with counts if we have counts available
                if 'count' in sample_df.columns:
                    query_counts = sample_df['query'].value_counts().head(5)
                    for i, (query, qcount) in enumerate(query_counts.items(), 1):
                        st.write(f"{i}. {query} ({qcount} times)")
                else:
                    # Otherwise just show a sample
                    for i, query in enumerate(sample_df.sample(min(5, len(sample_df)))['query'].values, 1):
                        st.write(f"{i}. {query}")


def display_query_search():
    """Display semantic search functionality using embeddings"""
    st.header("Query Search")

    if 'query_data' not in st.session_state:
        st.info("Please load or fetch data first")
        return

    if 'categorized' not in st.session_state['query_data']:
        st.warning("Please run category analysis first")
        return

    df = st.session_state['query_data']['categorized']['df']

    # Search options
    search_method = st.radio(
        "Search Method",
        options=["Text Search", "Semantic Search"],
        horizontal=True
    )

    if search_method == "Text Search":
        # Simple text search
        search_term = st.text_input("Enter search term")

        if search_term:
            # Search in queries
            search_results = df[df['query'].str.contains(
                search_term, case=False, na=False
            )]

            if len(search_results) > 0:
                st.success(f"Found {len(search_results)} matching queries")

                # Show results
                st.dataframe(
                    search_results[['query', 'category', 'country']].head(50),
                    use_container_width=True
                )
            else:
                st.info(f"No queries found containing '{search_term}'")
    else:
        # Semantic similarity search using embeddings
        similarity_query = st.text_area(
            "Enter a query to find similar queries", height=100)

        if similarity_query:
            with st.spinner("Generating embedding for search query..."):
                # Generate embedding for the search query
                search_df = pd.DataFrame([{"query": similarity_query}])
                search_with_embedding = generate_query_embeddings(search_df)

                if search_with_embedding is None or len(search_with_embedding) == 0:
                    st.error("Failed to generate embedding for search query")
                    return

                search_embedding = search_with_embedding['embedding'].iloc[0]

                # Calculate category
                if 'iab_embeddings' in st.session_state:
                    category = categorize_with_titan(
                        search_embedding, st.session_state.iab_embeddings)
                    st.info(f"Your query is categorized as: {category}")

                # Calculate similarities with all queries
                df_with_similarity = df.copy()
                df_with_similarity['similarity'] = df['embedding'].apply(
                    lambda x: np.dot(search_embedding, x) / (
                        np.linalg.norm(search_embedding) * np.linalg.norm(x)
                    ) if isinstance(x, list) and len(x) > 0 else 0.0
                )

                # Sort by similarity
                results_df = df_with_similarity.sort_values(
                    'similarity', ascending=False)

                # Display top matches
                top_n = 20
                top_matches = results_df.head(top_n)

                if len(top_matches) > 0:
                    st.success(
                        f"Found {len(top_matches)} semantically similar queries")

                    # Format for display
                    display_df = top_matches[[
                        'query', 'category', 'similarity', 'country']]
                    display_df['similarity'] = (
                        display_df['similarity'] * 100).round(1)

                    st.dataframe(
                        display_df,
                        use_container_width=True
                    )

                    # Visualize similarities
                    if len(top_matches) >= 5:
                        chart_data = top_matches.head(10).copy()
                        chart_data['similarity_pct'] = chart_data['similarity'] * 100

                        sim_chart = px.bar(
                            chart_data,
                            x='similarity_pct',
                            y='query',
                            orientation='h',
                            color='category',
                            title="Top 10 Similar Queries",
                            labels={
                                'similarity_pct': 'Similarity (%)', 'query': 'Query', 'category': 'IAB Category'}
                        )

                        sim_chart.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            xaxis_title="Similarity (%)",
                            yaxis_title=""
                        )

                        st.plotly_chart(sim_chart, use_container_width=True)
                else:
                    st.info("No similar queries found")


if __name__ == "__main__":
    show_query_explorer_view()
