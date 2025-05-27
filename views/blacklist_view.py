"""
Blacklist using .env but changes on streamlit do not get pushed to git repo
 i.e. new commits remove user changes to blacklist. 
"""
import streamlit as st
import re
import os
from dotenv import load_dotenv


def show_blacklist_view():
    """Simple blacklist editor that uses .env file."""
    st.title("Email Blacklist")

    # Get current blacklisted emails
    from config import BLACKLIST_EMAILS
    current_emails = sorted(list(BLACKLIST_EMAILS))
    email_text = "\n".join(current_emails)

    # Text area for editing
    new_email_text = st.text_area(
        "Edit blacklisted emails (one per line)",
        value=email_text,
        height=300
    )

    # Save button
    if st.button("Save Changes"):
        # Parse emails
        new_emails = [line.strip()
                      for line in new_email_text.split("\n") if line.strip()]

        # Validate emails
        invalid_emails = []
        valid_emails = []
        for email in new_emails:
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                valid_emails.append(email)
            else:
                invalid_emails.append(email)

        if invalid_emails:
            st.error(f"Invalid emails: {', '.join(invalid_emails)}")
        else:
            try:
                # Path to .env file
                env_path = os.path.join(os.path.dirname(
                    os.path.dirname(__file__)), ".env")

                # Read current env file
                env_content = ""
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        env_content = f.read()

                # Remove existing BLACKLIST_EMAILS line if it exists
                env_lines = env_content.split("\n")
                filtered_lines = [
                    line for line in env_lines if not line.startswith("BLACKLIST_EMAILS=")]

                # Add the new BLACKLIST_EMAILS line
                email_str = ",".join(valid_emails)
                filtered_lines.append(f"BLACKLIST_EMAILS={email_str}")

                # Write back to .env
                with open(env_path, 'w') as f:
                    f.write("\n".join(filtered_lines))

                # Update in-memory set
                BLACKLIST_EMAILS.clear()
                for email in valid_emails:
                    BLACKLIST_EMAILS.add(email)

                st.success("Blacklist updated. Refresh for full effect.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
