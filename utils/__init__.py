"""
Util functions
"""


def normalize_url_for_mongo(url):
    """
    Convert a URL to a safe MongoDB field name by replacing dots with underscores.
    This is used when storing URLs as field names in MongoDB documents.
    """
    if not url:
        return "writer"
    return url.replace(".", "_")

def denormalize_url_from_mongo(safe_key):
    """
    Convert a safe MongoDB field name back to the original URL by replacing underscores with dots.
    This is used when retrieving URLs from MongoDB documents.
    """
    if not safe_key or safe_key == "writer":
        return safe_key
    return safe_key.replace("_", ".")

