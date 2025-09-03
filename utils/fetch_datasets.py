
import streamlit as st
import requests
import time
from typing import Dict, List, Optional


def fetch_datasets(token: str, limit: int = 200, offset: int = 0, base_url: str = "https://klms.stelar.gr/stelar") -> dict:
    """
    Fetches process data from the Stelar API.

    Parameters:
        token (str): Bearer token for authentication.
        base_url (str): Base URL of the API. Defaults to the Stelar API base URL.

    Returns:
        dict: JSON response from the API if the request is successful.

    Raises:
        requests.HTTPError: If the request fails.
    """
    endpoint = f"/api/v2/datasets.fetch?limit={limit}&offset={offset}"
    url = f"{base_url}{endpoint}"

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    response = requests.get(url, headers=headers)

    # Raise an error for bad responses
    response.raise_for_status()

    return response.json()


def fetch_datasets_paginated(
    token: str, 
    base_url: str = "https://klms.stelar.gr/stelar", 
    limit: int = 200,
    max_retries: int = 3,
    retry_delay: int = 1,
    show_progress: bool = False
) -> Dict:
    """
    Fetches all dataset data from the Stelar API using pagination.

    Parameters:
        token (str): Bearer token for authentication.
        base_url (str): Base URL of the API. Defaults to the Stelar API base URL.
        limit (int): Number of items to fetch per page. Defaults to 200 (API max).
        max_retries (int): Maximum number of retry attempts for failed requests.
        retry_delay (int): Delay in seconds between retry attempts.
        show_progress (bool): Whether to show Streamlit progress indicators.

    Returns:
        dict: Combined JSON response with all datasets.
    """
    
    all_datasets = []
    page = 1
    total_pages = None
    total_count = None
    
    headers = {
        "Authorization": f"Bearer {token}", 
        "Accept": "application/json"
    }
    
    # Initialize Streamlit progress indicators if requested
    progress_bar = None
    status_text = None
    if show_progress:
        progress_bar = st.progress(0, text="Initializing dataset fetch...")
        status_text = st.empty()
    
    try:
        while True:
            # Construct URL with pagination parameters
            endpoint = f"/api/v2/datasets.fetch?limit={limit}&page={page}"
            url = f"{base_url}{endpoint}"
            
            if show_progress:
                if total_pages:
                    progress = (page - 1) / total_pages
                    progress_bar.progress(progress, text=f"Fetching page {page}/{total_pages}...")
                else:
                    status_text.text(f"Fetching page {page}...")
            
            # Retry logic for each request
            last_exception = None
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    break
                    
                except requests.RequestException as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        if show_progress:
                            status_text.text(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying...")
                        time.sleep(retry_delay)
                    else:
                        raise e
            
            # Parse response
            try:
                data = response.json()
            except ValueError as e:
                raise requests.RequestException(f"Invalid JSON response on page {page}: {str(e)}")
            
            # Extract datasets from current page
            current_datasets = data.get("result", [])
            all_datasets.extend(current_datasets)
            
            # Get pagination info from first response
            if page == 1:
                # Try to extract total count from various possible response structures
                if "total" in data:
                    total_count = data["total"]
                elif "count" in data:
                    total_count = data["count"]
                elif "metadata" in data and "total" in data["metadata"]:
                    total_count = data["metadata"]["total"]
                
                # Calculate total pages if we have total count
                if total_count is not None:
                    total_pages = (total_count + limit - 1) // limit  # Ceiling division
            
            # Check if we should continue pagination
            should_continue = False
            
            if len(current_datasets) == limit:
                # Current page is full, likely more pages exist
                should_continue = True
            elif len(current_datasets) > 0 and total_pages is not None and page < total_pages:
                # We have pagination info and haven't reached the last page
                should_continue = True
            elif len(current_datasets) > 0 and total_pages is None:
                # No pagination info, but current page has data - try next page
                should_continue = True
            
            if not should_continue:
                break
            
            page += 1
            
            # Safety check to prevent infinite loops
            if page > 1000:  # Reasonable upper limit
                if show_progress:
                    st.warning("Reached maximum page limit (1000). Stopping pagination.")
                break
            
            # Small delay between requests to be respectful to the API
            time.sleep(0.1)
        
        # Update progress to completion
        if show_progress:
            progress_bar.progress(1.0, text=f"Complete! Fetched {len(all_datasets)} datasets from {page} pages.")
            time.sleep(1)  # Show completion briefly
    
    finally:
        # Clean up progress indicators
        if show_progress and progress_bar:
            progress_bar.empty()
        if show_progress and status_text:
            status_text.empty()
    
    return {
        "result": all_datasets,
        "metadata": {
            "total_count": len(all_datasets),
            "pages_fetched": page,
            "api_total_count": total_count
        }
    }