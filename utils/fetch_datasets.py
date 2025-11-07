"""
STELAR KLMS API Client
This script provides functions to interact with the STELAR Knowledge Lake Management System API
for fetching and searching datasets.
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import time
import streamlit as st


class StelarKLMSClient:
    """Client for interacting with STELAR KLMS API"""
    
    def __init__(self, token: str, base_url: str = "https://klms.stelar.gr/stelar", 
                 timeout: int = 120, max_retries: int = 3):
        """
        Initialize the STELAR KLMS client
        
        Args:
            token: Bearer token for authentication
            base_url: Base URL for the KLMS API
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.token = token
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        print(f"[INFO] STELAR KLMS Client initialized")
        print(f"[INFO] Base URL: {base_url}")
        print(f"[INFO] Timeout: {timeout} seconds")
        print(f"[INFO] Max retries: {max_retries}")
        print(f"[INFO] Token: {'*' * 20}{token[-4:] if len(token) > 4 else '****'}")
        print("-" * 80)
    
    def fetch_datasets(
        self,
        keywords: Optional[List[str]] = None,
        spatial: Optional[Dict[str, Any]] = None,
        temporal: Optional[Dict[str, str]] = None,
        formats: Optional[List[str]] = None,
        batch_size: int = 1500,
        max_records: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch datasets using search endpoint with pagination
        
        Args:
            keywords: Optional list of keywords to filter
            spatial: Optional spatial filter	
            temporal: Optional temporal filter
            formats: Optional format filter
            batch_size: Number of records per batch (default: 100)
            max_records: Maximum records to fetch (default: None = fetch all)
        
        Returns:
            DataFrame containing dataset information
        """
        print(f"[INFO] Using search endpoint with pagination")
        print(f"[INFO] Batch size: {batch_size}")
        if max_records:
            print(f"[INFO] Max records: {max_records}")
        
        all_results = []
        offset = 0
        
        while True:
            remaining = None
            if max_records:
                remaining = max_records - offset
                if remaining <= 0:
                    break
                current_batch = min(batch_size, remaining)
            else:
                current_batch = batch_size
            
            batch = self.search_datasets(
                keywords=keywords,
                spatial=spatial,
                temporal=temporal,
                formats=formats,
                limit=current_batch,
                offset=offset
            )
            
            if batch.empty:
                break
            
            all_results.append(batch)
            offset += len(batch)
            st.info(f"[PROGRESS] Fetched {offset} datasets so far...")
            
            if len(batch) < current_batch:
                break
        
        if not all_results:
            print(f"[INFO] No results found")
            return pd.DataFrame()
        
        df = pd.concat(all_results, ignore_index=True)
        print(f"[SUCCESS] Total datasets fetched: {len(df)}")
        return df
    
    def search_datasets(
        self,
        keywords: Optional[List[str]] = None,
        spatial: Optional[Dict[str, Any]] = None,
        temporal: Optional[Dict[str, str]] = None,
        formats: Optional[List[str]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> pd.DataFrame:
        """
        Search for datasets using various criteria
        
        Args:
            keywords: List of keywords to search for (e.g., ['agriculture', 'crop'])
            spatial: Spatial filter with bounding box or geometry
                     Example: {'bbox': [lon_min, lat_min, lon_max, lat_max]}
            temporal: Temporal filter with start and end dates
                      Example: {'start': '2023-01-01', 'end': '2023-12-31'}
            formats: List of desired formats (e.g., ['csv', 'geojson'])
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
        
        Returns:
            DataFrame containing matching datasets
        """
        endpoint = "/api/v2/search/datasets"
        url = f"{self.base_url}{endpoint}"
        
        # Build search payload
        payload = {
            "limit": limit,
            "offset": offset
        }
        
        if keywords:
            payload["keywords"] = keywords
        if spatial:
            payload["spatial"] = spatial
        if temporal:
            payload["temporal"] = temporal
        if formats:
            payload["formats"] = formats
        
        print(f"\n[REQUEST] Searching datasets...")
        print(f"[URL] POST {url}")
        print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[TIMEOUT] {self.timeout} seconds")
        print(f"[PAYLOAD]")
        print(json.dumps(payload, indent=2))
        
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    wait_time = 2 ** (attempt - 1)  # Exponential backoff
                    print(f"[RETRY] Attempt {attempt}/{self.max_retries} (waiting {wait_time}s before retry)")
                    time.sleep(wait_time)
                else:
                    print(f"[INFO] Attempt {attempt}/{self.max_retries}")
                
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                print(f"[RESPONSE] Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"[SUCCESS] Search completed successfully")
                    
                    # Handle different response structures
                    if isinstance(data, dict):
                        if 'results' in data:
                            records = data['results']
                            total = data.get('total', len(records))
                            print(f"[INFO] Found {total} total matches")
                            print(f"[INFO] Returning {len(records)} results")
                        elif 'result' in data:
                            records = data['result']
                            print(f"[INFO] Found {len(records)} results")
                        else:
                            records = data
                            print(f"[INFO] Using root level data")
                    elif isinstance(data, list):
                        records = data
                        print(f"[INFO] Response is a list with {len(records)} items")
                    else:
                        raise ValueError(f"Unexpected response format: {type(data)}")
                    
                    # Convert to DataFrame
                    if isinstance(records, list) and len(records) > 0:
                        df = pd.DataFrame(records)
                        print(f"[INFO] Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
                        if len(df.columns) > 0:
                            print(f"[INFO] Columns: {', '.join(df.columns.tolist()[:10])}")
                            if len(df.columns) > 10:
                                print(f"[INFO] ... and {len(df.columns) - 10} more columns")
                    elif isinstance(records, list) and len(records) == 0:
                        print(f"[INFO] No results found matching the search criteria")
                        df = pd.DataFrame()
                    else:
                        df = pd.json_normalize(records)
                        print(f"[INFO] Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
                    
                    print("-" * 80)
                    return df
                
                elif response.status_code == 400:
                    raise Exception(
                        f"[ERROR] Bad request. Invalid search parameters.\n"
                        f"Response: {response.text}"
                    )
                elif response.status_code == 401:
                    raise Exception(f"[ERROR] Authentication failed. Please check your token.")
                elif response.status_code == 403:
                    raise Exception(f"[ERROR] Access forbidden. Insufficient permissions.")
                elif response.status_code >= 500:
                    # Server errors - retry
                    print(f"[WARNING] Server error {response.status_code}. Will retry if attempts remain.")
                    if attempt == self.max_retries:
                        raise Exception(
                            f"[ERROR] Server error after {self.max_retries} attempts. "
                            f"Status code: {response.status_code}\nResponse: {response.text}"
                        )
                    continue
                else:
                    raise Exception(
                        f"[ERROR] Search failed with status code {response.status_code}\n"
                        f"Response: {response.text}"
                    )
                    
            except requests.exceptions.Timeout:
                print(f"[WARNING] Search request timed out after {self.timeout} seconds")
                if attempt == self.max_retries:
                    raise Exception(
                        f"[ERROR] Search timed out after {self.max_retries} attempts. "
                        f"Try reducing the result limit or increasing the timeout parameter."
                    )
                continue
            except requests.exceptions.ConnectionError as e:
                print(f"[WARNING] Connection error: {str(e)}")
                if attempt == self.max_retries:
                    raise Exception(
                        f"[ERROR] Connection failed after {self.max_retries} attempts. "
                        f"Please check your network and the API URL"
                    )
                continue
            except requests.exceptions.RequestException as e:
                raise Exception(f"[ERROR] Search request failed: {str(e)}")
            except json.JSONDecodeError:
                raise Exception(f"[ERROR] Invalid JSON response: {response.text[:200]}")
        
        # This should never be reached
        raise Exception(f"[ERROR] Search failed after {self.max_retries} attempts")