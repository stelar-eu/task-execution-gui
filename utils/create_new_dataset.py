import requests
import streamlit as st


def create_new_ds(token: str, name: str, owner_org: str, base_url: str = "https://klms.stelar.gr/stelar") -> dict:
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
    endpoint = "/api/v2/dataset"
    url = f"{base_url}{endpoint}"

    payload = {
            "name": f"{name}",
            "owner_org": f"{owner_org}",
            }

            
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    response = requests.post    (url, json=payload ,headers=headers)

    # Raise an error for bad responses
    response.raise_for_status()

    if response.status_code != 200:
        st.error(f"Error creating dataset: {response.status_code} - {response.text}")
    else:
        st.success("Dataset created successfully!")
    
    return (response.json()['result']['id'], response.json()['result']['name'])

