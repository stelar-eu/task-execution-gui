import requests
import pandas as pd


def fetch_minio_credentials(
    token: str, base_url: str = "https://klms.stelar.gr/stelar"
) -> dict:
    # Define the endpoint URL
    base_url = "https://klms.stelar.gr/stelar"
    endpoint = "/api/v1/users/s3/credentials"
    url = f"{base_url}{endpoint}"

    # Set headers with Bearer Token for authentication
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    # Send the GET request
    response = requests.get(url, headers=headers)

    # Raise an error for bad responses
    response.raise_for_status()

    return response.json()

    # Convert JSON to DataFrame, adjust if response is a list or nested
    #     if isinstance(data, list):
    #         df = pd.DataFrame(data['result'])
    #     else:
    #         df = pd.json_normalize(data['result'])
    #     return df
    # else:
    #     raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
