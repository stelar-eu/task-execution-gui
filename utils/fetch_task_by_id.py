import requests
import json


def fetch_task_by_id(
    task_id: str, token: str, base_url: str = "https://klms.stelar.gr/stelar"
) -> dict:
    """
    Fetches task data from the Stelar API by task ID.

    Parameters:
        task_id (str): The unique identifier for the task.
        token (str): Bearer token for authentication.
        base_url (str): Base URL of the API. Defaults to the Stelar API base URL.

    Returns:
        dict: JSON response from the API if the request is successful.

    Raises:
        requests.HTTPError: If the request fails.
    """
    endpoint = f"/api/v2/task/{task_id}"
    url = f"{base_url}{endpoint}"

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    response = requests.get(url, headers=headers)

    # Raise an error for bad responses
    response.raise_for_status()

    return response.json()