import requests


def fetch_organization(
    token: str, base_url: str = "https://klms.stelar.gr/stelar"
) -> dict:
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
    endpoint = "/api/v2/organizations.fetch"
    url = f"{base_url}{endpoint}"

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    response = requests.get(url, headers=headers)

    # Raise an error for bad responses
    response.raise_for_status()

    data = response.json()
    if "result" in data and isinstance(data["result"], list):
        return [org["name"] for org in data["result"] if "name" in org]
    return []
