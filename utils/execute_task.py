import requests
import streamlit as st


def execute_task(token, config):
    """
    Execute a task by sending POST request to the STELAR API
    
    Parameters:
        token (str): Bearer token for authentication
        config (dict): Task configuration JSON (task_spec)
        
    Returns:
        dict: Response from the API
    """
    base_url = "https://klms.stelar.gr/stelar"
    endpoint = "/api/v2/task"
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        # POST the task_spec as JSON body
        response = requests.post(url, json=config, headers=headers, timeout=60)
        response.raise_for_status()
        
        return {
            "success": True,
            "data": response.json(),
            "status_code": response.status_code
        }
        
    except requests.exceptions.HTTPError as e:
        error_detail = None
        error_text = None
        
        try:
            if e.response and e.response.content:
                # Try to parse JSON error response
                error_detail = e.response.json()
        except:
            # If JSON parsing fails, get raw text
            if e.response:
                error_text = e.response.text
            
        return {
            "success": False,
            "error": str(e),
            "status_code": e.response.status_code if e.response else None,
            "response": error_detail,
            "response_text": error_text
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out after 60 seconds",
            "status_code": None,
            "response": None
        }
        
    except requests.exceptions.ConnectionError as e:
        return {
            "success": False,
            "error": f"Connection error: {str(e)}",
            "status_code": None,
            "response": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "status_code": None,
            "response": None
        }