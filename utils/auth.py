import streamlit as st
import requests


def authenticate(url, payload, headers):
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()["result"]
        if "token" not in data:
            raise Exception("Authentication failed: No token received.")
        st.session_state.authenticated = True
        st.session_state.token = data.get("token")
        st.session_state.username = payload["username"]
        st.success("Login successful!")
        st.rerun()
    else:
        raise Exception(
            f"Request failed with status code {response.status_code}: {response.text}"
        )


def login_user():
    if st.session_state.get("authenticated"):
        st.success("You are already logged in.")
        return  # Avoid showing login form again

    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        base_url = "https://klms.stelar.gr/stelar"
        endpoint = "/api/v1/users/token"
        url = f"{base_url}{endpoint}"

        payload = {"username": username, "password": password}
        headers = {"Content-Type": "application/json"}

        try:
            authenticate(url, payload, headers)
        except Exception as e:
            st.error(f"Login failed: {e}")
