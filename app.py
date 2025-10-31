import streamlit as st
from utils.auth import login_user
import importlib
import sys
from pages import Workflow, Datasets, S3, Tools

# Session State for Login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def display_header():
    """Display the STELAR logo as header"""
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image("Logo - Stelar project.png", width=200)


def create_navigation():
    """Create a custom navigation bar"""
    st.markdown(
        """
    <style>
    .nav-bar {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .nav-button {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0 0.5rem;
        cursor: pointer;
        display: inline-block;
        text-decoration: none;
        color: #333;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        background-color: #e8f4fd;
        border-color: #1f77b4;
    }
    .nav-button.active {
        background-color: #1f77b4;
        color: white;
        border-color: #1f77b4;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize current page in session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Workflow Processes"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "🔄 Workflow Processes", key="nav_workflow", use_container_width=True
        ):
            st.session_state.current_page = "Workflow Processes"

    with col2:
        if st.button("📊 Datasets", key="nav_datasets", use_container_width=True):
            st.session_state.current_page = "Datasets"

    with col3:
        if st.button("☁️ S3 Menu", key="nav_s3", use_container_width=True):
            st.session_state.current_page = "S3 Menu"

    with col4:
        if st.button("🛠️ Tool Selection", key="nav_tools", use_container_width=True):
            st.session_state.current_page = "Tool Selection"

    return st.session_state.current_page


def main():
    st.set_page_config(
        page_title="STELAR Task Executor", 
        initial_sidebar_state="collapsed", 
        layout="wide"
    )

    if not st.session_state.authenticated:
        login_user()
    else:
        # Display logo header
        display_header()
        
        # Create navigation bar
        current_page = create_navigation()

        # Display current page content
        if current_page == "Workflow Processes":
            if "pages.Workflow" in sys.modules:
                importlib.reload(sys.modules["pages.Workflow"])
            Workflow.run()

        elif current_page == "Datasets":
            if "pages.Datasets" in sys.modules:
                importlib.reload(sys.modules["pages.Datasets"])
            Datasets.run()

        elif current_page == "S3 Menu":
            if "pages.S3" in sys.modules:
                importlib.reload(sys.modules["pages.S3"])
            S3.run()

        elif current_page == "Tool Selection":
            if "pages.Tools" in sys.modules:
                importlib.reload(sys.modules["pages.Tools"])
            Tools.run()


if __name__ == "__main__":
    main()