import streamlit as st
import pandas as pd
from utils.s3_utils import fetch_minio_credentials
from minio import Minio
from datetime import datetime
import os

if "s3_resources" not in st.session_state:
    st.session_state.s3_resources = []

if "selected_bucket" not in st.session_state:
    st.session_state.selected_bucket = None

if "current_path" not in st.session_state:
    st.session_state.current_path = ""

if "navigation_history" not in st.session_state:
    st.session_state.navigation_history = []

if "s3_path" not in st.session_state:
    st.session_state.s3_path = ""


@st.cache_data(show_spinner=False)
def minio_creds(token):
    data = fetch_minio_credentials(token)
    if not data or "result" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["result"])
    return df


def get_minio_client(creds):
    """Create and return MinIO client"""
    return Minio(
        "minio.stelar.gr",
        access_key=creds["creds"].iloc[0],
        secret_key=creds["creds"].iloc[2],
        session_token=creds["creds"].iloc[3],
        secure=True,
    )


def format_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"


def parse_path_components(path):
    """Parse path into components for navigation"""
    if not path:
        return []
    return [comp for comp in path.split("/") if comp]


def build_breadcrumb_path(components, index):
    """Build path up to a specific breadcrumb index"""
    return "/".join(components[: index + 1])


def list_objects_in_path(client, bucket_name, prefix=""):
    """List objects and folders in a specific path"""
    try:
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=False)

        folders = set()
        files = []

        for obj in objects:
            relative_path = obj.object_name
            if prefix:
                relative_path = relative_path[len(prefix) :]

            if relative_path.endswith("/"):
                # This is a folder
                folder_name = relative_path.rstrip("/")
                if folder_name:
                    folders.add(folder_name)
            else:
                # This is a file
                if "/" in relative_path:
                    # File is in a subfolder
                    folder_name = relative_path.split("/")[0]
                    folders.add(folder_name)
                else:
                    # File is in current directory
                    files.append(
                        {
                            "name": relative_path,
                            "type": "file",
                            "size": obj.size,
                            "size_formatted": format_size(obj.size),
                            "last_modified": obj.last_modified,
                            "full_path": obj.object_name,
                        }
                    )

        # Add folders to the list
        folder_list = []
        for folder in sorted(folders):
            folder_list.append(
                {
                    "name": folder,
                    "type": "folder",
                    "size": 0,
                    "size_formatted": "-",
                    "last_modified": None,
                    "full_path": f"{prefix}{folder}/",
                }
            )

        return folder_list + files, True  # Return success flag

    except Exception as e:
        st.error(f"Error listing objects: {str(e)}")
        return [], False  # Return failure flag


def display_breadcrumb_navigation():
    """Display breadcrumb navigation"""
    if not st.session_state.selected_bucket:
        return

    path_components = parse_path_components(st.session_state.current_path)

    # Create breadcrumb navigation
    breadcrumb_cols = st.columns(len(path_components) + 2)

    # Root bucket link
    with breadcrumb_cols[0]:
        if st.button("🏠 " + st.session_state.selected_bucket, key="root_nav"):
            st.session_state.current_path = ""
            st.rerun()

    # Path components
    for i, component in enumerate(path_components):
        with breadcrumb_cols[i + 1]:
            display_text = f"📁 {component}"
            if st.button(display_text, key=f"breadcrumb_{i}"):
                st.session_state.current_path = (
                    build_breadcrumb_path(path_components, i) + "/"
                )
                st.rerun()


def display_folder_contents(client, bucket_name, current_path):
    """Display contents of current folder"""
    objects, success = list_objects_in_path(client, bucket_name, current_path)
    
    if not success:
        return False  # Return False if there was an error
    
    if not objects:
        st.info("This folder is empty.")
        return True  # Return True for successful execution (empty folder is valid)

    # Create DataFrame for display
    display_data = []
    for obj in objects:
        icon = "📁" if obj["type"] == "folder" else "📄"
        display_data.append(
            {
                "": icon,
                "Name": obj["name"],
                "Type": obj["type"].title(),
                "Size": obj["size_formatted"],
                "Last Modified": (
                    obj["last_modified"].strftime("%Y-%m-%d %H:%M:%S")
                    if obj["last_modified"]
                    else "-"
                ),
            }
        )

    df = pd.DataFrame(display_data)

    # Display table with selection
    selected_indices = st.dataframe(
        df,
        use_container_width=False,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Handle selection
    if selected_indices and len(selected_indices["selection"]["rows"]) > 0:
        selected_idx = selected_indices["selection"]["rows"][0]
        selected_object = objects[selected_idx]

        if selected_object["type"] == "folder":
            # Navigate to folder
            new_path = current_path + selected_object["name"] + "/"
            st.session_state.current_path = new_path
            st.rerun()
        else:
            # Display file details
            display_file_details(selected_object)
    
    return True  # Return True for successful execution


def display_file_details(file_obj):
    """Display details for a selected file"""
    st.subheader("File Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Name:** {file_obj['name']}")
        st.write(f"**Size:** {file_obj['size_formatted']} ({file_obj['size']:,} bytes)")
        st.write(f"**Type:** File")

    with col2:
        if file_obj["last_modified"]:
            st.write(
                f"**Last Modified:** {file_obj['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
        st.write(f"**Full Path:** {file_obj['full_path']}")

    # File actions
    st.subheader("File Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download", key=f"download_{file_obj['name']}"):
            st.info("Download functionality would be implemented here")

    # with col2:
    #     if st.button("ℹ️ Properties", key=f"props_{file_obj['name']}"):
    #         st.info("Properties dialog would be implemented here")

    # with col3:
    #     if st.button("🔗 Get Link", key=f"link_{file_obj['name']}"):
    #         st.info("Link generation would be implemented here")


def display_navigation_controls():
    """Display navigation controls"""
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("⬅️ Back", disabled=not st.session_state.current_path):
            # Go back one level
            current_components = parse_path_components(st.session_state.current_path)
            if current_components:
                if len(current_components) > 1:
                    st.session_state.current_path = (
                        "/".join(current_components[:-1]) + "/"
                    )
                else:
                    st.session_state.current_path = ""
                st.rerun()

    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()


def run():
    st.title("S3 Resource Manager")

    # Authentication check
    token = st.session_state.get("token")
    if not token:
        st.error("Authentication token not found. Please log in.")
        return ""  # Return empty string for consistency

    # Get credentials and create client
    try:
        creds = minio_creds(token)
        if creds.empty:
            st.error("Unable to fetch MinIO credentials.")
            return ""  # Return empty string for consistency
    except Exception as e:
        st.error(f"Error fetching credentials: {str(e)}")
        return ""  # Return empty string for consistency

    try:
        client = get_minio_client(creds)
        bucket_list = list(client.list_buckets())
        st.session_state.bucket_list = bucket_list

    except Exception as e:
        st.error(f"Error connecting to MinIO: {str(e)}")
        return ""  # Return empty string for consistency

    # Bucket selection
    st.subheader("Available Buckets")

    bucket_data = [
        {"Name": bucket.name, "Creation Date": bucket.creation_date}
        for bucket in bucket_list
    ]
    bucket_df = pd.DataFrame(bucket_data)
    s3_path = ""

    if not bucket_df.empty:
        # Display buckets with selection
        selected_bucket_indices = st.dataframe(
            bucket_df,
            use_container_width=False,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Handle bucket selection
        if (
            selected_bucket_indices
            and len(selected_bucket_indices["selection"]["rows"]) > 0
        ):
            selected_idx = selected_bucket_indices["selection"]["rows"][0]
            selected_bucket = bucket_data[selected_idx]["Name"]

            if st.session_state.selected_bucket != selected_bucket:
                st.session_state.selected_bucket = selected_bucket
                st.session_state.current_path = ""
                st.rerun()
        else:
            # No bucket selected - clear the selection
            if st.session_state.selected_bucket is not None:
                st.session_state.selected_bucket = None
                st.session_state.current_path = ""
                st.session_state.s3_path = ""
                st.rerun()

    # Folder navigation section
    if st.session_state.selected_bucket:
        st.divider()
        st.subheader(f"Browsing: {st.session_state.selected_bucket}")
        cols1, cols2, cols3 = st.columns([1, 1, 2])

        # Navigation controls
        display_navigation_controls()

        # Display current folder contents and check for success
        try:
            folder_display_success = display_folder_contents(
                client, st.session_state.selected_bucket, st.session_state.current_path
            )
        except Exception as e:
            st.error(f"Error displaying folder contents: {str(e)}")
            folder_display_success = False

        # Only initialize s3_path if display_folder_contents was successful
        if folder_display_success:
            s3_path = (
                st.session_state.selected_bucket + "/" + st.session_state.current_path
                if st.session_state.current_path
                else st.session_state.selected_bucket + "/"
            )
            st.session_state.s3_path = s3_path
            
            with cols1:
                st.info(f"S3 Path -> {s3_path}", icon="🔗")
        else:
            # Clear s3_path if there was an error
            st.session_state.s3_path = ""
            s3_path = ""
            with cols1:
                st.warning("You do not have enough permissions to view contents of this folder. Please, select another folder.", icon="⚠️")

    else:
        st.info("Select a bucket above to browse its contents.")
        # Clear s3_path when no bucket is selected
        s3_path = ""
        st.session_state.s3_path = ""
    
    return s3_path


if __name__ == "__main__":
    run()