import streamlit as st
import pandas as pd
from utils.fetch_datasets import StelarKLMSClient


@st.cache_data(show_spinner=False, ttl=3600)
def get_all_datasets(token):
    client = StelarKLMSClient(token=token)
    
    # Fetch all datasets with pagination
    all_data = []
    offset = 0
    batch_size = 1000
    status_placeholder = st.empty()  # Add this
    
    while True:
        df = client.search_datasets(limit=batch_size, offset=offset)
        
        if df.empty:
            break
        
        # Extract results from the response
        data = df.iloc[0]['results']
        all_data.extend(data)
        
        status_placeholder.info(f"📊 Fetched {len(all_data)} datasets so far...")  # Add this
        
        # Check if we got fewer results than requested (end of data)
        if len(data) < batch_size:
            break
        
        offset += batch_size
    
    status_placeholder.success(f"✅ Total datasets retrieved: {len(all_data)}")  # Add this
    
    df = pd.DataFrame(all_data)
    df['organization_name'] = df['organization'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    return df

# @st.cache_data(show_spinner=False, ttl=3600)
# def get_all_datasets(token):
#     client = StelarKLMSClient(token=token)
    
#     # Fetch all datasets with pagination
#     all_data = []
#     offset = 0
#     batch_size = 1000
    
#     while True:
#         df = client.search_datasets(limit=batch_size, offset=offset)
        
#         if df.empty:
#             break
        
#         # Extract results from the response
#         data = df.iloc[0]['results']
#         all_data.extend(data)
        
#         print(f"[INFO] Collected {len(all_data)} datasets total")
        
#         # Check if we got fewer results than requested (end of data)
#         if len(data) < batch_size:
#             break
        
#         offset += batch_size
    
#     df = pd.DataFrame(all_data)
#     df['organization_name'] = df['organization'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
#     return df


def refresh_cached_data():
    """Clear the cache and force data refresh"""
    get_all_datasets.clear()
    st.rerun()


def reset_filters():
    """Reset all filters to default state"""
    if "filters" in st.session_state:
        st.session_state["filters"] = {}


def reset_columns(default_columns):
    """Reset column selection to default"""
    st.session_state.selected_columns = default_columns
    st.rerun()


def apply_filters(df, filter_cols):
    """Apply filters to the dataframe"""
    if df.empty or not filter_cols:
        return df
        
    filtered_df = df.copy()

    for col in filter_cols:
        if col not in df.columns:
            continue

        col_dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_dtype):
            min_val, max_val = df[col].min(), df[col].max()
            if min_val != max_val:  # Only show slider if there's a range
                selected_range = st.slider(
                    f"Filter by range in '{col}'",
                    float(min_val),
                    float(max_val),
                    (float(min_val), float(max_val)),
                    key=f"slider_{col}",
                )
                filtered_df = filtered_df[filtered_df[col].between(*selected_range)]

        elif pd.api.types.is_datetime64_any_dtype(col_dtype):
            min_date, max_date = df[col].min(), df[col].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = st.date_input(
                    f"Date range for '{col}'",
                    value=(min_date.date(), max_date.date()),
                    key=f"date_{col}",
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (filtered_df[col] >= pd.to_datetime(start_date))
                        & (filtered_df[col] <= pd.to_datetime(end_date))
                    ]

        elif df[col].nunique() <= 20:
            unique_values = df[col].dropna().unique().tolist()
            if unique_values:  # Only show multiselect if there are values
                options = st.multiselect(
                    f"Select values for '{col}'",
                    unique_values,
                    key=f"multi_{col}_datasets",
                )
                if options:
                    filtered_df = filtered_df[filtered_df[col].isin(options)]

        else:
            text_input = st.text_input(f"Search in '{col}'", key=f"text_{col}")
            if text_input:
                filtered_df = filtered_df[
                    filtered_df[col]
                    .astype(str)
                    .str.contains(text_input, case=False, na=False)
                ]

    return filtered_df


def prepare_display_dataframe(df, selected_columns):
    """
    Prepare DataFrame for display by handling JSON string columns
    
    Parameters:
        df: Original DataFrame
        selected_columns: Columns selected by user
    
    Returns:
        DataFrame ready for display
    """
    if df.empty or not selected_columns:
        return df
    
    display_df = df[selected_columns].copy()
    
    # For each column, check if it contains JSON strings and truncate them for display
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            # Check if this looks like JSON data
            sample = display_df[col].dropna().head(1)
            if not sample.empty:
                sample_val = sample.iloc[0]
                if isinstance(sample_val, str) and (sample_val.startswith('{') or sample_val.startswith('[')):
                    # Truncate long JSON strings for display
                    display_df[col] = display_df[col].apply(
                        lambda x: x[:100] + '...' if isinstance(x, str) and len(x) > 100 else x
                    )
    
    return display_df


def get_selected_dataset_info(original_df, display_df, selected_row_indices):
    """Get dataset IDs and names from selected rows"""
    if not selected_row_indices:
        return [], []

    dataset_ids = []
    dataset_names = []

    for row_index in selected_row_indices:
        if row_index < len(display_df):
            # Get the selected row from display dataframe
            selected_display_row = display_df.iloc[row_index]

            # Find the corresponding row in original dataframe
            if "id" in selected_display_row.index and "id" in original_df.columns:
                selected_id = selected_display_row["id"]
                # Find the matching row in original dataframe
                matching_rows = original_df[original_df["id"] == selected_id]
                if not matching_rows.empty:
                    dataset_ids.append(matching_rows.iloc[0]["id"])
                    # Get dataset name
                    name_column = None
                    for col in ["name", "dataset_name", "title"]:
                        if col in original_df.columns:
                            name_column = col
                            break
                    
                    if name_column:
                        dataset_names.append(matching_rows.iloc[0][name_column])
                    else:
                        dataset_names.append(f"Dataset_{selected_id}")
            else:
                # Fallback: use index matching
                if row_index < len(original_df):
                    dataset_ids.append(original_df.iloc[row_index]["id"])
                    name_column = None
                    for col in ["name", "dataset_name", "title"]:
                        if col in original_df.columns:
                            name_column = col
                            break
                    
                    if name_column:
                        dataset_names.append(original_df.iloc[row_index][name_column])
                    else:
                        dataset_names.append(f"Dataset_{original_df.iloc[row_index]['id']}")

    return dataset_ids, dataset_names


def run():
    st.title("Dataset Selection")
    st.markdown("Select and filter datasets from the complete dataset catalog.")
    
    token = st.session_state.get("token")
    if not token:
        st.error("Authentication token not found. Please log in.")
        return

    # Fetch all datasets (cached)
    with st.spinner("Loading all datasets..."):
        df = get_all_datasets(token)

    if df.empty:
        st.error("No datasets available. The server may be experiencing issues.")
        if st.button("🔄 Retry"):
            refresh_cached_data()
        return

    # Initialize session state for filters and columns
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = df.columns.tolist()

    # Advanced Filters
    with st.expander("Advanced Filters", expanded=False):
        st.markdown("Apply filters to narrow down the dataset catalog.")

        filter_cols = st.multiselect(
            "Select columns to filter",
            df.columns.tolist(),
            key="filter_columns_datasets",
        )

        # Apply filters
        filtered_df = apply_filters(df, filter_cols)

        # Reset filters button
        if st.button("Reset Filters", key="reset_filters_btn_datasets"):
            reset_filters()
            st.rerun()

    # Column Visibility
    with st.expander("Column Visibility", expanded=False):
        st.markdown("Select which columns to display.")

        # Define preferred default columns in order of preference
        preferred_defaults = ["id", "name", "title", "author", "organization_name"]
        
        # Only use defaults that actually exist in the DataFrame
        available_defaults = [col for col in preferred_defaults if col in df.columns.tolist()]
        
        # If no preferred columns exist, use the first 5 columns or all if less than 5
        if not available_defaults:
            available_defaults = df.columns.tolist()[:5]

        selected_columns = st.multiselect(
            "Columns to display",
            options=df.columns.tolist(),
            default=available_defaults,
            key="column_selector_datasets",
        )

        # Update session state
        st.session_state.selected_columns = selected_columns

        # Reset columns button
        if st.button("Reset Columns to Default", key="reset_columns_btn_datasets"):
            reset_columns(df.columns.tolist())

    # Apply column selection and prepare for display
    if selected_columns:
        display_df = prepare_display_dataframe(filtered_df, selected_columns)
    else:
        display_df = filtered_df
        st.warning("No columns selected. Showing all columns.")

    # Display summary stats with visual ratio
    filtered_count = len(display_df)
    total_count = len(df)
    ratio = filtered_count / total_count if total_count > 0 else 0

    # Create visual circle indicator
    circle_symbols = {
        1.0: "⚫", 0.75: "🔵", 0.5: "🟡", 0.25: "🟠"
    }
    circle_symbol = next((symbol for threshold, symbol in circle_symbols.items() if ratio >= threshold), "🔴")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {circle_symbol} Filtered: {filtered_count:,}/{total_count:,}")
        if total_count > 0:
            st.progress(ratio, text=f"{ratio:.1%} of total")
    
    with col2:
        if st.button("🔄 Refresh Data", help="Clear cache and reload all datasets"):
            refresh_cached_data()

    # Display Filtered Data with multi-row selection
    st.write(f"### Datasets ({len(display_df):,} shown)")

    if not display_df.empty:
        st.info("Click on rows to select datasets. Hold Ctrl/Cmd to select multiple rows.")
        
        # Use st.dataframe with multi-row selection
        selection_result = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            key="dataset_dataframe_selection",
            on_select="rerun",
            selection_mode="multi-row",
        )

        # Check if rows are selected
        if (
            selection_result
            and "selection" in selection_result
            and "rows" in selection_result["selection"]
        ):
            selected_rows = selection_result["selection"]["rows"]

            if len(selected_rows) > 0:
                # Get the dataset IDs from selected rows
                ds_ids = get_selected_dataset_info(
                    filtered_df, display_df, selected_rows
                )

                if ds_ids:
                    st.success(f"Selected {len(selected_rows)} dataset(s)")
                    
                    # Show selection details in an expander
                    with st.expander("Selection Details", expanded=True):
                        selected_details = []
                        for row_idx in selected_rows:
                            if row_idx < len(display_df):
                                row_data = display_df.iloc[row_idx]
                                selected_details.append(
                                    {
                                        "Dataset ID": row_data.get("id", "N/A"),
                                        "Name": row_data.get("name", "N/A"),
                                        "Title": row_data.get("title", "N/A"),
                                        "Author": row_data.get("author", "N/A"),
                                        "Organization": row_data.get(
                                            "organization_name", "N/A"
                                        ),
                                    }
                                )

                        if selected_details:
                            st.dataframe(
                                pd.DataFrame(selected_details),
                                use_container_width=True,
                                hide_index=True,
                            )

                    # Store in session state and return
                    st.session_state.ds_id = ds_ids
                    return ds_ids
                else:
                    st.error("Could not retrieve dataset IDs for the selected rows.")
            else:
                st.info("No datasets selected yet.")
        else:
            st.info("No datasets selected yet.")

    else:
        st.info("No data to display with current filters.")


# Entry point
if __name__ == "__main__":
    run()