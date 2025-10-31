import streamlit as st
import pandas as pd
from utils.fetch_datasets import fetch_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


@st.cache_data(show_spinner=False)
def get_process_data(token, limit=100, offset=0):
    """
    Dataset loading with concurrent strategy and configurable parameters.
    
    Parameters:
        token (str): Bearer token for authentication
        limit (int): Number of datasets per request (default: 100)
        offset (int): Starting position in dataset (default: 0)
    
    Returns:
        tuple: (pd.DataFrame, dict) - DataFrame and metadata
    """
    try:
        # Load 3 batches concurrently with smaller batch size
        batches_to_load = 3
        all_results = []
        total_count = 0
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit concurrent requests
            futures = {}
            for i in range(batches_to_load):
                batch_offset = offset + (i * limit)
                future = executor.submit(fetch_datasets, token, limit, batch_offset)
                futures[future] = i
            
            # Collect results in order
            results_by_order = {}
            for future in as_completed(futures):
                try:
                    batch_data = future.result()
                    batch_order = futures[future]
                    results_by_order[batch_order] = batch_data.get("result", [])
                    
                    if batch_order == 0:  # Get total from first request
                        total_count = batch_data.get("total", 0) or batch_data.get("count", 0)
                except Exception as e:
                    st.warning(f"Failed to fetch batch {futures[future]}: {e}")
                    # Continue with other batches even if one fails
            
            # Combine results in order
            for i in range(batches_to_load):
                if i in results_by_order:
                    all_results.extend(results_by_order[i])
        
        data = {"result": all_results, "total": total_count}
        
        if not data or "result" not in data or not all_results:
            return pd.DataFrame(), {"total": 0, "loaded": 0, "offset": offset}

        df = pd.DataFrame(data["result"])
        
        # Process organization columns
        if not df.empty and "organization" in df.columns:
            df["organization_name"] = df["organization"].apply(
                lambda x: x.get("name") if x is not None else None
            )
            df["organization_title"] = df["organization"].apply(
                lambda x: x.get("title") if x is not None else None
            )
        elif not df.empty:
            df["organization_name"] = None
            df["organization_title"] = None

        metadata = {
            "total": total_count,
            "loaded": len(df),
            "offset": offset,
            "limit": limit,
            "has_more": len(df) == limit and (offset + len(df)) < total_count
        }
        
        return df, metadata
        
    except Exception as e:
        st.error(f"Error fetching datasets: {e}")
        return pd.DataFrame(), {"total": 0, "loaded": 0, "offset": offset}


@st.cache_data(show_spinner=False)
def get_process_data_single(token, limit=100, offset=0):
    """
    Fallback: Single request strategy for when concurrent requests fail.
    
    Parameters:
        token (str): Bearer token for authentication
        limit (int): Number of datasets per request
        offset (int): Starting position in dataset
    
    Returns:
        tuple: (pd.DataFrame, dict) - DataFrame and metadata
    """
    try:
        data = fetch_datasets(token, limit=limit, offset=offset)
        
        if not data or "result" not in data:
            return pd.DataFrame(), {"total": 0, "loaded": 0, "offset": offset}

        df = pd.DataFrame(data.get("result", []))
        
        # Process organization columns
        if not df.empty and "organization" in df.columns:
            df["organization_name"] = df["organization"].apply(
                lambda x: x.get("name") if x is not None else None
            )
            df["organization_title"] = df["organization"].apply(
                lambda x: x.get("title") if x is not None else None
            )
        elif not df.empty:
            df["organization_name"] = None
            df["organization_title"] = None

        total_count = data.get("total", 0) or data.get("count", 0)
        
        metadata = {
            "total": total_count,
            "loaded": len(df),
            "offset": offset,
            "limit": limit,
            "has_more": len(df) == limit and (offset + len(df)) < total_count
        }
        
        return df, metadata
        
    except Exception as e:
        st.error(f"Error fetching datasets: {e}")
        return pd.DataFrame(), {"total": 0, "loaded": 0, "offset": offset}


def load_more_datasets(token, current_offset, limit=100):
    """Load next batch of datasets"""
    try:
        data = fetch_datasets(token, limit=limit, offset=current_offset)
        df = pd.DataFrame(data.get("result", []))
        
        # Process organization columns
        if not df.empty and "organization" in df.columns:
            df["organization_name"] = df["organization"].apply(
                lambda x: x.get("name") if x is not None else None
            )
            df["organization_title"] = df["organization"].apply(
                lambda x: x.get("title") if x is not None else None
            )
        elif not df.empty:
            df["organization_name"] = None
            df["organization_title"] = None
            
        return df
    except Exception as e:
        st.error(f"Error loading more datasets: {e}")
        return pd.DataFrame()


def refresh_cached_data():
    """Clear the cache and force data refresh"""
    get_process_data.clear()
    get_process_data_single.clear()
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
    st.markdown("Select and filter datasets.")
    
    token = st.session_state.get("token")
    if not token:
        st.error("Authentication token not found. Please log in.")
        return

    # Initialize session state for pagination
    if "dataset_offset" not in st.session_state:
        st.session_state.dataset_offset = 0
    
    if "use_concurrent" not in st.session_state:
        st.session_state.use_concurrent = True
    
    # Reduced limit to avoid server overload
    limit = 100
    
    # Fetch data with error handling
    with st.spinner("Loading datasets..."):
        if st.session_state.use_concurrent:
            df, metadata = get_process_data(
                token, 
                limit=limit,
                offset=st.session_state.dataset_offset
            )
            
            # If concurrent loading fails, fall back to single request
            if df.empty and metadata['total'] == 0:
                st.warning("Concurrent loading failed. Trying single request...")
                st.session_state.use_concurrent = False
                df, metadata = get_process_data_single(
                    token,
                    limit=limit,
                    offset=st.session_state.dataset_offset
                )
        else:
            df, metadata = get_process_data_single(
                token,
                limit=limit,
                offset=st.session_state.dataset_offset
            )

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
        st.markdown("Apply filters to narrow down the current batch.")

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

        selected_columns = st.multiselect(
            "Columns to display",
            options=df.columns.tolist(),
            default=["id", "name", "title", "author", "organization_name"],
            key="column_selector_datasets",
        )

        # Update session state
        st.session_state.selected_columns = selected_columns

        # Reset columns button
        if st.button("Reset Columns to Default", key="reset_columns_btn_datasets"):
            reset_columns(df.columns.tolist())

    # Apply column selection
    if selected_columns:
        display_df = filtered_df[selected_columns] if not filtered_df.empty else filtered_df
    else:
        display_df = filtered_df
        st.warning("No columns selected. Showing all columns.")

    # Display summary stats with visual ratio
    filtered_count = len(display_df)
    total_loaded = len(df)
    ratio = filtered_count / total_loaded if total_loaded > 0 else 0

    # Create visual circle indicator
    circle_symbols = {
        1.0: "⚫", 0.75: "🔵", 0.5: "🟡", 0.25: "🟠"
    }
    circle_symbol = next((symbol for threshold, symbol in circle_symbols.items() if ratio >= threshold), "🔴")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### {circle_symbol} Filtered: {filtered_count}/{total_loaded}")
        if total_loaded > 0:
            st.progress(ratio, text=f"{ratio:.1%} of loaded")

    # Calculate max pages based on total and limit
    max_pages = min((metadata['total'] // limit) + 1, 50) if metadata['total'] > 0 else 5

    # Page slider and refresh button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        current_page = (st.session_state.dataset_offset // limit) + 1
        page_number = st.slider(
            "Navigate to page:",
            min_value=1,
            max_value=max_pages,
            value=current_page,
            step=1,
            help=f"Navigate through pages ({limit} datasets per page)"
        )
        
        # Update offset based on page selection
        new_offset = (page_number - 1) * limit
        if new_offset != st.session_state.dataset_offset:
            st.session_state.dataset_offset = new_offset
            get_process_data.clear()
            get_process_data_single.clear()
            st.rerun()

    with col2:
        if st.button("🔄 Refresh", help="Clear cache and reload data"):
            refresh_cached_data()

    # Display dataset information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_idx = st.session_state.dataset_offset + 1
        end_idx = st.session_state.dataset_offset + metadata['loaded']
        st.metric("Showing", f"{start_idx:,} - {end_idx:,}")
    
    with col2:
        st.metric("Page", f"{page_number} / {max_pages}")
    
    with col3:
        st.metric("Total Available", f"{metadata['total']:,}")

    # Progress indicator
    if metadata['total'] > 0:
        progress = (st.session_state.dataset_offset + metadata['loaded']) / metadata['total']
        st.progress(min(progress, 1.0), text=f"Dataset position: {progress:.1%}")

    # Display Filtered Data with multi-row selection
    st.write(f"### Datasets ({len(display_df)} shown)")

    if not display_df.empty:
        st.info("Click on rows to select datasets. Hold Ctrl/Cmd to select multiple rows.")
        
        # Use st.dataframe with multi-row selection
        selection_result = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            key=f"dataset_dataframe_selection_{st.session_state.dataset_offset}",
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