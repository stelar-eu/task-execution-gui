import streamlit as st
import pandas as pd
from utils.fetch_process import fetch_processes


@st.cache_data(show_spinner=False)
def get_process_data(token):
    data = fetch_processes(token)
    if not data or "result" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["result"])
    # Check if organization column exists and handle None values
    if "organization" in df.columns:
        # Extract both values in a single pass, handling None values
        df["organization_name"] = df["organization"].apply(
            lambda x: x.get("name") if x is not None else None
        )
        df["organization_title"] = df["organization"].apply(
            lambda x: x.get("title") if x is not None else None
        )
    else:
        # If organization column doesn't exist, create empty columns
        df["organization_name"] = None
        df["organization_title"] = None

    return df


def refresh_cached_data():
    """Clear the cache and force data refresh"""
    get_process_data.clear()
    st.rerun()


def reset_filters():
    """Reset all filters to default state"""
    st.session_state["filters"] = {}


def reset_columns(default_columns):
    """Reset column selection to default"""
    st.session_state.selected_columns = default_columns
    st.rerun()


def apply_filters(df, filter_cols):
    """Apply filters to the dataframe"""
    filtered_df = df.copy()

    for col in filter_cols:
        if col not in df.columns:
            continue

        col_dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_dtype):
            min_val, max_val = df[col].min(), df[col].max()
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
            date_range = st.date_input(
                f"Date range for '{col}'",
                value=(
                    min_date.date() if pd.notna(min_date) else None,
                    max_date.date() if pd.notna(max_date) else None,
                ),
                key=f"date_{col}",
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df[col] >= pd.to_datetime(start_date))
                    & (filtered_df[col] <= pd.to_datetime(end_date))
                ]

        elif df[col].nunique() <= 20:
            options = st.multiselect(
                f"Select values for '{col}'",
                df[col].dropna().unique().tolist(),
                key=f"multi_{col}",
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


def get_selected_row_data(original_df, display_df, selected_row_index):
    """Get the process ID, tasks, and name from the selected row"""
    if selected_row_index is None or selected_row_index >= len(display_df):
        return None, None, None

    # Get the selected row from display dataframe
    selected_display_row = display_df.iloc[selected_row_index]

    # Find the corresponding row in original dataframe
    # We'll match by a unique identifier (assuming 'id' exists and is unique)
    if "id" in selected_display_row.index and "id" in original_df.columns:
        selected_id = selected_display_row["id"]
        original_row = original_df[original_df["id"] == selected_id].iloc[0]
        return original_row["id"], original_row["tasks"], original_row["name"]
    else:
        # Fallback: use index matching (less reliable if filtering is applied)
        original_row = original_df.iloc[selected_row_index]
        return original_row["id"], original_row["tasks"], original_row["name"]


def run():
    st.title("Workflow Process Selection")
    st.markdown("Select and filter workflow processes.")
    token = st.session_state.get("token")

    if not token:
        st.error("Authentication token not found. Please log in.")
        return None, None, None  # Return consistent tuple of 3 values

    # Add Update button at the top
    # col1, col2 = st.columns([1, 4])
    # with col1:
    #     if st.button("🔄 Update Data", key="update_process_data_btn", help="Refresh cached data to get latest processes"):
    #         refresh_cached_data()

    # Fetch data
    try:
        df = get_process_data(token)
        if df.empty:
            st.warning("No processes available.")
            return None, None, None  # Return consistent tuple of 3 values
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None, None  # Return consistent tuple of 3 values

    # Initialize session state
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = df.columns.tolist()

    # Advanced Filters
    with st.expander("🔍 Advanced Filters", expanded=False):
        st.markdown("Apply filters to narrow down the processes.")

        filter_cols = st.multiselect(
            "Select columns to filter",
            df.columns.tolist(),
            key="filter_columns_workflows",
        )

        # Apply filters
        filtered_df = apply_filters(df, filter_cols)

        # Reset filters button
        if st.button("Reset Filters", key="reset_filters_btn_workflows"):
            reset_filters()
            st.rerun()

    # Column Visibility
    with st.expander("🧩 Column Visibility", expanded=False):
        st.markdown("Select which columns to display.")

        selected_columns = st.multiselect(
            "Columns to display",
            options=df.columns.tolist(),
            default=["name", "title", "author", "organization_name", "exec_state"],
            key="column_selector_workflows",
        )

        # Update session state
        st.session_state.selected_columns = selected_columns

        # Reset columns button
        if st.button("Reset Columns to Default", key="reset_columns_btn_workflows"):
            reset_columns(df.columns.tolist())

    # Apply column selection
    if selected_columns:
        display_df = filtered_df[selected_columns]
    else:
        display_df = filtered_df
        st.warning("No columns selected. Showing all columns.")

    # Display summary stats with visual ratio
    filtered_count = len(display_df)
    total_count = len(df)
    ratio = filtered_count / total_count if total_count > 0 else 0

    # Create visual circle indicator
    if ratio == 1.0:
        circle_symbol = "⚫"  # Full circle when all processes shown
    elif ratio >= 0.75:
        circle_symbol = "🔵"  # Blue circle for high ratio
    elif ratio >= 0.5:
        circle_symbol = "🟡"  # Yellow circle for medium ratio
    elif ratio >= 0.25:
        circle_symbol = "🟠"  # Orange circle for low-medium ratio
    else:
        circle_symbol = "🔴"  # Red circle for very low ratio

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"### {circle_symbol} Filtered Processes: {filtered_count}/{total_count}"
        )
        if total_count > 0:
            st.progress(ratio, text=f"{ratio:.1%} of total")

    # Display Filtered Data with single row selection
    st.write(f"### 📄 Filtered Processes ({len(display_df)} rows)")

    if not display_df.empty:
        # Use st.dataframe with single row selection
        selection_result = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            key="process_dataframe_workflows",
            on_select="rerun",
            selection_mode="single-row",
        )

        # Check if a row is selected
        if (
            selection_result
            and "selection" in selection_result
            and "rows" in selection_result["selection"]
        ):
            selected_rows = selection_result["selection"]["rows"]

            if len(selected_rows) > 0:
                selected_row_index = selected_rows[0]  # Get the first (and only) selected row

                # Get the process ID, tasks, and name from the selected row
                with st.expander("🔍 Selected Process Tasks", expanded=False):
                    st.markdown("Click on a row to view detailed task information.")
                    proc_id, tasks, proc_name = get_selected_row_data(
                        filtered_df, display_df, selected_row_index
                    )

                    if proc_id is not None and tasks is not None:
                        st.write(f"### 🛠️ Selected Tasks for Process ID: {proc_id}")
                        st.dataframe(tasks, use_container_width=False)
                        st.session_state.proc_id = proc_id
                        st.session_state.tasks = tasks
                        st.session_state.proc_name = proc_name
                        return proc_id, tasks, proc_name  # Return consistent tuple of 3 values
                    else:
                        st.error("Could not retrieve process details for the selected row.")
                        return None, None, None  # Return consistent tuple of 3 values
            else:
                st.info("Click on a row to view detailed task information.")
                return None, None, None  # Return consistent tuple of 3 values
        else:
            st.info("Click on a row to view detailed task information.")
            return None, None, None  # Return consistent tuple of 3 values

    else:
        st.info("No data to display with current filters.")
        return None, None, None  # Return consistent tuple of 3 values

# Entry point
if __name__ == "__main__":
    run()