import importlib
import sys
import streamlit as st
import pandas as pd
from utils.fetch_tools import fetch_tools
from utils.fetch_task_by_id import fetch_task_by_id
from utils.fetch_organization import fetch_organization
from utils.create_new_dataset import create_new_ds
from utils.execute_task import execute_task


from pages import Workflow, Datasets, S3
from utils.tool_configs import get_tool_config


@st.cache_data(show_spinner=False)
def get_tools(token):
    """Fetch and process tools data with caching"""
    data = fetch_tools(token)
    if not data or "result" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["result"])

    # Extract both values in a single pass
    org_data = df["organization"].apply(
        lambda x: {"name": x.get("name"), "title": x.get("title")}
    )
    df["organization_name"] = org_data.apply(lambda x: x["name"])
    df["organization_title"] = org_data.apply(lambda x: x["title"])

    return df


@st.cache_data(show_spinner=False)
def get_organization_data(token):
    """Fetch organization data with caching"""
    return fetch_organization(token)


def refresh_cached_data():
    """Clear all caches and force data refresh"""
    get_tools.clear()
    get_organization_data.clear()
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


def get_selected_tool_info(display_df, selected_indices):
    """Get tool information from selected row"""
    if not selected_indices or selected_indices[0] >= len(display_df):
        return None
        
    row_data = display_df.iloc[selected_indices[0]]
    return {
        'name': row_data.get('name'),
        'inputs': row_data.get('inputs'),
        'outputs': row_data.get('outputs'),
        'parameters': row_data.get('parameters')
    }


def render_selection_status(proc_id, ds_id, s3_path):
    """Render the selection status indicators"""
    st.write("### Selection Status")
    col_status1, col_status2, col_status3 = st.columns(3)
    
    # Check selections
    workflow_selected = proc_id is not None and proc_id != "No output"
    dataset_selected = ds_id is not None and ds_id != "No output"
    s3_selected = s3_path and s3_path != "" and s3_path != "Select S3 Path"
    
    with col_status1:
        if workflow_selected:
            st.success("✅ Workflow Process Selected")
        else:
            st.warning("⏳ Workflow Process Pending")
    
    with col_status2:
        if dataset_selected:
            st.success("✅ Dataset Selected")
        else:
            st.warning("⏳ Dataset Pending")
    
    with col_status3:
        if s3_selected:
            st.success("✅ S3 Path Selected")
        else:
            st.warning("⏳ S3 Path Pending")
    
    return workflow_selected and dataset_selected and s3_selected


def get_tool_config_template(tool_name):
    """Get configuration template for different tools"""
    templates = {
        "agri-products-match": {
            "modes": ["fertilizers", "pesticides"],
            "inputs": {
                "fertilizers": ["npk_values", "fertilizer_dataset"],
                "pesticides": ["active_substances", "pesticides_dataset"]
            },
            "outputs": {
                "fertilizers": "matched_fertilizers",
                "pesticides": "matched_products"
            },
            "parameters": {
                "fertilizers": ["mode"],
                "pesticides": ["mode", "input_language", "db_language"]
            }
        },
        "missing-data-interpolation": {
            "modes": ["interpolation"],
            "inputs": {
                "interpolation": ["meteo_file", "coords_file"]
            },
            "outputs": {
                "interpolation": "interpolated_file"
            },
            "parameters": {
                "interpolation": []
            }
        },
        "vocational-score-raster": {
            "modes": ["classification"],
            "inputs": {
                "classification": ["rasters"]
            },
            "outputs": {
                "classification": "scored_files"
            },
            "parameters": {
                "classification": ["raster_parameters"]  # Dynamic parameters based on raster files
            }
        }
    }
    
    return templates.get(tool_name.lower().replace(" ", "-"), {
        "modes": ["default"],
        "inputs": {"default": []},
        "outputs": {"default": "output"},
        "parameters": {"default": []}
    })


def create_task_config_agri_products(task_name, selected_mode, task_pid, input_datasets, output_dataset, s3_path_final, filename, proc_id=None, ds_ids=None, additional_params=None):
    """Create task configuration JSON for agri-products-match tool"""
    actual_proc_id = proc_id if proc_id else task_pid
    
    # Map dataset names to IDs if ds_ids is provided
    if ds_ids and isinstance(ds_ids, (list, tuple)) and len(ds_ids) >= 2:
        if isinstance(ds_ids[0], list) and isinstance(ds_ids[1], list):
            name_to_id = dict(zip(ds_ids[1], ds_ids[0]))
            input_datasets = [name_to_id.get(ds, ds) for ds in input_datasets]
            output_dataset = name_to_id.get(output_dataset, output_dataset)
    
    if selected_mode.lower() == "fertilizers":
        config = {
            "process_id": actual_proc_id,
            "name": task_name,
            "image": "petroud/agri-products-match:latest",
            "inputs": {
                "npk_values": [input_datasets[0]] if len(input_datasets) > 0 else [],
                "fertilizer_dataset": [input_datasets[1]] if len(input_datasets) > 1 else [],
            },
            "datasets": {"d0": output_dataset},
            "parameters": {"mode": "fertilizers"},
            "outputs": {
                "matched_fertilizers": {
                    "url": f"s3://{s3_path_final}/{filename}" if not s3_path_final.endswith('/') else f"s3://{s3_path_final}{filename}",
                    "dataset": "d0",
                    "resource": {
                        "name": "Matched Fertilizers based on NPK values",
                        "relation": "matched",
                    },
                }
            },
        }
    else:  # pesticides
        config = {
            "process_id": actual_proc_id,
            "name": task_name,
            "image": "petroud/agri-products-match:latest",
            "inputs": {
                "active_substances": [input_datasets[0]] if len(input_datasets) > 0 else [],
                "pesticides_dataset": [input_datasets[1]] if len(input_datasets) > 1 else [],
            },
            "datasets": {"d0": output_dataset},
            "parameters": {
                "mode": "pesticides",
                "input_language": additional_params.get("input_language", "italiano") if additional_params else "italiano",
                "db_language": additional_params.get("db_language", "italiano") if additional_params else "italiano"
            },
            "outputs": {
                "matched_products": {
                    "url": f"s3://{s3_path_final}/{filename}" if not s3_path_final.endswith('/') else f"s3://{s3_path_final}{filename}",
                    "dataset": "d0",
                    "resource": {
                        "name": "Matched Pesticides based on active substances values",
                        "relation": "matched",
                    },
                }
            },
        }
    
    return config

def create_task_config_mdi(task_name, task_pid, input_datasets, output_dataset, s3_path_final, filename, proc_id=None, ds_ids=None):
    """Create task configuration JSON for missing-data-interpolation tool"""
    actual_proc_id = proc_id if proc_id else task_pid
    
    # Map dataset names to IDs if ds_ids is provided
    if ds_ids and isinstance(ds_ids, (list, tuple)) and len(ds_ids) >= 2:
        if isinstance(ds_ids[0], list) and isinstance(ds_ids[1], list):
            name_to_id = dict(zip(ds_ids[1], ds_ids[0]))
            input_datasets = [name_to_id.get(ds, ds) for ds in input_datasets]
            output_dataset = name_to_id.get(output_dataset, output_dataset)
    
    return {
        "process_id": actual_proc_id,
        "name": task_name,
        "image": "petroud/mdi:latest",
        "inputs": {
            "meteo_file": [input_datasets[0]] if len(input_datasets) > 0 else [],
            "coords_file": [input_datasets[1]] if len(input_datasets) > 1 else [],
        },
        "datasets": {"d0": output_dataset},
        "parameters": {},
        "outputs": {
            "interpolated_file": {
                "url": f"s3://{s3_path_final}/{filename}" if not s3_path_final.endswith('/') else f"s3://{s3_path_final}{filename}",
                "dataset": "d0",
                "resource": {
                    "name": "Interpolated Meteo Station Data",
                    "relation": "owned"
                }
            }
        }
    }

def create_task_config_vsr(task_name, task_pid, input_datasets, output_dataset, s3_path_final, filename, proc_id=None, ds_ids=None, raster_params=None, dataset_config=None):
    """Create task configuration JSON for VSR tool with enhanced parameter handling"""
    actual_proc_id = proc_id if proc_id else task_pid
    
    # Map dataset names to IDs if ds_ids is provided
    if ds_ids and isinstance(ds_ids, (list, tuple)) and len(ds_ids) >= 2:
        if isinstance(ds_ids[0], list) and isinstance(ds_ids[1], list):
            name_to_id = dict(zip(ds_ids[1], ds_ids[0]))
            input_datasets = [name_to_id.get(ds, ds) for ds in input_datasets]
            if isinstance(output_dataset, str):
                output_dataset = name_to_id.get(output_dataset, output_dataset)
    
    # Handle dataset configuration - determine d0 (input) and d1 (output)
    datasets_config = {}
    
    # d0 is the input raster dataset
    if input_datasets and input_datasets[0]:
        datasets_config["d0"] = input_datasets[0]
    else:
        # Fallback to a default or raise error
        datasets_config["d0"] = "16adb665-77ea-410c-8476-132e34160b53"  # Default from example
    
    # d1 is the output dataset configuration
    if dataset_config:
        # Use the custom dataset configuration
        datasets_config["d1"] = dataset_config
    elif isinstance(output_dataset, dict):
        # Output dataset is already a configuration object
        datasets_config["d1"] = output_dataset
    elif isinstance(output_dataset, str) and output_dataset not in ["Select Output Dataset", "No Dataset Selected", "", None]:
        # Use existing dataset ID
        datasets_config["d1"] = output_dataset
    else:
        # Create a default configuration if no specific output dataset is provided
        datasets_config["d1"] = {
            "name": f"{task_name.lower().replace(' ', '-')}-output",
            "owner_org": "default-org",
            "notes": f"Output from {task_name} VSR execution",
            "tags": ["VSR", "STELAR", "Raster"]
        }
    
    # Process raster parameters - ensure they're in the correct format
    processed_params = {}
    if raster_params and isinstance(raster_params, dict):
        processed_params = raster_params.copy()
    
    # Validate parameter structure
    for raster_name, config in processed_params.items():
        if not isinstance(config, dict):
            st.warning(f"Invalid parameter format for {raster_name}")
            continue
        
        # Ensure all required keys exist
        required_keys = ["val_min", "val_max", "new_val"]
        for key in required_keys:
            if key not in config:
                st.warning(f"Missing '{key}' parameter for {raster_name}")
                config[key] = 0.0  # Default value
    
    # Construct the full configuration
    config = {
        "process_id": actual_proc_id,
        "name": task_name,
        "image": "petroud/vsr:latest",
        "inputs": {
            "rasters": "d0::owned"
        },
        "datasets": datasets_config,
        "parameters": processed_params,
        "outputs": {
            "scored_files": {
                "url": f"s3://{s3_path_final}/output" if not filename else (f"s3://{s3_path_final}/{filename}" if not s3_path_final.endswith('/') else f"s3://{s3_path_final}{filename}"),
                "dataset": "d1",
                "resource": {
                    "name": "Scored Classified rasters via VSR",
                    "relation": "raster"
                }
            }
        }
    }
    
    # Add validation summary
    if processed_params:
        st.success(f"✅ VSR Configuration created with {len(processed_params)} raster parameters")
        
        # Display parameter summary
        with st.expander("📊 Parameter Summary"):
            param_summary = []
            for raster_name, raster_config in processed_params.items():
                param_summary.append({
                    "Raster": raster_name,
                    "Classification Range": f"{raster_config['val_min']} - {raster_config['val_max']}",
                    "Output Value": raster_config['new_val']
                })
            
            if param_summary:
                summary_df = pd.DataFrame(param_summary)
                st.dataframe(summary_df, hide_index=True)
    else:
        st.warning("⚠️ No raster parameters configured. The VSR tool may not work correctly without proper parameter configuration.")
    
    return config


def render_tool_parameters(tool_name, selected_mode=None):
    """Render tool-specific parameter inputs"""
    params = {}
    
    if tool_name == "agri-products-match" and selected_mode == "pesticides":
        col1, col2 = st.columns(2)
        with col1:
            params["input_language"] = st.selectbox(
                "Input Language",
                ["italiano", "english", "français", "deutsch"],
                key="input_language_select"
            )
        with col2:
            params["db_language"] = st.selectbox(
                "Database Language", 
                ["italiano", "english", "français", "deutsch"],
                key="db_language_select"
            )
    
    elif tool_name == "vocational-score-raster":
        st.subheader("Raster Parameters Configuration")
        st.info("Configure parameters for each raster file. Add as many raster configurations as needed.")
        
        if "raster_params" not in st.session_state:
            st.session_state.raster_params = {}
        
        # Add new raster parameter
        with st.expander("Add Raster Parameter"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                raster_name = st.text_input("Raster Filename", key="new_raster_name")
            with col2:
                val_min = st.number_input("Min Value", key="new_val_min", format="%.2f")
            with col3:
                val_max = st.number_input("Max Value", key="new_val_max", format="%.2f")
            with col4:
                new_val = st.number_input("New Value", key="new_val", format="%.2f")
            
            if st.button("Add Raster Parameter"):
                if raster_name:
                    st.session_state.raster_params[raster_name] = {
                        "val_min": val_min,
                        "val_max": val_max, 
                        "new_val": new_val
                    }
                    st.success(f"Added parameters for {raster_name}")
                    st.rerun()
        
        # Display current parameters
        if st.session_state.raster_params:
            st.subheader("Current Raster Parameters")
            for raster_name, raster_config in st.session_state.raster_params.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{raster_name}**: min={raster_config['val_min']}, max={raster_config['val_max']}, new_val={raster_config['new_val']}")
                with col2:
                    if st.button(f"Remove", key=f"remove_{raster_name}"):
                        del st.session_state.raster_params[raster_name]
                        st.rerun()
        
        params["raster_parameters"] = st.session_state.raster_params
        
        # Dataset configuration for VSR
        # st.subheader("Output Dataset Configuration")
        # use_existing = st.radio(
        #     "Dataset Type",
        #     ["Use Existing Dataset", "Create New Dataset Configuration"],
        #     key="vsr_dataset_type"
        # )
        
        # if use_existing == "Create New Dataset Configuration":
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         dataset_name = st.text_input("Dataset Name", key="vsr_dataset_name")
        #         owner_org = st.text_input("Owner Organization", key="vsr_owner_org")
        #     with col2:
        #         notes = st.text_area("Dataset Notes", key="vsr_dataset_notes")
        #         tags_input = st.text_input("Tags (comma-separated)", key="vsr_dataset_tags")
        #         tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
            
        #     if dataset_name and owner_org:
        #         params["dataset_config"] = {
        #             "name": dataset_name,
        #             "owner_org": owner_org,
        #             "notes": notes,
        #             "tags": tags
        #         }
    
    return params


def get_input_labels(tool_name, selected_mode=None):
    """Get input field labels for different tools and modes"""
    labels = {
        "agri-products-match": {
            "fertilizers": ["NPK Values Dataset", "Fertilizer Dataset"],
            "pesticides": ["Active Substances Dataset", "Pesticides Dataset"]
        },
        "missing-data-interpolation": {
            "interpolation": ["Meteorological File Dataset", "Coordinates File Dataset"]
        },
        "vsr": {
            "classification": ["Raster Dataset"]
        }
    }
    
    tool_key = tool_name.lower().replace(" ", "-")
    mode_key = selected_mode.lower() if selected_mode else "default"
    
    return labels.get(tool_key, {}).get(mode_key, ["Input Dataset 1", "Input Dataset 2"])




def handle_task_execution(task_execution, tool_name, selected_mode, proc_id, proc_name, ds_id, s3_path, tasks, token, owner_org):
    """Handle task execution logic"""
    if task_execution == "Create New Task":
        st.header("Task Execution")
        
        # Clear session state for new task
        for key in ["proc_id", "proc_name", "ds_id", "s3_path"]:
            st.session_state[key] = None
            
        # Get current values
        pid = proc_id or st.session_state.get("proc_id", "No output")
        pname = proc_name or st.session_state.get("proc_name", "No output")
        did = ds_id or st.session_state.get("ds_id", "No output")
        current_s3_path = s3_path or st.session_state.get("s3_path", "")

        # Task configuration inputs
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            task_name = st.text_input(
                label="Task Name",
                key="task_name",
                placeholder="Enter task name",
            )
            
        with c2:
            # Get process names for display, but store actual process ID
            process_names = pname if isinstance(pname, list) else [pname] if pname != "No output" else ["No Process Selected"]
            task_pid = st.selectbox(
                options=process_names, 
                key="task_pid_selectbox", 
                label="Select Process ID"
            )
            
        with c3:
            # Get dataset names for display
            dataset_names = []
            if isinstance(did, (list, tuple)) and len(did) > 1:
                dataset_names = did[1] if isinstance(did[1], list) else [str(did[1])]
            else:
                dataset_names = [str(did)] if did != "No output" else ["No Dataset Selected"]
            
            # Get input labels based on tool and mode
            input_labels = get_input_labels(tool_name, selected_mode)
            input_datasets = []
            
            # Dynamic input dataset selection based on tool requirements
            for i, label in enumerate(input_labels):
                selected_dataset = st.selectbox(
                    options=dataset_names,
                    key=f"task_did_selectbox_{i}",
                    label=f"Select {label}",
                )
                input_datasets.append(selected_dataset)
            
            # Output dataset handling
            st.info("Create Output Dataset, if not yet created", icon="ℹ️")
            
            output_ds = st.segmented_control(
                label="Choose Output Dataset",
                options=["Existing Output Dataset", "New Output Dataset"],
                key="output_ds_segmented_control",
            )

            # Handle dataset creation
            new_ds_name = None
            if output_ds == "New Output Dataset":
                with st.expander("Create New Dataset"):
                    new_ds_name = st.text_input(
                        label="New Dataset Name",
                        key="new_dataset_name",
                        placeholder="Enter new dataset name",
                    )
                    owner = st.selectbox(
                        label="Owner Organization",
                        key="new_dataset_owner_org",
                        options=owner_org if isinstance(owner_org, list) else [owner_org],
                    )
                    if st.button("Create Dataset", key="create_dataset_button"):
                        if new_ds_name and owner:
                            try:
                                new_ds_id, created_name = create_new_ds(
                                    token=token,
                                    name=new_ds_name,
                                    owner_org=owner,
                                )
                                st.success("New dataset created successfully!")
                                new_ds_name = created_name
                            except Exception as e:
                                st.error(f"Error creating dataset: {str(e)}")

            # Output dataset selection
            if output_ds == "Existing Output Dataset":
                output_options = dataset_names
            else:
                output_options = ['Select Output Dataset', new_ds_name] if new_ds_name else ['Select Output Dataset']
            
            output_dataset = st.selectbox(
                options=output_options,
                key="output_dataset_selectbox",
                label="Select Output Dataset ID",
            )
            
        with c4:
            s3_path_options = (
                [current_s3_path] if current_s3_path and current_s3_path != "" 
                else ["Select S3 Path"]
            )
            s3_path_final = st.selectbox(
                options=s3_path_options,
                key="s3_path_selectbox",
                label="Select S3 Path",
            )
            filename = st.text_input(
                label="File Name",
                key="file_name",
                placeholder="Enter file name (e.g., output.csv)",
            )
        
        # Tool-specific parameters
        additional_params = render_tool_parameters(tool_name, selected_mode)
        
        # Execute task
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Execute Task", key="execute_task_button", type="primary", use_container_width=True):
                # Validate required fields based on tool requirements
                required_fields = [task_name, task_pid, s3_path_final, filename]
                required_fields.extend(input_datasets)
                required_fields.append(output_dataset)
                
                invalid_values = ["Select Output Dataset", "Select S3 Path", "No Process Selected", "No Dataset Selected", "", None]
                
                if all(field not in invalid_values for field in required_fields):
                    with st.spinner("Executing task..."):
                        # Create configuration based on tool type
                        tool_key = tool_name.lower().replace(" ", "-")
                        
                        if tool_key == "agri-products-match":
                            config = create_task_config_agri_products(
                                task_name, selected_mode, task_pid, input_datasets, 
                                output_dataset, s3_path_final, filename,
                                proc_id=pid, ds_ids=did, additional_params=additional_params
                            )
                        elif tool_key == "missing-data-interpolation":
                            config = create_task_config_mdi(
                                task_name, task_pid, input_datasets, output_dataset,
                                s3_path_final, filename, proc_id=pid, ds_ids=did
                            )
                        elif tool_key == "vocational-score-raster":
                            config = create_task_config_vsr(
                                task_name, task_pid, input_datasets, output_dataset,
                                s3_path_final, filename, proc_id=pid, ds_ids=did,
                                raster_params=additional_params.get("raster_parameters"),
                                dataset_config=additional_params.get("dataset_config")
                            )
                        else:
                            st.error(f"Unknown tool: {tool_name}")
                            return
                        
                        # Execute the task via POST request
                        result = execute_task(token, config)
                        
                        if result["success"]:
                            st.success("✅ Task executed successfully!")
                            
                            # Display task details
                            with st.expander("📋 Task Execution Details", expanded=True):
                                if "data" in result and result["data"]:
                                    response_data = result["data"]
                                    
                                    # Create a summary
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Status Code", result["status_code"])
                                    with col_b:
                                        if "result" in response_data and "id" in response_data["result"]:
                                            st.metric("Task ID", response_data["result"]["id"])
                                    with col_c:
                                        if "result" in response_data and "state" in response_data["result"]:
                                            st.metric("State", response_data["result"]["state"])
                                    
                                    # Show full response
                                    st.json(response_data, expanded=False)
                                else:
                                    st.info("Task submitted successfully. No additional details available.")
                            
                            # Show configuration that was sent
                            with st.expander("📄 Task Configuration (Sent)", expanded=False):
                                st.json(config, expanded=True)
                                
                        else:
                            st.error(f"❌ Task execution failed!")
                            
                            with st.expander("🔍 Error Details", expanded=True):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Status Code", result.get("status_code", "N/A"))
                                with col_b:
                                    st.write("**Error:**")
                                    st.code(result.get("error", "Unknown error"))
                                
                                if result.get("response"):
                                    st.write("**Response from server:**")
                                    st.json(result["response"], expanded=True)
                                
                                # Show configuration that was attempted
                                st.write("**Configuration that was sent:**")
                                st.json(config, expanded=False)
                else:
                    st.error("❌ Please fill all required fields before executing the task.")
                    
                    # Show which fields are missing
                    with st.expander("Missing Fields"):
                        missing_fields = []
                        if not task_name or task_name in invalid_values:
                            missing_fields.append("Task Name")
                        if task_pid in invalid_values:
                            missing_fields.append("Process ID")
                        if s3_path_final in invalid_values:
                            missing_fields.append("S3 Path")
                        if not filename or filename in invalid_values:
                            missing_fields.append("File Name")
                        for i, ds in enumerate(input_datasets):
                            if ds in invalid_values:
                                missing_fields.append(f"Input Dataset {i+1}")
                        if output_dataset in invalid_values:
                            missing_fields.append("Output Dataset")
                        
                        if missing_fields:
                            st.write("Please provide values for:")
                            for field in missing_fields:
                                st.write(f"- {field}")
        
        with col2:
            # Preview configuration button
            if st.button("👁️ Preview JSON", key="preview_json_button", use_container_width=True):
                # Generate config for preview
                tool_key = tool_name.lower().replace(" ", "-")
                
                try:
                    if tool_key == "agri-products-match":
                        config = create_task_config_agri_products(
                            task_name or "preview-task", selected_mode, task_pid, input_datasets, 
                            output_dataset, s3_path_final, filename or "output.csv",
                            proc_id=pid, ds_ids=did, additional_params=additional_params
                        )
                    elif tool_key == "missing-data-interpolation":
                        config = create_task_config_mdi(
                            task_name or "preview-task", task_pid, input_datasets, output_dataset,
                            s3_path_final, filename or "output.csv", proc_id=pid, ds_ids=did
                        )
                    elif tool_key == "vocational-score-raster":
                        config = create_task_config_vsr(
                            task_name or "preview-task", task_pid, input_datasets, output_dataset,
                            s3_path_final, filename or "output", proc_id=pid, ds_ids=did,
                            raster_params=additional_params.get("raster_parameters"),
                            dataset_config=additional_params.get("dataset_config")
                        )
                    else:
                        st.error(f"Unknown tool: {tool_name}")
                        return
                    
                    with st.expander("📄 Configuration Preview", expanded=True):
                        st.json(config, expanded=True)
                        
                except Exception as e:
                    st.error(f"Error generating preview: {str(e)}")
                    
    elif task_execution == "Modify Existing Task":
        st.header("Modify Existing Task")
        st.warning("Please fill all fields before executing the task.")
        
        mod_task = tasks or st.session_state.get("tasks", "No output")
        if isinstance(mod_task, list) and mod_task:
            mod_task_df = pd.DataFrame(mod_task)
            
            selection_result = st.dataframe(
                data=mod_task_df, 
                use_container_width=False,
                selection_mode="single-row",
                key="modify_task_dataframe",
                on_select="rerun",
            )
            
            if (selection_result and selection_result.get("selection", {}).get("rows")):
                row_index = selection_result["selection"]["rows"][0]
                selected_task_id = mod_task_df.iloc[row_index]["id"]
                
                st.session_state.selected_task_id = selected_task_id
                st.success(f"Selected Task ID: {selected_task_id}")

                try:
                    task_details = fetch_task_by_id(selected_task_id, token)
                    if task_details and 'result' in task_details:
                        st.dataframe(task_details['result'], use_container_width=False)
                    else:
                        st.error("No task details found for the selected ID.")
                except Exception as e:
                    st.error(f"Error fetching task details: {str(e)}")
        else:
            st.info("No tasks available to modify.")
def process_tool_request(tool_name, custom_params=None):
    """Process a tool request using the configuration"""
    try:
        base_config = get_tool_config(tool_name)
        config = base_config.copy()
        
        if custom_params:
            config['parameters'].update(custom_params)
        
        print(f"Processing tool: {config['name']}")
        print(f"Tool type: {config['tool']}")
        
        return config
        
    except ValueError as e:
        st.info(f"Error processing tool request: {e}")
        return None


def run():
    st.title("STELAR Tool")
    token = st.session_state.get("token")

    if not token:
        st.error("Authentication token not found. Please log in.")
        return

    # Fetch data with error handling
    try:
        owner_org = get_organization_data(token)
        df = get_tools(token)
        if df.empty:
            st.warning("No tools available.")
            return
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return

    # Initialize session state
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = df.columns.tolist()

    # Advanced Filters
    with st.expander("🔍 Advanced Filters", expanded=False):
        st.markdown("Apply filters to narrow down the tools.")

        filter_cols = st.multiselect(
            "Select columns to filter", 
            df.columns.tolist(), 
            key="filter_columns_tools"
        )

        filtered_df = apply_filters(df, filter_cols)

        if st.button("Reset Filters", key="reset_filters_btn_tools"):
            reset_filters()
            st.rerun()

    # Column Visibility
    with st.expander("🧩 Column Visibility", expanded=False):
        st.markdown("Select which columns to display.")

        selected_columns = st.multiselect(
            "Columns to display",
            options=df.columns.tolist(),
            default=["name", "author", "organization_name"],
            key="column_selector_tools",
        )

        st.session_state.selected_columns = selected_columns

        if st.button("Reset Columns to Default", key="reset_columns_btn_tools"):
            reset_columns(df.columns.tolist())

    # Apply column selection
    if selected_columns:
        display_df = filtered_df[selected_columns]
    else:
        display_df = filtered_df
        st.warning("No columns selected. Showing all columns.")

    # Display summary stats
    filtered_count = len(display_df)
    total_count = len(df)
    ratio = filtered_count / total_count if total_count > 0 else 0

    # Visual indicator
    circle_symbols = {
        1.0: "⚫", 0.75: "🔵", 0.5: "🟡", 0.25: "🟠"
    }
    circle_symbol = next((symbol for threshold, symbol in circle_symbols.items() if ratio >= threshold), "🔴")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### {circle_symbol} Filtered Tools: {filtered_count}/{total_count}")
        if total_count > 0:
            st.progress(ratio, text=f"{ratio:.1%} of total")

    # Tool selection and processing
    if not display_df.empty:
        st.info("Select a tool by clicking on a row to continue.")

        selected_rows = st.dataframe(
            display_df,
            use_container_width=False,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="process_dataframe_tools",
        )

        selected_indices = selected_rows.selection.rows
        if selected_indices:
            tool_info = get_selected_tool_info(display_df, selected_indices)
            if not tool_info:
                return
                
            tool_name = tool_info['name']
            if tool_name in ['agri-products-match', 'missing-data-interpolation', 'vocational-score-raster']:
                
            
                # Mode selection based on tool type
                x1, x2, x3 = st.columns(3)
                with x1:
                    st.success("Tool selected successfully!")
                
                # Get tool configuration template
                tool_config = get_tool_config_template(tool_name)
                selected_mode = None
                
                # Handle different tools
                if tool_name.lower().replace(" ", "-") == "agri-products-match":
                    selected_mode = st.segmented_control(
                        label="Select Mode",
                        options=["Fertilizers", "Pesticides"],
                        key="mode_selectbox",
                    )

                    if not selected_mode:
                        return

                    # Initialize variables
                    proc_id = proc_name = ds_id = s3_path = tasks = None

                    if selected_mode == "Fertilizers":
                        y1, y2, y3 = st.columns(3)
                        with y1:
                            st.success("You have selected the Fertilizers mode. Continue to select workflow processes and datasets and S3 location.")
                            
                        with st.expander("Show I/O Parameters Graph"):
                            st.image(image="pages/image3.png", caption="Agri Products Match Tool Fertilizers Mode Diagram")
                            
                    elif selected_mode == "Pesticides":
                        y1, y2, y3 = st.columns(3)
                        with y1:
                            st.success("You have selected the Pesticides mode. Continue to select workflow processes and datasets and S3 location.")
                            
                        with st.expander("Show I/O Parameters Graph"):
                            st.info("Pesticides mode processes active substances and matches them with pesticides database.")

                    # Tabs for different selections
                    tab1, tab2, tab3 = st.tabs(["Workflow Processes", "Datasets", "S3 Menu"])

                    with tab1:
                        if "pages.Workflow" in sys.modules:
                            importlib.reload(sys.modules["pages.Workflow"])
                        proc_id, tasks, proc_name = Workflow.run()
                        
                    with tab2:
                        if "pages.Datasets" in sys.modules:
                            importlib.reload(sys.modules["pages.Datasets"])
                        ds_id = Datasets.run()
                        
                    with tab3:
                        if "pages.S3" in sys.modules:
                            importlib.reload(sys.modules["pages.S3"])
                        s3_path = S3.run()

                elif tool_name.lower().replace(" ", "-") == "missing-data-interpolation":
                    selected_mode = "interpolation"
                    st.success("Missing Data Interpolation tool selected. This tool interpolates missing meteorological data.")
                    
                    with st.expander("Show Tool Information"):
                        st.info("This tool takes meteorological file data and coordinates file to interpolate missing values.")
                        if tool_info['inputs']:
                            st.write("### 🛠️ Inputs:")
                            st.dataframe(tool_info['inputs'])
                        if tool_info['outputs']:
                            st.write("### 📦 Outputs:")
                            st.dataframe(tool_info['outputs'])

                    # Tabs for different selections
                    tab1, tab2, tab3 = st.tabs(["Workflow Processes", "Datasets", "S3 Menu"])

                    with tab1:
                        if "pages.Workflow" in sys.modules:
                            importlib.reload(sys.modules["pages.Workflow"])
                        proc_id, tasks, proc_name = Workflow.run()
                        
                    with tab2:
                        if "pages.Datasets" in sys.modules:
                            importlib.reload(sys.modules["pages.Datasets"])
                        ds_id = Datasets.run()
                        
                    with tab3:
                        if "pages.S3" in sys.modules:
                            importlib.reload(sys.modules["pages.S3"])
                        s3_path = S3.run()

                elif tool_name.lower().replace(" ", "-") == "vocational-score-raster":
                    selected_mode = "classification"
                    st.success("VSR (Vocational Score Rating) tool selected. This tool processes raster data for agricultural suitability scoring.")
                    
                    with st.expander("Show Tool Information"):
                        st.info("VSR tool processes raster files and applies classification parameters to generate scored outputs.")
                        if tool_info['inputs']:
                            st.write("### 🛠️ Inputs:")
                            st.dataframe(tool_info['inputs'])
                        if tool_info['outputs']:
                            st.write("### 📦 Outputs:")
                            st.dataframe(tool_info['outputs'])

                    # Tabs for different selections
                    tab1, tab2, tab3 = st.tabs(["Workflow Processes", "Datasets", "S3 Menu"])

                    with tab1:
                        if "pages.Workflow" in sys.modules:
                            importlib.reload(sys.modules["pages.Workflow"])
                        proc_id, tasks, proc_name = Workflow.run()
                        
                    with tab2:
                        if "pages.Datasets" in sys.modules:
                            importlib.reload(sys.modules["pages.Datasets"])
                        ds_id = Datasets.run()
                        
                    with tab3:
                        if "pages.S3" in sys.modules:
                            importlib.reload(sys.modules["pages.S3"])
                        s3_path = S3.run()

                else:
                    # Generic tool handling
                    selected_mode = "default"
                    st.success(f"{tool_name} tool selected.")
                    
                    with st.expander("Show Tool Information"):
                        if tool_info['inputs']:
                            st.write("### 🛠️ Inputs:")
                            st.dataframe(tool_info['inputs'])
                        if tool_info['outputs']:
                            st.write("### 📦 Outputs:")
                            st.dataframe(tool_info['outputs'])
                        if tool_info['parameters']:
                            st.write("### ⚙️ Parameters:")
                            st.dataframe(tool_info['parameters'])

                    # Tabs for different selections
                    tab1, tab2, tab3 = st.tabs(["Workflow Processes", "Datasets", "S3 Menu"])

                    with tab1:
                        if "pages.Workflow" in sys.modules:
                            importlib.reload(sys.modules["pages.Workflow"])
                        proc_id, tasks, proc_name = Workflow.run()
                        
                    with tab2:
                        if "pages.Datasets" in sys.modules:
                            importlib.reload(sys.modules["pages.Datasets"])
                        ds_id = Datasets.run()
                        
                    with tab3:
                        if "pages.S3" in sys.modules:
                            importlib.reload(sys.modules["pages.S3"])
                        s3_path = S3.run()

                # Check selection status and handle task execution
                all_selections_made = render_selection_status(proc_id, ds_id, s3_path)
                
                if all_selections_made:
                    st.success("🎉 All selections complete! You can now execute tasks.")
                    task_execution = st.selectbox(
                        label="Execute Task",
                        options=["Select Task Execution", "Create New Task"],
                        key=f"task_execution_selectbox_{tool_name.lower().replace(' ', '_')}",
                    )
                    
                    # Handle task execution
                    if task_execution and task_execution != "Select Task Execution":
                        handle_task_execution(
                            task_execution, tool_name, selected_mode, proc_id, 
                            proc_name, ds_id, s3_path, tasks, token, owner_org
                        )
                else:
                    st.info("📋 Please complete all selections above to proceed with task execution.")
            else:
                st.warning("Selected tool is not supported yet.")
        else:
            st.info("No tools selected yet.")


# Entry point
if __name__ == "__main__":
    run()