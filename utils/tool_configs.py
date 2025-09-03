# utils/tool_configs.py

"""
Tool configuration definitions for various processing tools.
Contains JSON configurations for missing data interpolation, 
vocational score raster, and agricultural products matching.
"""

# Missing Data Interpolation Tool Configuration
MISSING_DATA_INTERPOLATION_CONFIG = {
    "process_id": "f9645b89-34e4-4de2-8ecd-dc10163d9aed",
    "name": "Missing Data Interpolation",
    "tool": "missing-data-interpolation",
    "inputs": {
        "meteo_file": [
            "413f30a3-4653-4b70-8da9-99d7659b23c0"
        ],
        "coords_file": [
            "87668a7e-8d88-4074-a7f2-1a502f40c659"
        ]
    },
    "datasets": {
        "d0": "0fc717e0-5567-4943-a843-9be47aed6eb9"
    },
    "parameters": {},
    "outputs": {
        "interpolated_file": {
            "url": "s3://abaco-bucket/MISSING_DATA/interpolated.xlsx",
            "dataset": "d0",
            "resource": {
                "name": "Interpolated Meteo Station Data",
                "relation": "owned"
            }
        }
    }
}

# Vocational Score Raster Tool Configuration
VOCATIONAL_SCORE_RASTER_CONFIG = {
    "process_id": "f9645b89-34e4-4de2-8ecd-dc10163d9aed",
    "name": "VSR on 1981-2021",
    "tool": "vocational-score-raster",
    "inputs": {
        "rasters": [
            "d0::owned"
        ]
    },
    "datasets": {
        "d0": "16adb665-77ea-410c-8476-132e34160b53",
        "d1": {
            "name": "vsr-output-exp",
            "owner_org": "abaco-group",
            "notes": "The result of VSR execution on Summer and April Dataset VSR Input",
            "tags": ["ABACO", "VSR"]
        }
    },
    "parameters": {
        "Tmax_max_summer_1981_1990.tif": {"val_min": 26, "val_max": 28.5, "new_val": 1},
        "Tmax_max_summer_1991_2000.tif": {"val_min": 26, "val_max": 28.5, "new_val": 1},
        "Tmax_max_summer_2001_2010.tif": {"val_min": 26, "val_max": 28.5, "new_val": 1},
        "Tmax_max_summer_2011_2021.tif": {"val_min": 26, "val_max": 28.5, "new_val": 1},
        "Tmin_min_april_1981_1990.tif": {"val_min": 0.5, "val_max": 8, "new_val": 1},
        "Tmin_min_april_1991_2000.tif": {"val_min": 0.5, "val_max": 8, "new_val": 1},
        "Tmin_min_april_2001_2010.tif": {"val_min": 0.5, "val_max": 8, "new_val": 1},
        "Tmin_min_april_2011_2021.tif": {"val_min": 0.5, "val_max": 8, "new_val": 1}
    },
    "outputs": {
        "scored_files": {
            "url": "s3://abaco-bucket/VOCATIONAL_SCORE/output",
            "dataset": "d1",
            "resource": {
                "name": "Scored Classified rasters via VSR",
                "relation": "raster"
            }
        }
    }
}

# Agricultural Products Match Tool Configuration
AGRI_PRODUCTS_MATCH_CONFIG = {
    "process_id": "f9645b89-34e4-4de2-8ecd-dc10163d9aed",
    "name": "Agri Products Match",
    "tool": "agri-products-match",
    "inputs": {
        "npk_values": [
            "325fb7c7-b269-4a1e-96f6-a861eb2fe325"
        ],
        "fertilizer_dataset": [
            "41da3a81-3768-47db-b7ac-121c92ec3f6d"
        ]
    },
    "datasets": {
        "d0": "2f8a651b-a40b-4edd-b82d-e9ea3aba4d13"
    },
    "parameters": {
        "mode": "fertilizers"
    },
    "outputs": {
        "matched_fertilizers": {
            "url": "s3://abaco-bucket/MATCH/matched_fertilizers.csv",
            "dataset": "d0",
            "resource": {
                "name": "Matched Fertilizers based on NPK values",
                "relation": "matched"
            }
        }
    }
}

# Tool configuration mapping
TOOL_CONFIGS = {
    "missing-data-interpolation": MISSING_DATA_INTERPOLATION_CONFIG,
    "vocational-score-raster": VOCATIONAL_SCORE_RASTER_CONFIG,
    "agri-products-match": AGRI_PRODUCTS_MATCH_CONFIG
}

def get_tool_config(tool_name):
    """
    Returns the JSON configuration object for a given tool name.
    
    Args:
        tool_name (str): The name of the tool to get config for.
                        Valid options: 'missing-data-interpolation', 
                                     'vocational-score-raster', 
                                     'agri-products-match'
    
    Returns:
        dict: The JSON configuration object for the specified tool
    
    Raises:
        ValueError: If the tool name is not recognized
    """
    if tool_name not in TOOL_CONFIGS:
        available_tools = list(TOOL_CONFIGS.keys())
        raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
    
    return TOOL_CONFIGS[tool_name]

# def get_available_tools():
#     """
#     Returns a list of all available tool names.
    
#     Returns:
#         list: List of available tool names
#     """
#     return list(TOOL_CONFIGS.keys())

# def get_all_configs():
#     """
#     Returns all tool configurations.
    
#     Returns:
#         dict: Dictionary containing all tool configurations
#     """
#     return TOOL_CONFIGS.copy()