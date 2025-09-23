# STELAR Task Execution GUI

A comprehensive Streamlit-based application for task invocation, design, and workflow management within the STELAR KLMS (Knowledge and Learning Management System).

## Overview

This application provides a user-friendly interface for managing and executing various data processing tasks in the STELAR ecosystem. It supports workflow processes, dataset management, S3 storage integration, and specialized tools for agricultural and meteorological data processing.

## Features

### 🔐 Authentication

- Secure token-based authentication with STELAR API
- Session management for authenticated users

### 📊 Workflow Management

- Browse and filter available workflow processes
- Process selection with detailed task information
- Advanced filtering and column visibility controls
- Real-time process status monitoring

### 📁 Dataset Management

- Paginated dataset browsing with concurrent loading
- Multi-dataset selection capabilities
- Advanced filtering by various data types
- Organization-based dataset categorization

### ☁️ S3 Storage Integration

- MinIO-based S3 storage browser
- Bucket exploration with folder navigation
- File management and path selection
- Secure credential handling

### 🛠️ Specialized Tools

#### Agricultural Products Match Tool

- **Fertilizers Mode**: Matches NPK values with fertilizer datasets
- **Pesticides Mode**: Matches active substances with pesticide databases
- Multi-language support (Italian, English, French, German)

#### Missing Data Interpolation Tool

- Interpolates missing meteorological station data
- Processes meteorological files with coordinate data
- Generates complete datasets for analysis

#### Vocational Score Raster (VSR) Tool

- Agricultural suitability scoring for raster data
- Configurable classification parameters
- Temperature and climate-based scoring

## Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd task-execution-gui
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the application:**

```bash
streamlit run app.py
```

## Dependencies

Key dependencies include:

- `streamlit>=1.45.1` - Web application framework
- `pandas>=2.3.0` - Data manipulation and analysis
- `requests>=2.32.4` - HTTP library for API calls
- `minio>=7.2.15` - S3-compatible object storage client
- `openpyxl>=3.1.5` - Excel file processing

See `requirements.txt` for the complete list of dependencies.

## Configuration

### Environment Variables

- `STELAR_BASE_URL`: Base URL for STELAR API (default: `https://klms.stelar.gr/stelar`)

### Tool Configurations

Tool configurations are defined in `utils/tool_configs.py` and include:

- Process IDs and tool specifications
- Input/output dataset mappings
- Parameter configurations
- S3 storage paths

## Project Structure

```
task-execution-gui/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── utils/                     # Utility modules
│   ├── auth.py               # Authentication handling
│   ├── fetch_datasets.py     # Dataset API operations
│   ├── fetch_process.py      # Process API operations
│   ├── fetch_tools.py        # Tools API operations
│   ├── fetch_organization.py # Organization data
│   ├── fetch_task_by_id.py   # Task details retrieval
│   ├── create_new_dataset.py # Dataset creation
│   ├── s3_utils.py           # S3/MinIO utilities
│   ├── tool_configs.py       # Tool configuration definitions
│   ├── workflow_utils.py     # Workflow utilities
│   └── sample_json.py        # Sample task configurations
└── pages/                    # Application pages
    ├── Workflow.py           # Workflow process management
    ├── Datasets.py           # Dataset browser and selection
    ├── S3.py                 # S3 storage interface
    └── Tools.py              # Tool selection and execution
```

## Usage Guide

### 1. Authentication

- Launch the application and log in with your STELAR credentials
- The system will authenticate and store your session token

### 2. Workflow Process Selection

- Navigate to "Workflow Processes" tab
- Browse available processes with filtering options
- Select a process to view associated tasks
- Process details include execution state and organization info

### 3. Dataset Management

- Use the "Datasets" tab to browse available datasets
- Apply filters to narrow down results
- Select single or multiple datasets as needed
- Navigate through paginated results (up to 3,750 datasets)

### 4. S3 Storage Navigation

- Access the "S3 Menu" to browse storage buckets
- Navigate folder structures with breadcrumb navigation
- Select paths for task output destinations
- View file details and manage storage resources

### 5. Tool Execution

- Go to "Tool Selection" tab and choose from available tools
- Configure tool-specific parameters:
  - **Agri Products Match**: Select mode (fertilizers/pesticides) and language preferences
  - **Missing Data Interpolation**: Configure meteorological data processing
  - **VSR Tool**: Set up raster classification parameters
- Complete the workflow by selecting:
  - Process ID from workflow management
  - Input datasets from dataset browser
  - Output S3 path from storage navigator
- Execute tasks with generated JSON configurations

## API Integration

The application integrates with STELAR KLMS API endpoints:

- `/api/v1/users/token` - Authentication
- `/api/v2/processes.fetch` - Workflow processes
- `/api/v2/datasets.fetch` - Dataset management
- `/api/v2/tools.fetch` - Available tools
- `/api/v2/organizations.fetch` - Organization data
- `/api/v2/task/{id}` - Task details
- `/api/v1/users/s3/credentials` - S3 credentials

## Task Configuration Examples

### Fertilizer Matching Task

```json
{
  "process_id": "f9645b89-34e4-4de2-8ecd-dc10163d9aed",
  "name": "Agri Products Match",
  "tool": "agri-products-match",
  "inputs": {
    "npk_values": ["dataset-id-1"],
    "fertilizer_dataset": ["dataset-id-2"]
  },
  "parameters": { "mode": "fertilizers" },
  "outputs": {
    "matched_fertilizers": {
      "url": "s3://bucket/path/output.csv",
      "dataset": "output-dataset-id"
    }
  }
}
```

### VSR Classification Task

```json
{
  "process_id": "f9645b89-34e4-4de2-8ecd-dc10163d9aed",
  "name": "VSR Classification",
  "tool": "vocational-score-raster",
  "inputs": { "rasters": "d0::owned" },
  "parameters": {
    "Tmax_max_summer_2011_2021.tif": {
      "val_min": 26,
      "val_max": 28.5,
      "new_val": 1
    }
  },
  "outputs": {
    "scored_files": {
      "url": "s3://bucket/VSR/output",
      "dataset": "output-dataset-id"
    }
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is part of the STELAR project ecosystem. Please refer to the project's licensing terms.

## Support

For issues and questions:

- Check the application logs for error details
- Ensure proper authentication credentials
- Verify API endpoint availability
- Contact the STELAR development team for system-specific issues

## Technical Notes

- The application uses caching (`@st.cache_data`) for performance optimization
- Concurrent data loading is implemented for large dataset operations
- Session state management maintains user selections across page navigation
- Error handling includes retry logic for API requests
- The interface supports both single and multi-selection workflows
