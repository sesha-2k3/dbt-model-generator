# DBT Model Generator

An AI agent-based system that automatically generates DBT transformation code from source-to-target column mapping documents, supporting Snowflake and Databricks cloud data warehouses.

## Overview

This tool converts Excel/CSV mapping specifications into production-ready DBT artifacts:
- **SQL Models** — Transformation logic with dialect-specific syntax
- **source.yml** — Source table configurations
- **schema.yml** — Data quality tests and column descriptions

## Features

- **LLM-Powered Generation** — Uses Llama 3.3-70B via Groq API for intelligent SQL generation
- **20+ Transformation Patterns** — Regex validation, type casting, value mapping, row filtering, and more
- **Multi-Dialect Support** — Snowflake and PostgreSQL (Databricks) syntax
- **Ad-hoc Prompt Modifications** — Runtime logic customization via natural language
- **Inferred Data Quality Tests** — Automatic test generation based on column patterns
- **Batch Download** — Export all artifacts as a ZIP bundle


## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dbt-model-generator.git
cd dbt-model-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

```bash
streamlit run app.py
```

1. **Upload** your mapping file (Excel/CSV)
2. **Configure** SQL dialect and schema name
3. **Select** source → target table pair
4. **Add** additional instructions (optional)
5. **Generate** and download artifacts

## Supported Transformations

| Pattern | Example |
|---------|---------|
| Direct move | `direct move` |
| Regex validation | `validate must have only letters` |
| Type conversion | `convert to date format YYYY-MM-DD` |
| Value mapping | `map 'A' to 'Active', 'I' to 'Inactive'` |
| Concatenation | `concatenate first_name and last_name` |
| Row filtering | `consider only rows where status = 'Active'` |
| String operations | `trim whitespace`, `convert to uppercase` |
| Range validation | `validate must be between 0 and 100` |

## Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
groq>=0.4.0
pyyaml>=6.0
python-dotenv>=1.0.0
openpyxl>=3.1.0
```
