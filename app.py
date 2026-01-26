"""
DBT Model Generator
Generates DBT transformation SQL, source.yml, and schema.yml from mapping documents.
"""

import os
import io
import zipfile
import streamlit as st
import pandas as pd
import yaml
from groq import Groq
from dotenv import load_dotenv
from config import (
    MODEL_NAME, TEMPERATURE, MAX_TOKENS,
    SUPPORTED_DIALECTS, COLUMN_PATTERNS,
    SQL_SYSTEM_PROMPT, SCHEMA_YAML_SYSTEM_PROMPT,
    get_dialect_prompt
)

load_dotenv()

st.set_page_config(page_title="DBT Model Generator", layout="wide")


# --- File Parsing ---

def parse_file(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """
    Parse uploaded Excel or CSV file into a DataFrame.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Tuple of (DataFrame, None) on success, or (None, error_message) on failure
    """
    filename = uploaded_file.name.lower()

    if not filename.endswith(('.csv', '.xlsx', '.xls')):
        return None, "Unsupported file type. Use .xlsx, .xls, or .csv"

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.lower().str.strip()
        return df, None

    except pd.errors.EmptyDataError:
        return None, "File is empty"
    except pd.errors.ParserError as e:
        return None, f"Parse error: {e}"
    except ValueError as e:
        return None, f"Invalid file format: {e}"
    except Exception as e:
        return None, f"Unexpected error reading file: {e}"


# --- Column Mapping ---

def get_column_mapping(df: pd.DataFrame) -> dict[str, str]:
    """
    Map expected standard column names to actual DataFrame columns.

    Args:
        df: DataFrame with columns to map

    Returns:
        Dictionary mapping standard names to actual column names
    """
    mapping = {}
    df_columns = list(df.columns)

    for std_name, patterns in COLUMN_PATTERNS.items():
        for col in df_columns:
            col_normalized = col.lower().strip()

            is_exact_match = col_normalized in patterns
            is_substring_match = any(p in col_normalized for p in patterns)

            if is_exact_match or is_substring_match:
                mapping[std_name] = col
                break

    return mapping


def get_table_pairs(
    df: pd.DataFrame,
    col_map: dict[str, str]
) -> dict[str, tuple[str, str]]:
    """
    Extract unique source-target table pairs from DataFrame.

    Args:
        df: DataFrame containing table mapping data
        col_map: Column name mapping dictionary

    Returns:
        Dictionary mapping display string to (source, target) tuple
    """
    src_col = col_map.get('source_table')
    tgt_col = col_map.get('target_table')

    if not (src_col and tgt_col):
        return {}

    pairs = df[[src_col, tgt_col]].drop_duplicates()

    return {
        f"{row[src_col]} → {row[tgt_col]}": (row[src_col], row[tgt_col])
        for _, row in pairs.iterrows()
    }


def filter_by_table_pair(
    df: pd.DataFrame,
    col_map: dict[str, str],
    src_table: str,
    tgt_table: str
) -> pd.DataFrame:
    """
    Filter DataFrame for a specific source-target table pair.

    Args:
        df: Full mapping DataFrame
        col_map: Column name mapping dictionary
        src_table: Source table name to filter
        tgt_table: Target table name to filter

    Returns:
        Filtered DataFrame copy
    """
    src_col = col_map.get('source_table')
    tgt_col = col_map.get('target_table')

    if not (src_col and tgt_col):
        return df.copy()

    mask = (df[src_col] == src_table) & (df[tgt_col] == tgt_table)
    return df[mask].copy()


# --- Metadata Extraction ---

def extract_source_columns(
    df: pd.DataFrame,
    col_map: dict[str, str]
) -> list[dict]:
    """
    Extract source column metadata from mapping DataFrame.

    Args:
        df: Filtered mapping DataFrame
        col_map: Column name mapping dictionary

    Returns:
        List of dicts with 'name' and 'datatype' keys
    """
    columns = []
    seen = set()

    for _, row in df.iterrows():
        col_name = str(row.get(col_map.get('source_column', ''), '')).strip()
        col_type = str(row.get(col_map.get('source_datatype', ''), '')).strip()

        if col_name and col_name not in seen:
            seen.add(col_name)
            columns.append({
                'name': col_name,
                'datatype': col_type if col_type else 'VARCHAR'
            })

    return columns


def extract_target_columns(
    df: pd.DataFrame,
    col_map: dict[str, str]
) -> list[dict]:
    """
    Extract target column metadata from mapping DataFrame.

    Args:
        df: Filtered mapping DataFrame
        col_map: Column name mapping dictionary

    Returns:
        List of dicts with 'name', 'datatype', and 'logic' keys
    """
    columns = []
    seen = set()

    for _, row in df.iterrows():
        col_name = str(row.get(col_map.get('target_column', ''), '')).strip()
        col_type = str(row.get(col_map.get('target_datatype', ''), '')).strip()
        logic = str(row.get(col_map.get('transformation_logic', ''), '')).strip()

        if col_name and col_name not in seen:
            seen.add(col_name)
            columns.append({
                'name': col_name,
                'datatype': col_type if col_type else 'VARCHAR',
                'logic': logic if logic else 'direct move'
            })

    return columns


# --- SQL Generation ---

def format_type_info(src_type: str, tgt_type: str) -> str:
    """Format source and target data types into a display string."""
    types = [src_type, tgt_type]

    if not any(types):
        return ""

    formatted = ' → '.join(t for t in types if t)
    return f" ({formatted})"


def build_mapping_text(df: pd.DataFrame, col_map: dict[str, str]) -> str:
    """Format mapping DataFrame as text for LLM prompt."""
    lines = []

    for _, row in df.iterrows():
        src_col = row.get(col_map.get('source_column', ''), 'N/A')
        tgt_col = row.get(col_map.get('target_column', ''), 'N/A')
        src_type = row.get(col_map.get('source_datatype', ''), '') or ''
        tgt_type = row.get(col_map.get('target_datatype', ''), '') or ''
        logic = row.get(col_map.get('transformation_logic', ''), 'direct move')

        type_info = format_type_info(str(src_type), str(tgt_type))
        line = f"- {src_col}{type_info} → {tgt_col}: {logic}"
        lines.append(line)

    return "\n".join(lines)


def build_sql_user_prompt(
    df: pd.DataFrame,
    col_map: dict[str, str],
    src_table: str,
    tgt_table: str,
    schema: str
) -> str:
    """Build the user prompt for SQL generation."""
    mapping_text = build_mapping_text(df, col_map)

    return f"""<request>
Generate a DBT SQL model for the following transformation.
</request>

<source_info>
SOURCE TABLE: {src_table}
TARGET TABLE: {tgt_table}
SCHEMA: {schema}
</source_info>

<column_mappings>
{mapping_text}
</column_mappings>

<requirements>
1. Use {{{{ source('{schema}', '{src_table}') }}}} for the source table reference
2. Apply all transformation logic exactly as specified
3. Return NULL for any value that fails validation
4. Add row-level filters to the WHERE clause
5. Return only the SQL code
</requirements>

Generate the complete DBT model SQL now:"""


# --- Source YAML Generation (Python-based) ---

def build_source_yaml(
    schema_name: str,
    database_name: str | None,
    table_name: str,
    columns: list[dict]
) -> str:
    """
    Build source.yml content using Python (deterministic, no LLM calls).

    Args:
        schema_name: Name of the schema/source
        database_name: Optional database name
        table_name: Name of the source table
        columns: List of column dicts with 'name' and 'datatype'

    Returns:
        Formatted YAML string
    """
    source_def = {
        'name': schema_name,
        'description': f"Source tables from {schema_name}",
        'tables': [{
            'name': table_name,
            'description': f"Raw {table_name} data",
            'columns': [
                {
                    'name': col['name'],
                    'description': f"{col['name']} ({col['datatype']})"
                }
                for col in columns
            ]
        }]
    }

    if database_name and database_name.strip():
        source_def['database'] = database_name.strip()

    source_yaml = {
        'version': 2,
        'sources': [source_def]
    }

    return yaml.dump(
        source_yaml,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=80
    )


# --- Schema YAML Generation (LLM-based) ---

def build_schema_yaml_prompt(model_name: str, columns: list[dict]) -> str:
    """
    Build the user prompt for schema.yml generation.

    Args:
        model_name: Name of the DBT model
        columns: List of column dicts with 'name', 'datatype', 'logic'

    Returns:
        Formatted prompt string
    """
    column_lines = []
    for col in columns:
        line = f"- {col['name']} ({col['datatype']}): {col['logic']}"
        column_lines.append(line)

    columns_text = "\n".join(column_lines)

    return f"""<request>
Generate a DBT schema.yml file with appropriate data tests for this model.
</request>

<model_info>
Model Name: {model_name}
</model_info>

<columns>
{columns_text}
</columns>

<requirements>
1. Infer appropriate tests based on column names and transformation logic
2. Include meaningful descriptions for each column
3. Use data_tests (not tests) for test definitions
4. Return only valid YAML, no markdown fences or explanations
</requirements>

Generate the complete schema.yml now:"""


def generate_schema_yaml(
    model_name: str,
    columns: list[dict]
) -> tuple[str | None, str | None]:
    """
    Generate schema.yml using LLM.

    Args:
        model_name: Name of the DBT model
        columns: List of column dicts with metadata

    Returns:
        Tuple of (yaml_content, None) on success, or (None, error) on failure
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None, "GROQ_API_KEY not found in .env file"

    try:
        client = Groq(api_key=api_key)

        user_prompt = build_schema_yaml_prompt(model_name, columns)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SCHEMA_YAML_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        raw_output = response.choices[0].message.content
        yaml_content = strip_yaml_fences(raw_output)

        is_valid, error = validate_yaml(yaml_content)
        if not is_valid:
            return None, f"Generated invalid YAML: {error}"

        return yaml_content, None

    except Exception as e:
        return None, f"API Error: {e}"


# --- Utilities ---

def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output (SQL)."""
    text = text.strip()

    if text.startswith("```sql"):
        text = text[6:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def strip_yaml_fences(text: str) -> str:
    """Remove markdown code fences from LLM output (YAML)."""
    text = text.strip()

    if text.startswith("```yaml"):
        text = text[7:]
    elif text.startswith("```yml"):
        text = text[6:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def validate_yaml(content: str) -> tuple[bool, str | None]:
    """
    Validate YAML syntax.

    Args:
        content: YAML string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        yaml.safe_load(content)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)


def call_llm(system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
    """
    Call Groq API for SQL generation.

    Args:
        system_prompt: System message for LLM context
        user_prompt: User message with mapping details

    Returns:
        Tuple of (content, None) on success, or (None, error) on failure
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None, "GROQ_API_KEY not found in .env file"

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        raw_output = response.choices[0].message.content
        return strip_markdown_fences(raw_output), None

    except Exception as e:
        return None, f"API Error: {e}"


def create_download_bundle(artifacts: dict, base_name: str) -> bytes:
    """
    Create a ZIP file containing all generated artifacts.

    Args:
        artifacts: Dict with 'sql', 'source_yml', 'schema_yml' keys
        base_name: Base name for files (typically target table name)

    Returns:
        ZIP file as bytes
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        if artifacts.get('sql'):
            zf.writestr(f"{base_name}.sql", artifacts['sql'])

        if artifacts.get('source_yml'):
            zf.writestr("source.yml", artifacts['source_yml'])

        if artifacts.get('schema_yml'):
            zf.writestr("schema.yml", artifacts['schema_yml'])

    buffer.seek(0)
    return buffer.getvalue()


# --- UI Components ---

def render_sidebar() -> tuple[str, str, str | None, dict]:
    """
    Render sidebar with configuration options.

    Returns:
        Tuple of (dialect, schema_name, database_name, artifact_options)
    """
    with st.sidebar:
        st.header("Configuration")

        dialect = st.selectbox(
            "SQL Dialect",
            options=SUPPORTED_DIALECTS,
            help="Select target SQL dialect"
        )

        st.divider()

        schema_name = st.text_input(
            "Schema Name",
            value="source_schema",
            help="Schema name for DBT source() reference"
        )

        database_name = st.text_input(
            "Database Name (optional)",
            value="",
            help="Database name for source.yml"
        )

        st.divider()

        st.subheader("Artifacts to Generate")

        artifact_options = {
            'sql': st.checkbox("SQL Model", value=True, help="Generate DBT SQL model"),
            'source_yml': st.checkbox("source.yml", value=True, help="Generate source definition"),
            'schema_yml': st.checkbox("schema.yml", value=True, help="Generate schema with tests")
        }

        st.divider()

        with st.expander("How to use"):
            st.markdown("""
            1. **Upload** your mapping Excel/CSV file
            2. **Configure** dialect and schema name
            3. **Select** artifacts to generate
            4. **Choose** source-target table pair
            5. **Click** Generate to create artifacts
            6. **Download** individual files or ZIP bundle
            """)

        with st.expander("Expected File Format"):
            st.markdown("""
            **Required columns:**
            - Source Table
            - Source Column
            - Target Table
            - Target Column
            - Transformation Logic

            **Optional columns:**
            - Source Column Datatype
            - Target Column Datatype
            """)

    return dialect, schema_name, database_name, artifact_options


def render_artifact_tabs(artifacts: dict) -> None:
    """
    Render artifacts in a tabbed interface.

    Args:
        artifacts: Dict with 'sql', 'source_yml', 'schema_yml' keys
    """
    available_tabs = []
    tab_contents = []

    if artifacts.get('sql'):
        available_tabs.append("SQL Model")
        tab_contents.append(('sql', artifacts['sql']))

    if artifacts.get('source_yml'):
        available_tabs.append("source.yml")
        tab_contents.append(('yaml', artifacts['source_yml']))

    if artifacts.get('schema_yml'):
        available_tabs.append("schema.yml")
        tab_contents.append(('yaml', artifacts['schema_yml']))

    if not available_tabs:
        st.warning("No artifacts generated yet.")
        return

    tabs = st.tabs(available_tabs)

    for tab, (lang, content) in zip(tabs, tab_contents):
        with tab:
            st.code(content, language=lang)


# --- Main Application ---

def main():
    """Main application entry point."""

    if 'artifacts' not in st.session_state:
        st.session_state.artifacts = {
            'sql': None,
            'source_yml': None,
            'schema_yml': None
        }

    if 'generation_complete' not in st.session_state:
        st.session_state.generation_complete = False

    st.title("DBT Model Generator")
    st.markdown("Generate DBT transformation SQL, source.yml, and schema.yml from mapping documents")
    st.divider()

    dialect, schema_name, database_name, artifact_options = render_sidebar()

    st.subheader("Upload Mapping File")
    uploaded_file = st.file_uploader(
        "Choose your mapping file",
        type=["xlsx", "xls", "csv"],
        help="Upload Excel or CSV file with column mappings"
    )

    if not uploaded_file:
        st.info("👆 Upload a mapping file to get started")
        return

    df, error = parse_file(uploaded_file)

    if error:
        st.error(error)
        return

    col_map = get_column_mapping(df)

    st.subheader("Mapping Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("Select Table Pair")
    table_pairs = get_table_pairs(df, col_map)

    if not table_pairs:
        st.warning("Could not detect source/target tables. Check column names.")
        return

    selected = st.selectbox("Source → Target Table", options=list(table_pairs.keys()))
    src_table, tgt_table = table_pairs[selected]

    st.divider()

    st.subheader("Generate Artifacts")

    any_selected = any(artifact_options.values())

    if not any_selected:
        st.warning("Select at least one artifact to generate in the sidebar.")

    if st.button("Generate Artifacts", type="primary", disabled=not any_selected):
        filtered_df = filter_by_table_pair(df, col_map, src_table, tgt_table)
        source_columns = extract_source_columns(filtered_df, col_map)
        target_columns = extract_target_columns(filtered_df, col_map)

        st.session_state.artifacts = {
            'sql': None,
            'source_yml': None,
            'schema_yml': None
        }

        errors = []

        progress_bar = st.progress(0, text="Starting generation...")
        total_steps = sum(artifact_options.values())
        current_step = 0

        if artifact_options['sql']:
            progress_bar.progress(
                current_step / total_steps,
                text="Generating SQL model..."
            )

            system_prompt = SQL_SYSTEM_PROMPT + get_dialect_prompt(dialect)
            user_prompt = build_sql_user_prompt(
                filtered_df, col_map, src_table, tgt_table, schema_name
            )

            sql, err = call_llm(system_prompt, user_prompt)

            if err:
                errors.append(f"SQL: {err}")
            else:
                st.session_state.artifacts['sql'] = sql

            current_step += 1

        if artifact_options['source_yml']:
            progress_bar.progress(
                current_step / total_steps,
                text="Building source.yml..."
            )

            source_yml = build_source_yaml(
                schema_name,
                database_name,
                src_table,
                source_columns
            )
            st.session_state.artifacts['source_yml'] = source_yml
            current_step += 1

        if artifact_options['schema_yml']:
            progress_bar.progress(
                current_step / total_steps,
                text="Generating schema.yml with tests..."
            )

            schema_yml, err = generate_schema_yaml(tgt_table, target_columns)

            if err:
                errors.append(f"Schema: {err}")
            else:
                st.session_state.artifacts['schema_yml'] = schema_yml

            current_step += 1

        progress_bar.progress(1.0, text="Complete!")

        if errors:
            for err in errors:
                st.error(err)

        successful = sum(1 for v in st.session_state.artifacts.values() if v)

        if successful > 0:
            st.success(f"Generated {successful} artifact(s) successfully!")
            st.session_state.generation_complete = True

    if st.session_state.generation_complete and any(st.session_state.artifacts.values()):
        st.divider()
        st.subheader("Generated Artifacts")

        render_artifact_tabs(st.session_state.artifacts)

        st.divider()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.session_state.artifacts.get('sql'):
                st.download_button(
                    label="⬇️ Download SQL",
                    data=st.session_state.artifacts['sql'],
                    file_name=f"{tgt_table.lower()}.sql",
                    mime="text/plain"
                )

        with col2:
            if st.session_state.artifacts.get('source_yml'):
                st.download_button(
                    label="⬇️ Download source.yml",
                    data=st.session_state.artifacts['source_yml'],
                    file_name="source.yml",
                    mime="text/yaml"
                )

        with col3:
            if st.session_state.artifacts.get('schema_yml'):
                st.download_button(
                    label="⬇️ Download schema.yml",
                    data=st.session_state.artifacts['schema_yml'],
                    file_name="schema.yml",
                    mime="text/yaml"
                )

        with col4:
            artifacts_count = sum(
                1 for v in st.session_state.artifacts.values() if v
            )

            if artifacts_count > 1:
                zip_data = create_download_bundle(
                    st.session_state.artifacts,
                    tgt_table.lower()
                )

                st.download_button(
                    label="⬇️ Download All (ZIP)",
                    data=zip_data,
                    file_name=f"{tgt_table.lower()}_dbt_artifacts.zip",
                    mime="application/zip"
                )


if __name__ == "__main__":
    main()