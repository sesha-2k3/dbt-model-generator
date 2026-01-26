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

    Returns:
        Tuple of (DataFrame, None) on success, or (None, error_message) on failure
    """
    filename = uploaded_file.name.lower()

    if not filename.endswith(('.csv', '.xlsx', '.xls')):
        return None, "Unsupported file type. Use .xlsx, .xls, or .csv"

    try:
        df = pd.read_csv(uploaded_file) if filename.endswith('.csv') else pd.read_excel(uploaded_file)
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
    """Map expected standard column names to actual DataFrame columns."""
    mapping = {}

    for std_name, patterns in COLUMN_PATTERNS.items():
        for col in df.columns:
            col_normalized = col.lower().strip()
            if col_normalized in patterns or any(p in col_normalized for p in patterns):
                mapping[std_name] = col
                break

    return mapping


def get_table_pairs(df: pd.DataFrame, col_map: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Extract unique source-target table pairs from DataFrame."""
    src_col, tgt_col = col_map.get('source_table'), col_map.get('target_table')

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
    """Filter DataFrame for a specific source-target table pair."""
    src_col, tgt_col = col_map.get('source_table'), col_map.get('target_table')

    if not (src_col and tgt_col):
        return df.copy()

    mask = (df[src_col] == src_table) & (df[tgt_col] == tgt_table)
    return df[mask].copy()


# --- Metadata Extraction ---

def extract_columns(
    df: pd.DataFrame,
    col_map: dict[str, str],
    column_type: str = 'source'
) -> list[dict]:
    """
    Extract column metadata from mapping DataFrame.

    Args:
        df: Filtered mapping DataFrame
        col_map: Column name mapping dictionary
        column_type: 'source' or 'target'

    Returns:
        List of column dicts. Target includes 'logic' key.
    """
    columns = []
    seen = set()

    col_key = f'{column_type}_column'
    type_key = f'{column_type}_datatype'

    for _, row in df.iterrows():
        col_name = str(row.get(col_map.get(col_key, ''), '')).strip()
        col_dtype = str(row.get(col_map.get(type_key, ''), '')).strip() or 'VARCHAR'

        if col_name and col_name not in seen:
            seen.add(col_name)
            col_info = {'name': col_name, 'datatype': col_dtype}

            if column_type == 'target':
                logic = str(row.get(col_map.get('transformation_logic', ''), '')).strip()
                col_info['logic'] = logic or 'direct move'

            columns.append(col_info)

    return columns


# --- SQL Generation ---

def build_mapping_text(df: pd.DataFrame, col_map: dict[str, str]) -> str:
    """Format mapping DataFrame as text for LLM prompt."""
    lines = []

    for _, row in df.iterrows():
        src_col = row.get(col_map.get('source_column', ''), 'N/A')
        tgt_col = row.get(col_map.get('target_column', ''), 'N/A')
        src_type = str(row.get(col_map.get('source_datatype', ''), '') or '')
        tgt_type = str(row.get(col_map.get('target_datatype', ''), '') or '')
        logic = row.get(col_map.get('transformation_logic', ''), 'direct move')

        # Inline type formatting
        types = [t for t in [src_type, tgt_type] if t]
        type_info = f" ({' → '.join(types)})" if types else ""

        lines.append(f"- {src_col}{type_info} → {tgt_col}: {logic}")

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
    """Build source.yml content using Python (deterministic, no LLM)."""
    source_def = {
        'name': schema_name,
        'description': f"Source tables from {schema_name}",
        'tables': [{
            'name': table_name,
            'description': f"Raw {table_name} data",
            'columns': [
                {'name': col['name'], 'description': f"{col['name']} ({col['datatype']})"}
                for col in columns
            ]
        }]
    }

    if database_name and database_name.strip():
        source_def['database'] = database_name.strip()

    return yaml.dump(
        {'version': 2, 'sources': [source_def]},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=80
    )


# --- Schema YAML Generation (LLM-based) ---

def build_schema_yaml_prompt(model_name: str, columns: list[dict]) -> str:
    """Build the user prompt for schema.yml generation."""
    columns_text = "\n".join(
        f"- {col['name']} ({col['datatype']}): {col['logic']}"
        for col in columns
    )

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


# --- Utilities ---

def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()

    for prefix in ('```sql', '```yaml', '```yml', '```'):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    if text.endswith('```'):
        text = text[:-3]

    return text.strip()


def validate_yaml(content: str) -> tuple[bool, str | None]:
    """Validate YAML syntax. Returns (is_valid, error_message)."""
    try:
        yaml.safe_load(content)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    validate_as_yaml: bool = False
) -> tuple[str | None, str | None]:
    """
    Call Groq API for content generation.

    Args:
        system_prompt: System message for LLM context
        user_prompt: User message with details
        validate_as_yaml: If True, validate output as YAML

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

        content = strip_code_fences(response.choices[0].message.content)

        if validate_as_yaml:
            is_valid, error = validate_yaml(content)
            if not is_valid:
                return None, f"Generated invalid YAML: {error}"

        return content, None

    except Exception as e:
        return None, f"API Error: {e}"


def create_download_bundle(artifacts: dict, base_name: str) -> bytes:
    """Create a ZIP file containing all generated artifacts."""
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
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.header("Configuration")

        dialect = st.selectbox("SQL Dialect", options=SUPPORTED_DIALECTS)

        st.divider()

        schema_name = st.text_input("Schema Name", value="source_schema")
        database_name = st.text_input("Database Name (optional)", value="")

        st.divider()

        st.subheader("Artifacts")
        artifact_options = {
            'sql': st.checkbox("SQL Model", value=True),
            'source_yml': st.checkbox("source.yml", value=True),
            'schema_yml': st.checkbox("schema.yml", value=True)
        }

        st.divider()

        with st.expander("How to use"):
            st.markdown("""
            1. **Upload** mapping Excel/CSV file
            2. **Configure** dialect and schema
            3. **Select** artifacts to generate
            4. **Choose** table pair
            5. **Generate** and download
            """)

        with st.expander("File Format"):
            st.markdown("""
            **Required:** Source Table, Source Column, Target Table, Target Column, Transformation Logic

            **Optional:** Source/Target Column Datatype
            """)

    return dialect, schema_name, database_name, artifact_options


def render_artifact_tabs(artifacts: dict) -> None:
    """Render artifacts in a tabbed interface."""
    tab_config = [
        ('sql', 'SQL Model', 'sql'),
        ('source_yml', 'source.yml', 'yaml'),
        ('schema_yml', 'schema.yml', 'yaml')
    ]

    available = [(key, label, lang) for key, label, lang in tab_config if artifacts.get(key)]

    if not available:
        st.warning("No artifacts generated yet.")
        return

    tabs = st.tabs([label for _, label, _ in available])

    for tab, (key, _, lang) in zip(tabs, available):
        with tab:
            st.code(artifacts[key], language=lang)


# --- Main Application ---

def main():
    """Main application entry point."""

    if 'artifacts' not in st.session_state:
        st.session_state.artifacts = {'sql': None, 'source_yml': None, 'schema_yml': None}

    st.title("🔧 DBT Model Generator")
    st.markdown("Generate DBT transformation SQL, source.yml, and schema.yml from mapping documents")
    st.divider()

    dialect, schema_name, database_name, artifact_options = render_sidebar()

    st.subheader("Upload Mapping File")
    uploaded_file = st.file_uploader(
        "Choose your mapping file",
        type=["xlsx", "xls", "csv"]
    )

    if not uploaded_file:
        st.info("Upload a mapping file to get started")
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
        st.warning("Select at least one artifact in the sidebar.")

    if st.button("Generate Artifacts", type="primary", disabled=not any_selected):
        filtered_df = filter_by_table_pair(df, col_map, src_table, tgt_table)
        source_columns = extract_columns(filtered_df, col_map, 'source')
        target_columns = extract_columns(filtered_df, col_map, 'target')

        st.session_state.artifacts = {'sql': None, 'source_yml': None, 'schema_yml': None}
        errors = []

        total_steps = sum(artifact_options.values())
        progress = st.progress(0, text="Starting...")

        step = 0

        if artifact_options['sql']:
            progress.progress(step / total_steps, text="Generating SQL model...")
            sql, err = call_llm(
                SQL_SYSTEM_PROMPT + get_dialect_prompt(dialect),
                build_sql_user_prompt(filtered_df, col_map, src_table, tgt_table, schema_name)
            )
            if err:
                errors.append(f"SQL: {err}")
            else:
                st.session_state.artifacts['sql'] = sql
            step += 1

        if artifact_options['source_yml']:
            progress.progress(step / total_steps, text="Building source.yml...")
            st.session_state.artifacts['source_yml'] = build_source_yaml(
                schema_name, database_name, src_table, source_columns
            )
            step += 1

        if artifact_options['schema_yml']:
            progress.progress(step / total_steps, text="Generating schema.yml...")
            schema_yml, err = call_llm(
                SCHEMA_YAML_SYSTEM_PROMPT,
                build_schema_yaml_prompt(tgt_table, target_columns),
                validate_as_yaml=True
            )
            if err:
                errors.append(f"Schema: {err}")
            else:
                st.session_state.artifacts['schema_yml'] = schema_yml
            step += 1

        progress.progress(1.0, text="Complete!")

        for err in errors:
            st.error(err)

        successful = sum(1 for v in st.session_state.artifacts.values() if v)
        if successful:
            st.success(f"Generated {successful} artifact(s) successfully!")

    # Display artifacts if any exist
    if any(st.session_state.artifacts.values()):
        st.divider()
        st.subheader("Generated Artifacts")
        render_artifact_tabs(st.session_state.artifacts)

        st.divider()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.session_state.artifacts.get('sql'):
                st.download_button("⬇️ SQL", st.session_state.artifacts['sql'],
                                   f"{tgt_table.lower()}.sql", "text/plain")

        with col2:
            if st.session_state.artifacts.get('source_yml'):
                st.download_button("⬇️ source.yml", st.session_state.artifacts['source_yml'],
                                   "source.yml", "text/yaml")

        with col3:
            if st.session_state.artifacts.get('schema_yml'):
                st.download_button("⬇️ schema.yml", st.session_state.artifacts['schema_yml'],
                                   "schema.yml", "text/yaml")

        with col4:
            if sum(1 for v in st.session_state.artifacts.values() if v) > 1:
                st.download_button(
                    "⬇️ All (ZIP)",
                    create_download_bundle(st.session_state.artifacts, tgt_table.lower()),
                    f"{tgt_table.lower()}_dbt.zip",
                    "application/zip"
                )


if __name__ == "__main__":
    main()