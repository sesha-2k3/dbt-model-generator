"""
DBT Model Generator
Generates DBT transformation SQL, sources.yml, and schema.yml from mapping documents.
"""

import os
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from config import (
    MODEL_NAME, TEMPERATURE, MAX_TOKENS,
    SUPPORTED_DIALECTS, COLUMN_PATTERNS,
    SQL_MODEL_SYSTEM_PROMPT, SOURCES_SYSTEM_PROMPT, SCHEMA_SYSTEM_PROMPT,
    get_dialect_prompt
)

load_dotenv()

st.set_page_config(page_title="DBT Model Generator", layout="wide")


# --- Helper Functions ---

def parse_file(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """Parse uploaded Excel or CSV file into a DataFrame."""
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


def get_column_mapping(df: pd.DataFrame) -> dict[str, str]:
    """Map expected standard column names to actual DataFrame columns."""
    mapping = {}
    df_columns = list(df.columns)
    
    for std_name, patterns in COLUMN_PATTERNS.items():
        for col in df_columns:
            col_normalized = col.lower().strip()
            is_exact = col_normalized in patterns
            is_substring = any(p in col_normalized for p in patterns)
            
            if is_exact or is_substring:
                mapping[std_name] = col
                break
    
    return mapping


def get_table_pairs(df: pd.DataFrame, col_map: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Extract unique source-target table pairs from DataFrame."""
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
    df: pd.DataFrame, col_map: dict[str, str], src_table: str, tgt_table: str
) -> pd.DataFrame:
    """Filter DataFrame for a specific source-target table pair."""
    src_col = col_map.get('source_table')
    tgt_col = col_map.get('target_table')
    
    if not (src_col and tgt_col):
        return df.copy()
    
    mask = (df[src_col] == src_table) & (df[tgt_col] == tgt_table)
    return df[mask].copy()


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
        lines.append(f"- {src_col}{type_info} → {tgt_col}: {logic}")
    
    return "\n".join(lines)


def extract_columns_info(df: pd.DataFrame, col_map: dict[str, str], is_source: bool = True) -> str:
    """Extract column info for sources.yml or schema.yml prompts."""
    if is_source:
        col_key, type_key = 'source_column', 'source_datatype'
    else:
        col_key, type_key = 'target_column', 'target_datatype'
    
    col_name = col_map.get(col_key, '')
    type_name = col_map.get(type_key, '')
    logic_name = col_map.get('transformation_logic', '')
    
    lines = []
    seen = set()
    
    for _, row in df.iterrows():
        col = row.get(col_name, 'N/A')
        if col in seen:
            continue
        seen.add(col)
        
        dtype = row.get(type_name, '') or 'VARCHAR'
        logic = row.get(logic_name, '') or 'direct move' if not is_source else ''
        
        if is_source:
            lines.append(f"- {col} ({dtype})")
        else:
            lines.append(f"- {col} ({dtype}): {logic}")
    
    return "\n".join(lines)


# --- Prompt Builders ---

def build_sql_model_prompt(
    df: pd.DataFrame, col_map: dict[str, str],
    src_table: str, tgt_table: str, schema: str
) -> str:
    """Build user prompt for SQL model generation."""
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


def build_sources_prompt(
    df: pd.DataFrame, col_map: dict[str, str], src_table: str, schema: str
) -> str:
    """Build user prompt for sources.yml generation."""
    columns_info = extract_columns_info(df, col_map, is_source=True)
    
    return f"""<request>
Generate a DBT sources.yml file for the following source table.
</request>

<source_info>
SOURCE TABLE: {src_table}
SCHEMA: {schema}
</source_info>

<source_columns>
{columns_info}
</source_columns>

<requirements>
1. Use version: 2 format
2. Use '{schema}' as the source name
3. Include meaningful descriptions for the table and each column
4. Return only valid YAML content, no markdown fences
</requirements>

Generate the sources.yml content now:"""


def build_schema_prompt(
    df: pd.DataFrame, col_map: dict[str, str], tgt_table: str
) -> str:
    """Build user prompt for schema.yml generation."""
    columns_info = extract_columns_info(df, col_map, is_source=False)
    
    return f"""<request>
Generate a DBT schema.yml file for the following target model.
</request>

<model_info>
MODEL NAME: {tgt_table}
</model_info>

<target_columns>
{columns_info}
</target_columns>

<requirements>
1. Use version: 2 format
2. Include meaningful descriptions reflecting transformations
3. Add appropriate tests (unique, not_null, accepted_values) where sensible
4. Return only valid YAML content, no markdown fences
</requirements>

Generate the schema.yml content now:"""


# --- LLM Functions ---

def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    
    for fence in ["```sql", "```yaml", "```yml", "```"]:
        if text.startswith(fence):
            text = text[len(fence):]
            break
    
    if text.endswith("```"):
        text = text[:-3]
    
    return text.strip()


def call_llm(system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
    """Call Groq API to generate content."""
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
        
        raw = response.choices[0].message.content
        return strip_markdown_fences(raw), None
        
    except Exception as e:
        return None, f"API Error: {e}"


def generate_dbt_artifacts(
    df: pd.DataFrame, col_map: dict[str, str],
    src_table: str, tgt_table: str, schema: str, dialect: str
) -> dict[str, tuple[str | None, str | None]]:
    """
    Generate SQL model, sources.yml, and schema.yml in parallel.
    
    Returns:
        Dictionary with keys 'sql', 'sources', 'schema', each containing
        a tuple of (content, error).
    """
    sql_system = SQL_MODEL_SYSTEM_PROMPT + get_dialect_prompt(dialect)
    sql_user = build_sql_model_prompt(df, col_map, src_table, tgt_table, schema)
    
    sources_system = SOURCES_SYSTEM_PROMPT
    sources_user = build_sources_prompt(df, col_map, src_table, schema)
    
    schema_system = SCHEMA_SYSTEM_PROMPT
    schema_user = build_schema_prompt(df, col_map, tgt_table)
    
    tasks = {
        'sql': (sql_system, sql_user),
        'sources': (sources_system, sources_user),
        'schema': (schema_system, schema_user)
    }
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_key = {
            executor.submit(call_llm, sys_p, usr_p): key
            for key, (sys_p, usr_p) in tasks.items()
        }
        
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                content, error = future.result()
                results[key] = (content, error)
            except Exception as e:
                results[key] = (None, f"Execution error: {e}")
    
    return results


# --- UI Functions ---

def render_sidebar() -> tuple[str, str]:
    """Render sidebar with configuration options."""
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
        
        st.divider()
        
        with st.expander("How to use"):
            st.markdown("""
            1. **Upload** your mapping Excel/CSV file
            2. **Select** the SQL dialect
            3. **Choose** source-target table pair
            4. **Click** Generate to create DBT artifacts
            5. **Download** the generated files
            """)
        
        with st.expander("Expected File Format"):
            st.markdown("""
            Required columns:
            - Source Table
            - Source Column
            - Target Table
            - Target Column
            - Transformation Logic
            
            Optional columns:
            - Source Column Datatype
            - Target Column Datatype
            """)
    
    return dialect, schema_name


def create_zip_file(results: dict, tgt_table: str) -> bytes | None:
    """
    Create a zip file containing all generated artifacts.
    
    Returns:
        Bytes of the zip file, or None if no content to zip.
    """
    files = {
        'sql': (f"{tgt_table.lower()}.sql", results.get('sql', (None, None))[0]),
        'sources': ("sources.yml", results.get('sources', (None, None))[0]),
        'schema': ("schema.yml", results.get('schema', (None, None))[0])
    }
    
    valid_files = {k: v for k, v in files.items() if v[1] is not None}
    
    if not valid_files:
        return None
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for _, (filename, content) in valid_files.items():
            zf.writestr(filename, content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def display_results(results: dict, tgt_table: str):
    """Display generated artifacts in tabs with a single zip download button."""
    
    zip_data = create_zip_file(results, tgt_table)
    if zip_data:
        st.download_button(
            label="📦 Download All as ZIP",
            data=zip_data,
            file_name=f"{tgt_table.lower()}_dbt_artifacts.zip",
            mime="application/zip",
            key="download_zip"
        )
    
    tab_sql, tab_sources, tab_schema = st.tabs(["SQL Model", "sources.yml", "schema.yml"])
    
    with tab_sql:
        content, error = results.get('sql', (None, "Not generated"))
        if error:
            st.error(f"SQL Model Error: {error}")
        elif content:
            st.code(content, language="sql")
    
    with tab_sources:
        content, error = results.get('sources', (None, "Not generated"))
        if error:
            st.error(f"sources.yml Error: {error}")
        elif content:
            st.code(content, language="yaml")
    
    with tab_schema:
        content, error = results.get('schema', (None, "Not generated"))
        if error:
            st.error(f"schema.yml Error: {error}")
        elif content:
            st.code(content, language="yaml")


# --- Main Application ---

def main():
    """Main application entry point."""
    
    if 'generated_results' not in st.session_state:
        st.session_state.generated_results = None
    if 'current_tgt_table' not in st.session_state:
        st.session_state.current_tgt_table = None
    
    st.title("DBT Model Generator")
    st.markdown("Generate DBT SQL model, sources.yml, and schema.yml from mapping documents")
    st.divider()
    
    dialect, schema_name = render_sidebar()
    
    st.subheader("Upload Mapping File")
    uploaded_file = st.file_uploader(
        "Choose your mapping file",
        type=["xlsx", "xls", "csv"],
        help="Upload Excel or CSV file with column mappings"
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
    
    st.subheader("Generate DBT Artifacts")
    
    if st.button("Generate All", type="primary"):
        with st.spinner("Generating DBT artifacts..."):
            filtered_df = filter_by_table_pair(df, col_map, src_table, tgt_table)
            
            results = generate_dbt_artifacts(
                filtered_df, col_map, src_table, tgt_table, schema_name, dialect
            )
            
            st.session_state.generated_results = results
            st.session_state.current_tgt_table = tgt_table
            
            errors = [k for k, (c, e) in results.items() if e]
            if errors:
                st.warning(f"Completed with errors in: {', '.join(errors)}")
            else:
                st.success("All DBT artifacts generated successfully!")
    
    if st.session_state.generated_results:
        st.divider()
        st.subheader("Generated Artifacts")
        display_results(
            st.session_state.generated_results,
            st.session_state.current_tgt_table
        )


if __name__ == "__main__":
    main()