"""
DBT Model Generator
Generates DBT transformation SQL from mapping documents.
"""

import os
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from config import (
    MODEL_NAME, TEMPERATURE, MAX_TOKENS,
    SUPPORTED_DIALECTS, COLUMN_PATTERNS,
    SYSTEM_PROMPT, get_dialect_prompt
)

load_dotenv()

st.set_page_config(page_title="DBT Model Generator", layout="wide")


# --- Helper Functions ---

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


def get_column_mapping(df: pd.DataFrame) -> dict[str, str]:
    """
    Map expected standard column names to actual DataFrame columns.
    
    Uses two matching strategies:
    1. Exact match against known patterns
    2. Substring match for flexible column naming
    
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
            is_substring_match = any(pattern in col_normalized for pattern in patterns)
            
            if is_exact_match or is_substring_match:
                mapping[std_name] = col
                break
    
    return mapping


def get_table_pairs(df: pd.DataFrame, col_map: dict[str, str]) -> dict[str, tuple[str, str]]:
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


def format_type_info(src_type: str, tgt_type: str) -> str:
    """
    Format source and target data types into a display string.
    
    Args:
        src_type: Source column data type
        tgt_type: Target column data type
        
    Returns:
        Formatted type string, e.g., " (VARCHAR → INT)" or empty string
    """
    types = [src_type, tgt_type]
    
    if not any(types):
        return ""
    
    formatted = ' → '.join(t for t in types if t)
    return f" ({formatted})"


def build_mapping_text(df: pd.DataFrame, col_map: dict[str, str]) -> str:
    """
    Format mapping DataFrame as text for LLM prompt.
    
    Args:
        df: Filtered mapping DataFrame
        col_map: Column name mapping dictionary
        
    Returns:
        Formatted mapping text with one line per column mapping
    """
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


def build_user_prompt(
    df: pd.DataFrame, 
    col_map: dict[str, str], 
    src_table: str, 
    tgt_table: str, 
    schema: str
) -> str:
    """
    Build the user prompt with mapping data for LLM.
    
    Args:
        df: Filtered mapping DataFrame
        col_map: Column name mapping dictionary
        src_table: Source table name
        tgt_table: Target table name
        schema: Schema name for DBT source reference
        
    Returns:
        Formatted prompt string
    """
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


def strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences from LLM output.
    
    Handles common patterns like ```sql, ```, etc.
    
    Args:
        text: Raw LLM output text
        
    Returns:
        Cleaned SQL text without markdown formatting
    """
    text = text.strip()
    
    if text.startswith("```sql"):
        text = text[6:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    return text.strip()


def call_llm(system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
    """
    Call Groq API to generate DBT model SQL.
    
    Args:
        system_prompt: System message for LLM context
        user_prompt: User message with mapping details
        
    Returns:
        Tuple of (sql_code, None) on success, or (None, error_message) on failure
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
        sql = strip_markdown_fences(raw_output)
        
        return sql, None
        
    except Exception as e:
        return None, f"API Error: {e}"


def render_sidebar() -> tuple[str, str]:
    """
    Render sidebar with configuration options.
    
    Returns:
        Tuple of (selected_dialect, schema_name)
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
        
        st.divider()
        
        with st.expander("How to use"):
            st.markdown("""
            1. **Upload** your mapping Excel/CSV file
            2. **Select** the SQL dialect
            3. **Choose** source-target table pair
            4. **Click** Generate to create DBT model
            5. **Download** the generated SQL file
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


# --- Main Application ---

def main():
    """Main application entry point."""
    
    if 'generated_sql' not in st.session_state:
        st.session_state.generated_sql = None
    
    st.title("DBT Model Generator")
    st.markdown("Generate DBT transformation SQL from mapping documents")
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
    
    st.subheader("Generate DBT Model")
    
    if st.button("Generate SQL", type="primary"):
        with st.spinner("Generating DBT model..."):
            filtered_df = filter_by_table_pair(df, col_map, src_table, tgt_table)
            
            system_prompt = SYSTEM_PROMPT + get_dialect_prompt(dialect)
            user_prompt = build_user_prompt(
                filtered_df, col_map, src_table, tgt_table, schema_name
            )
            
            sql, error = call_llm(system_prompt, user_prompt)
            
            if error:
                st.error(error)
            else:
                st.session_state.generated_sql = sql
                st.success("DBT model generated successfully!")
    
    if st.session_state.generated_sql:
        st.subheader("Generated DBT Model")
        st.code(st.session_state.generated_sql, language="sql")
        
        st.download_button(
            label="Download SQL File",
            data=st.session_state.generated_sql,
            file_name=f"{tgt_table.lower()}.sql",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()