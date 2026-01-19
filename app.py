"""
DBT Model Generator - Streamlit Application
Generates DBT transformation SQL from mapping documents using LLM.
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

# Load environment variables
load_dotenv()

# PAGE CONFIGURATION
st.set_page_config(
    page_title="DBT Model Generator",
    page_icon="🔄",
    layout="wide"
)

# HELPER FUNCTIONS
def parse_file(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """Parse uploaded Excel or CSV file."""
    try:
        filename = uploaded_file.name.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type. Use .xlsx, .xls, or .csv"
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        return df, None
        
    except Exception as e:
        return None, f"Error parsing file: {str(e)}"


def get_column_mapping(df: pd.DataFrame) -> dict:
    """Map expected column names to actual DataFrame columns."""
    mapping = {}
    df_columns = list(df.columns)
    
    for std_name, patterns in COLUMN_PATTERNS.items():
        for col in df_columns:
            col_lower = col.lower().strip()
            if col_lower in patterns or any(p in col_lower for p in patterns):
                mapping[std_name] = col
                break
    
    return mapping


def get_table_pairs(df: pd.DataFrame, col_map: dict) -> list:
    """Extract unique source-target table pairs."""
    src_col = col_map.get('source_table')
    tgt_col = col_map.get('target_table')
    
    if src_col and tgt_col:
        pairs = df[[src_col, tgt_col]].drop_duplicates()
        return pairs.values.tolist()
    return []


def filter_by_table_pair(df: pd.DataFrame, col_map: dict, src_table: str, tgt_table: str) -> pd.DataFrame:
    """Filter DataFrame for specific source-target table pair."""
    src_col = col_map.get('source_table')
    tgt_col = col_map.get('target_table')
    
    if src_col and tgt_col:
        mask = (df[src_col] == src_table) & (df[tgt_col] == tgt_table)
        return df[mask].copy()
    return df.copy()


def build_mapping_text(df: pd.DataFrame, col_map: dict) -> str:
    """Format mapping DataFrame as text for LLM prompt."""
    lines = []
    
    for _, row in df.iterrows():
        src_col = row.get(col_map.get('source_column', ''), 'N/A')
        tgt_col = row.get(col_map.get('target_column', ''), 'N/A')
        src_type = row.get(col_map.get('source_datatype', ''), '')
        tgt_type = row.get(col_map.get('target_datatype', ''), '')
        logic = row.get(col_map.get('transformation_logic', ''), 'direct move')
        
        line = f"- **{src_col}** → **{tgt_col}**"
        if src_type and tgt_type:
            line += f" ({src_type} → {tgt_type})"
        line += f"\n  Transformation: {logic}"
        lines.append(line)
    
    return "\n".join(lines)


def build_user_prompt(df: pd.DataFrame, col_map: dict, src_table: str, tgt_table: str, schema: str) -> str:
    """Build the user prompt with mapping data."""
    mapping_text = build_mapping_text(df, col_map)
    
    return f"""Generate a DBT SQL model for the following transformation:

## Source Information
- **Source Table**: {src_table}
- **Target Table**: {tgt_table}
- **Schema Name**: {schema}

## Column Mappings
{mapping_text}

## Requirements
1. Generate a complete DBT model SQL file
2. Use `{{{{ source('{schema}', '{src_table}') }}}}` for the source table
3. Apply all transformation logic as specified
4. For validations: return NULL if validation fails
5. Output ONLY the SQL code, no explanations

Generate the DBT model SQL:"""


def call_llm(system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
    """Call Groq API to generate DBT model."""
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
        
        sql = response.choices[0].message.content.strip()
        
        # Clean markdown fences if present
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        
        return sql.strip(), None
        
    except Exception as e:
        return None, f"API Error: {str(e)}"

# UI COMPONENTS
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


# MAIN APPLICATION
def main():
    # Initialize session state
    if 'generated_sql' not in st.session_state:
        st.session_state.generated_sql = None
    
    # Header
    st.title("DBT Model Generator")
    st.markdown("Generate DBT transformation SQL from mapping documents")
    st.divider()
    
    # Sidebar
    dialect, schema_name = render_sidebar()
    
    # File upload
    st.subheader("Upload Mapping File")
    uploaded_file = st.file_uploader(
        "Choose your mapping file",
        type=["xlsx", "xls", "csv"],
        help="Upload Excel or CSV file with column mappings"
    )
    
    if not uploaded_file:
        st.info("Upload a mapping file to get started")
        return
    
    # Parse file
    df, error = parse_file(uploaded_file)
    
    if error:
        st.error(error)
        return
    
    # Get column mapping
    col_map = get_column_mapping(df)
    
    # Preview
    st.subheader("Mapping Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Table pair selection
    st.subheader("Select Table Pair")
    table_pairs = get_table_pairs(df, col_map)
    
    if not table_pairs:
        st.warning("Could not detect source/target tables. Check column names.")
        return
    
    options = [f"{src} → {tgt}" for src, tgt in table_pairs]
    selected = st.selectbox("Source → Target Table", options)
    
    idx = options.index(selected)
    src_table, tgt_table = table_pairs[idx]
    
    st.divider()
    
    # Generation
    st.subheader("Generate DBT Model")
    
    if st.button("Generate SQL", type="primary"):
        with st.spinner("Generating DBT model..."):
            # Filter data for selected table pair
            filtered_df = filter_by_table_pair(df, col_map, src_table, tgt_table)
            
            # Build prompts
            system_prompt = SYSTEM_PROMPT + get_dialect_prompt(dialect)
            user_prompt = build_user_prompt(filtered_df, col_map, src_table, tgt_table, schema_name)
            
            # Call LLM
            sql, error = call_llm(system_prompt, user_prompt)
            
            if error:
                st.error(error)
            else:
                st.session_state.generated_sql = sql
                st.success("DBT model generated successfully!")
    
    # Output
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