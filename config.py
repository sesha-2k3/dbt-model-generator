"""
Configuration for DBT Model Generator.
Contains constants, SQL dialect configs, and system prompt.
"""

# GROQ MODEL CONFIGURATION
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1
MAX_TOKENS = 4096


# SUPPORTED SQL DIALECTS
SUPPORTED_DIALECTS = ["Snowflake", "PostgreSQL"]

DIALECT_CONFIGS = {
    "Snowflake": {
        "name": "Snowflake",
        "regex_match": "REGEXP_LIKE(column, 'pattern')",
        "regex_replace": "REGEXP_REPLACE(column, 'pattern', 'replacement')",
        "date_convert": "TO_DATE(column, 'YYYY-MM-DD')",
        "timestamp_convert": "TO_TIMESTAMP(column, 'YYYY-MM-DD HH24:MI:SS')",
        "string_length": "LENGTH(column)",
        "casting": "column::DATA_TYPE or CAST(column AS DATA_TYPE)",
        "concat": "CONCAT(col1, col2) or col1 || col2",
        "case_syntax": "CASE WHEN condition THEN result ELSE NULL END",
        "string_functions": "UPPER(), LOWER(), TRIM(), SUBSTRING()",
        "notes": "Use REGEXP_LIKE for regex. Date formats: 'YYYY-MM-DD'."
    },
    "PostgreSQL": {
        "name": "PostgreSQL",
        "regex_match": "column ~ 'pattern'",
        "regex_not_match": "column !~ 'pattern'",
        "regex_replace": "REGEXP_REPLACE(column, 'pattern', 'replacement')",
        "date_convert": "TO_DATE(column, 'YYYY-MM-DD')",
        "timestamp_convert": "TO_TIMESTAMP(column, 'YYYY-MM-DD HH24:MI:SS')",
        "string_length": "LENGTH(column)",
        "casting": "column::DATA_TYPE or CAST(column AS DATA_TYPE)",
        "concat": "CONCAT(col1, col2) or col1 || col2",
        "case_syntax": "CASE WHEN condition THEN result ELSE NULL END",
        "string_functions": "UPPER(), LOWER(), TRIM(), SUBSTRING()",
        "notes": "Use ~ for regex matching. Use ~* for case-insensitive."
    }
}

# COLUMN NAME MAPPINGS
COLUMN_PATTERNS = {
    'source_table': ['source table', 'source_table', 'sourcetable'],
    'source_column': ['source column', 'source_column', 'sourcecolumn'],
    'source_datatype': ['source column datatype', 'source_datatype', 'source datatype'],
    'target_table': ['target table', 'target_table', 'targettable'],
    'target_column': ['target column', 'target_column', 'targetcolumn'],
    'target_datatype': ['target column datatype', 'target_datatype', 'target datatype'],
    'transformation_logic': ['transformation logic', 'transformation_logic', 'logic']
}

# SYSTEM PROMPT
SYSTEM_PROMPT = """You are an expert data engineer specializing in DBT (Data Build Tool) transformations. Your task is to generate DBT-compatible SQL models based on column mapping specifications.

## Core Rules

1. **Output Format**: Generate ONLY the SQL code for the DBT model. No explanations, no markdown code fences, no additional commentary.

2. **Validation Handling**: When transformation logic specifies validation:
   - If a value PASSES validation → keep the original value
   - If a value FAILS validation → set to NULL
   - Use CASE WHEN statements for validation logic

3. **Direct Move**: When transformation logic says "direct move" → simply select the column with its target alias

4. **Row Filtering**: When transformation logic specifies filtering conditions (e.g., "consider only customers who are 'Active'") → add to WHERE clause

5. **DBT Source Syntax**: Use `{{ source('schema_name', 'table_name') }}` for source table references

6. **Model Structure**:
   - Use CTEs for clarity when needed
   - Include a final SELECT with all transformed columns
   - Add appropriate WHERE clauses for row-level filters
   - Use clear column aliases matching target column names

## Transformation Logic Interpretation

| Plain English | SQL Pattern |
|---------------|-------------|
| "direct move" | column AS target_name |
| "validate must have only letters" | CASE WHEN REGEXP_LIKE(col, '^[A-Za-z]+$') THEN col ELSE NULL END |
| "validate must have only numbers" | CASE WHEN REGEXP_LIKE(col, '^[0-9]+$') THEN col ELSE NULL END |
| "phone must have 10 digits, may have hyphen" | CASE WHEN LENGTH(REGEXP_REPLACE(col, '-', '')) = 10 THEN col ELSE NULL END |
| "convert to date format" | TO_DATE(col, 'format') |
| "ensure valid email" | CASE WHEN col LIKE '%@%.%' THEN col ELSE NULL END |
| "uppercase" / "lowercase" | UPPER(col) / LOWER(col) |
| "trim whitespace" | TRIM(col) |
| "concatenate X and Y" | CONCAT(X, ' ', Y) |
| "consider only rows where X = 'value'" | WHERE X = 'value' |

## Few-Shot Examples

### Example 1: Name validation
**Logic**: "validate first name must have english literals, no numeric characters"
**SQL**:
CASE 
    WHEN REGEXP_LIKE(first_name, '^[A-Za-z]+$') THEN first_name
    ELSE NULL
END AS first_name

### Example 2: Phone validation
**Logic**: "phone number must have 10 digits may have hyphen in between"
**SQL**:
CASE 
    WHEN LENGTH(REGEXP_REPLACE(phone, '-', '')) = 10 
         AND REGEXP_LIKE(REGEXP_REPLACE(phone, '-', ''), '^[0-9]+$')
    THEN phone
    ELSE NULL
END AS phone

### Example 3: Date conversion
**Logic**: "convert the signup_date column to exact date format"
**SQL**:
TO_DATE(signup_date, 'YYYY-MM-DD') AS signup_date

### Example 4: Row filtering
**Logic**: "consider only the customers who are 'Active'"
**SQL**:
WHERE status = 'Active'
"""


def get_dialect_prompt(dialect: str) -> str:
    """Generate dialect-specific prompt section."""
    config = DIALECT_CONFIGS.get(dialect, DIALECT_CONFIGS["PostgreSQL"])
    
    return f"""
## SQL Dialect: {config['name']}

Use the following syntax for this dialect:
- **Regex Matching**: {config['regex_match']}
- **Regex Replace**: {config['regex_replace']}
- **Date Conversion**: {config['date_convert']}
- **Timestamp Conversion**: {config['timestamp_convert']}
- **String Length**: {config['string_length']}
- **Type Casting**: {config['casting']}
- **Concatenation**: {config['concat']}
- **CASE Syntax**: {config['case_syntax']}
- **String Functions**: {config['string_functions']}

**Notes**: {config['notes']}
"""