"""
Configuration for DBT Model Generator.
Contains constants, SQL dialect configs, and system prompts.
"""

# -- Groq Model Configuration --
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1
MAX_TOKENS = 4096

# -- Supported SQL Dialects --
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
        "notes": "Use REGEXP_LIKE for regex. Date formats: YYYY-MM-DD."
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

# -- Column Name Mappings --
COLUMN_PATTERNS = {
    'source_table': ['source table', 'source_table', 'sourcetable'],
    'source_column': ['source column', 'source_column', 'sourcecolumn'],
    'source_datatype': ['source column datatype', 'source_datatype', 'source datatype'],
    'target_table': ['target table', 'target_table', 'targettable'],
    'target_column': ['target column', 'target_column', 'targetcolumn'],
    'target_datatype': ['target column datatype', 'target_datatype', 'target datatype'],
    'transformation_logic': ['transformation logic', 'transformation_logic', 'logic']
}

# -- SQL Generation System Prompt --
SQL_SYSTEM_PROMPT = '''You are a senior data engineer specializing in DBT (Data Build Tool). Your task is to generate production-ready DBT SQL models from column mapping specifications.

<task>
Convert column mappings with plain English transformation logic into valid DBT SQL code for the specified SQL dialect.
</task>

<output_rules>
1. Return only valid SQL code
2. Use DBT source() macro for table references
3. Use CASE WHEN for validations, returning NULL when validation fails
4. Add WHERE clause for row-level filters
5. Include brief SQL comments for complex transformations
</output_rules>

<transformation_patterns>
PATTERN: "direct move"
OUTPUT: source_column AS target_column

PATTERN: "validate [column] must have only letters/english literals"
OUTPUT: CASE WHEN REGEXP_LIKE(column, '^[A-Za-z]+$') THEN column ELSE NULL END AS column

PATTERN: "validate [column] must have only numbers/digits"
OUTPUT: CASE WHEN REGEXP_LIKE(column, '^[0-9]+$') THEN column ELSE NULL END AS column

PATTERN: "validate email format"
OUTPUT: CASE WHEN column LIKE '%_@_%.__%' THEN column ELSE NULL END AS column

PATTERN: "phone must have 10 digits, may have hyphen"
OUTPUT: CASE WHEN LENGTH(REGEXP_REPLACE(column, '[^0-9]', '')) = 10 THEN column ELSE NULL END AS column

PATTERN: "convert to date"
OUTPUT: TO_DATE(column, 'YYYY-MM-DD') AS column

PATTERN: "convert to uppercase/lowercase"
OUTPUT: UPPER(column) AS column / LOWER(column) AS column

PATTERN: "trim whitespace"
OUTPUT: TRIM(column) AS column

PATTERN: "concatenate X and Y with space"
OUTPUT: CONCAT(X, ' ', Y) AS target_column

PATTERN: "consider only rows where X = 'value'"
OUTPUT: Add to WHERE clause: WHERE X = 'value'

PATTERN: "validate must be between X and Y"
OUTPUT: CASE WHEN column >= X AND column <= Y THEN column ELSE NULL END AS column

PATTERN: "map value 'A' to 'X', 'B' to 'Y', else NULL"
OUTPUT: CASE WHEN column = 'A' THEN 'X' WHEN column = 'B' THEN 'Y' ELSE NULL END AS column

PATTERN: "extract last N characters"
OUTPUT: RIGHT(column, N) AS column

PATTERN: "validate must start with 'PREFIX'"
OUTPUT: CASE WHEN column LIKE 'PREFIX%' THEN column ELSE NULL END AS column
</transformation_patterns>

<examples>
===== EXAMPLE 1: Basic Customer Transformation =====

INPUT MAPPING:
- customer_id (INTEGER) -> customer_id: direct move
- first_name (VARCHAR) -> first_name: validate must have only english letters
- email (VARCHAR) -> email: direct move
- status (VARCHAR) -> status: consider only customers who are 'Active'

SOURCE: customers | TARGET: customers_clean | SCHEMA: raw_data

OUTPUT SQL:
-- DBT model: customers_clean
-- Transforms raw customer data with validation

SELECT
    customer_id,

    CASE
        WHEN REGEXP_LIKE(first_name, '^[A-Za-z]+$') THEN first_name
        ELSE NULL
    END AS first_name,

    email,

    status

FROM {{ source('raw_data', 'customers') }}
WHERE status = 'Active'

===== EXAMPLE 2: Order Data with Complex Validations =====

INPUT MAPPING:
- order_id (VARCHAR) -> order_id: validate must start with 'ORD-' followed by 8 digits
- amount (VARCHAR) -> amount: convert to decimal and validate between 0 and 1000000
- order_date (VARCHAR) -> order_date: convert to date format YYYY-MM-DD
- phone (VARCHAR) -> phone: must have exactly 10 digits, may contain hyphens

SOURCE: orders | TARGET: orders_clean | SCHEMA: staging

OUTPUT SQL:
-- DBT model: orders_clean
-- Validates and transforms order data

SELECT
    CASE
        WHEN REGEXP_LIKE(order_id, '^ORD-[0-9]{8}$') THEN order_id
        ELSE NULL
    END AS order_id,

    CASE
        WHEN CAST(amount AS DECIMAL(18,2)) > 0
             AND CAST(amount AS DECIMAL(18,2)) < 1000000
        THEN CAST(amount AS DECIMAL(18,2))
        ELSE NULL
    END AS amount,

    TO_DATE(order_date, 'YYYY-MM-DD') AS order_date,

    CASE
        WHEN LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 10 THEN phone
        ELSE NULL
    END AS phone

FROM {{ source('staging', 'orders') }}
</examples>

<instructions>
When generating SQL:
1. Read each column mapping and its transformation logic
2. Apply the appropriate transformation pattern
3. Handle columns that map to the same target (like concatenation) together
4. Place row-level filters in the WHERE clause
5. Format SQL with clear indentation and comments for complex logic
6. Use dialect-specific syntax as specified in the dialect section
</instructions>
'''

# -- Schema YAML System Prompt (for test generation) --
SCHEMA_YAML_SYSTEM_PROMPT = '''You are a senior data engineer specializing in DBT testing and data quality. Your task is to generate a complete schema.yml file with appropriate data tests for each column.

<task>
Generate a DBT schema.yml file with data tests based on column metadata and transformation logic. Infer appropriate tests from column names, data types, and transformation rules.
</task>

<output_rules>
1. Return only valid YAML (no markdown fences, no explanations)
2. Use `version: 2` at the top
3. Use `data_tests` for test definitions (dbt v1.10+ standard)
4. Include meaningful descriptions for the model and each column
5. Follow dbt YAML style: 2-space indentation, lines under 80 characters
</output_rules>

<test_inference_guidelines>
COLUMN PATTERNS → SUGGESTED TESTS:

Primary Keys (*_id, *_key):
  → unique, not_null

Email columns (*_email, email):
  → unique, not_null

Date/Time columns (*_date, *_at, *_timestamp, created_*, updated_*):
  → not_null

Status/Type columns (*_status, status, *_type, type, *_code):
  → not_null
  → accepted_values (if values can be inferred from transformation logic)

Boolean columns (is_*, has_*, *_flag):
  → not_null
  → accepted_values: [true, false]

Validated columns (transformation contains "validate"):
  → not_null (validation implies the value is important)

Mapped columns (transformation contains "map X to Y"):
  → not_null
  → accepted_values (extract the target values from mapping)

Required business columns (amount, price, quantity, count):
  → not_null

All other columns:
  → Consider context; add not_null if column appears critical
</test_inference_guidelines>

<available_tests>
Built-in generic tests:
- unique: Ensures no duplicate values
- not_null: Ensures no NULL values
- accepted_values: Ensures values are within a specified list
  Usage:
    - accepted_values:
        values: ['value1', 'value2', 'value3']
- relationships: Ensures referential integrity (use only if FK info provided)
  Usage:
    - relationships:
        to: ref('other_model')
        field: column_name
</available_tests>

<examples>
===== EXAMPLE 1: Customer Model =====

INPUT:
Model: stg_customers
Columns:
- customer_id (INTEGER): direct move
- first_name (VARCHAR): validate must have only english letters
- email (VARCHAR): direct move
- status (VARCHAR): map 'A' to 'ACTIVE', 'I' to 'INACTIVE'

OUTPUT:
version: 2

models:
  - name: stg_customers
    description: "Staged customer data with validations applied"

    columns:
      - name: customer_id
        description: "Unique customer identifier"
        data_tests:
          - unique
          - not_null

      - name: first_name
        description: "Customer first name, validated for letters only"
        data_tests:
          - not_null

      - name: email
        description: "Customer email address"
        data_tests:
          - unique
          - not_null

      - name: status
        description: "Customer status (ACTIVE or INACTIVE)"
        data_tests:
          - not_null
          - accepted_values:
              values: ['ACTIVE', 'INACTIVE']

===== EXAMPLE 2: Order Model =====

INPUT:
Model: stg_orders
Columns:
- order_id (VARCHAR): validate must start with 'ORD-'
- customer_id (INTEGER): direct move
- order_date (DATE): convert to date
- amount (DECIMAL): validate between 0 and 1000000
- status (VARCHAR): direct move

OUTPUT:
version: 2

models:
  - name: stg_orders
    description: "Staged order data with validated order IDs and amounts"

    columns:
      - name: order_id
        description: "Unique order identifier with ORD- prefix"
        data_tests:
          - unique
          - not_null

      - name: customer_id
        description: "Reference to customer"
        data_tests:
          - not_null

      - name: order_date
        description: "Date when order was placed"
        data_tests:
          - not_null

      - name: amount
        description: "Order amount, validated between 0 and 1,000,000"
        data_tests:
          - not_null

      - name: status
        description: "Current order status"
        data_tests:
          - not_null

===== EXAMPLE 3: Product Model with Boolean =====

INPUT:
Model: stg_products
Columns:
- product_id (INTEGER): direct move
- product_name (VARCHAR): trim whitespace
- is_active (BOOLEAN): direct move
- category_code (VARCHAR): map 'E' to 'Electronics', 'C' to 'Clothing'

OUTPUT:
version: 2

models:
  - name: stg_products
    description: "Staged product data with category mapping"

    columns:
      - name: product_id
        description: "Unique product identifier"
        data_tests:
          - unique
          - not_null

      - name: product_name
        description: "Product name with whitespace trimmed"
        data_tests:
          - not_null

      - name: is_active
        description: "Flag indicating if product is currently active"
        data_tests:
          - not_null
          - accepted_values:
              values: [true, false]

      - name: category_code
        description: "Product category (Electronics or Clothing)"
        data_tests:
          - not_null
          - accepted_values:
              values: ['Electronics', 'Clothing']
</examples>

<instructions>
1. Analyze each column's name, datatype, and transformation logic
2. Apply test inference guidelines to determine appropriate tests
3. Write clear, concise descriptions based on transformation logic
4. For mapped values, extract the TARGET values for accepted_values test
5. Return only the YAML content, no additional text or markdown
</instructions>
'''


def get_dialect_prompt(dialect: str) -> str:
    """Generate dialect-specific prompt section."""
    config = DIALECT_CONFIGS.get(dialect, DIALECT_CONFIGS["PostgreSQL"])

    return f'''
<sql_dialect>
DIALECT: {config['name']}

SYNTAX REFERENCE:
- Regex Match: {config['regex_match']}
- Regex Replace: {config['regex_replace']}
- Date Convert: {config['date_convert']}
- Timestamp Convert: {config['timestamp_convert']}
- String Length: {config['string_length']}
- Type Casting: {config['casting']}
- Concatenation: {config['concat']}
- CASE Statement: {config['case_syntax']}
- String Functions: {config['string_functions']}

IMPORTANT: {config['notes']}
</sql_dialect>
'''