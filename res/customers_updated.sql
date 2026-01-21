WITH transformed_customers AS (
    SELECT 
        customer_id AS customer_id,
        -- Validate first name to have only english literals and no numeric characters
        CASE 
            WHEN REGEXP_LIKE(first_name, '^[A-Za-z]+$') THEN first_name
            ELSE NULL
        END AS first_name,
        -- Validate last name to have only english literals and no numeric characters
        CASE 
            WHEN REGEXP_LIKE(last_name, '^[A-Za-z]+$') THEN last_name
            ELSE NULL
        END AS last_name,
        email AS email,
        -- Validate phone number to have 10 digits, may have hyphen in between
        CASE 
            WHEN LENGTH(REGEXP_REPLACE(phone, '-', '')) = 10 
                 AND REGEXP_LIKE(REGEXP_REPLACE(phone, '-', ''), '^[0-9]+$')
            THEN phone
            ELSE NULL
        END AS phone,
        -- Basic validation for city, consider using a reference table for comprehensive validation
        CASE 
            WHEN city IS NOT NULL AND TRIM(city) != '' THEN city
            ELSE NULL
        END AS city,
        country AS country,
        -- Convert signup_date to date format
        TO_DATE(signup_date, 'YYYY-MM-DD') AS signup_date,
        status AS status
    FROM 
        {{ source('source_schema', 'customers') }}
)
SELECT 
    customer_id,
    first_name,
    last_name,
    email,
    phone,
    city,
    country,
    signup_date,
    status
FROM 
    transformed_customers
WHERE 
    -- Filter rows to consider only 'Active' customers
    status = 'Active'