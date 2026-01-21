-- DBT model: orders_cleaned
-- Transforms and validates order data

SELECT
    -- Validate order_id format: ORD- followed by exactly 8 digits
    CASE 
        WHEN REGEXP_LIKE(order_id, '^ORD-[0-9]{8}$') THEN order_id
        ELSE NULL
    END AS order_id,

    -- Validate email format: contains @ symbol and a valid domain with at least one dot after @
    CASE 
        WHEN customer_email LIKE '%_@_%.__%' THEN customer_email
        ELSE NULL
    END AS customer_email,

    -- Extract only numeric digits from phone and validate it must be exactly 10 digits
    CASE 
        WHEN LENGTH(REGEXP_REPLACE(customer_phone, '[^0-9]', '')) = 10 THEN REGEXP_REPLACE(customer_phone, '[^0-9]', '')
        ELSE NULL
    END AS customer_phone,

    -- Concatenate first_name and last_name with a single space in between
    CONCAT(first_name, ' ', last_name) AS full_name,

    -- Convert to decimal and validate amount must be greater than 0 and less than 1000000
    CASE 
        WHEN CAST(order_amount AS DECIMAL(18,2)) > 0 AND CAST(order_amount AS DECIMAL(18,2)) < 1000000 THEN CAST(order_amount AS DECIMAL(18,2))
        ELSE NULL
    END AS order_amount,

    -- Validate currency_code must be exactly 3 uppercase letters (ISO 4217 format)
    CASE 
        WHEN LENGTH(currency_code) = 3 AND REGEXP_LIKE(currency_code, '^[A-Z]{3}$') THEN currency_code
        ELSE NULL
    END AS currency_code,

    -- Convert order_date string to DATE format YYYY-MM-DD
    TO_DATE(order_date, 'YYYY-MM-DD') AS order_date,

    -- Convert order_timestamp string to TIMESTAMP format YYYY-MM-DD HH:MI:SS
    TO_TIMESTAMP(order_timestamp, 'YYYY-MM-DD HH24:MI:SS') AS order_timestamp,

    -- Trim leading and trailing whitespace and replace multiple spaces with single space
    REGEXP_REPLACE(TRIM(shipping_address), ' +', ' ') AS shipping_address,

    -- Validate US postal code must be either 5 digits or 5 digits followed by hyphen and 4 digits (XXXXX or XXXXX-XXXX)
    CASE 
        WHEN REGEXP_LIKE(postal_code, '^[0-9]{5}$') OR REGEXP_LIKE(postal_code, '^[0-9]{5}-[0-9]{4}$') THEN postal_code
        ELSE NULL
    END AS postal_code,

    -- Validate country_code must be exactly 2 uppercase letters and convert to uppercase if lowercase
    CASE 
        WHEN LENGTH(country_code) = 2 AND REGEXP_LIKE(UPPER(country_code), '^[A-Z]{2}$') THEN UPPER(country_code)
        ELSE NULL
    END AS country_code,

    -- Map status values: 'P' or 'pending' to 'PENDING', 'C' or 'completed' to 'COMPLETED', 'F' or 'failed' to 'FAILED', else NULL
    CASE 
        WHEN LOWER(order_status) IN ('p', 'pending') THEN 'PENDING'
        WHEN LOWER(order_status) IN ('c', 'completed') THEN 'COMPLETED'
        WHEN LOWER(order_status) IN ('f', 'failed') THEN 'FAILED'
        ELSE NULL
    END AS order_status,

    -- Convert to decimal and validate must be between 0 and 100 inclusive
    CASE 
        WHEN CAST(discount_percentage AS DECIMAL(5,2)) >= 0 AND CAST(discount_percentage AS DECIMAL(5,2)) <= 100 THEN CAST(discount_percentage AS DECIMAL(5,2))
        ELSE NULL
    END AS discount_percentage,

    -- Validate IPv4 format with 4 octets separated by dots where each octet is 0-255
    CASE 
        WHEN REGEXP_LIKE(ip_address, '^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$') THEN ip_address
        ELSE NULL
    END AS ip_address,

    -- Extract last 4 characters and validate they must be all digits
    CASE 
        WHEN LENGTH(RIGHT(credit_card_last4, 4)) = 4 AND REGEXP_LIKE(RIGHT(credit_card_last4, 4), '^[0-9]{4}$') THEN RIGHT(credit_card_last4, 4)
        ELSE NULL
    END AS credit_card_last4,

    created_at,

    updated_at,

    -- Convert 0 to FALSE and 1 to TRUE and validate value must be either 0 or 1
    CASE 
        WHEN is_deleted = 0 THEN FALSE
        WHEN is_deleted = 1 THEN TRUE
        ELSE NULL
    END AS is_deleted,

    -- Truncate notes to first 1000 characters and remove any HTML tags
    REGEXP_REPLACE(SUBSTRING(notes, 1, 1000), '<.*?>', '') AS notes,

    -- Convert to integer and validate quantity must be a positive whole number greater than 0
    CASE 
        WHEN CAST(quantity AS INTEGER) > 0 THEN CAST(quantity AS INTEGER)
        ELSE NULL
    END AS quantity,

    -- Validate SKU must be alphanumeric with optional hyphens and uppercase all letters
    CASE 
        WHEN REGEXP_LIKE(sku_code, '^[A-Z0-9-]+$') THEN UPPER(sku_code)
        ELSE NULL
    END AS sku_code,

    payment_method

FROM {{ source('source_schema', 'orders') }}
WHERE payment_method IN ('credit_card', 'debit_card', 'paypal')