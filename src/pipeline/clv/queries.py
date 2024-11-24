"""SQL queries for CLV pipeline"""
from typing import Dict, Any, Optional

def get_transaction_data_query(config: Dict[str, Any]) -> str:
    """
    Generate query to fetch transaction data from BigQuery based on CLV_360 table structure
    
    Args:
        config: Configuration dictionary from data_processing_config.yaml
    """
    # Handle date parameters
    max_date = config.get('MAX_PURCHASE_DATE', 'CURRENT_DATE()')
    min_date = f"DATE('{config['MIN_PURCHASE_DATE']}')"
    cohort_month = f"DATE('{config['COHORT_MONTH']}')"
    
    # Build channel condition
    channel_conditions = []
    if config['INCLUDE_ONLINE']:
        channel_conditions.append("has_online_purchases = 1")
    if config['INCLUDE_STORE']:
        channel_conditions.append("has_store_purchases = 1")
    channel_filter = f"({' OR '.join(channel_conditions)})" if channel_conditions else "TRUE"
    
    query = f"""
    WITH fin AS (
        SELECT
            CAST(customer_id AS STRING) AS customer_id,
            CAST(cohort_month AS STRING) AS cohort_month,
            CAST(recency_days AS INT64) AS recency,
            CAST(frequency AS INT64) AS frequency,
            ROUND(total_revenue,2) AS monetary,
            ROUND(total_revenue,2) AS total_revenue,
            ROUND(revenue_trend,4) AS revenue_trend,
            ROUND(avg_transaction_value,2) AS avg_transaction_value,
            CAST(first_purchase_date AS DATE) AS first_purchase_date,
            CAST(last_purchase_date AS DATE) AS last_purchase_date,
            CAST(customer_age_days AS INT64) AS customer_age_days,
            CAST(distinct_categories AS INT64) AS distinct_categories,
            CAST(distinct_brands AS INT64) AS distinct_brands,
            ROUND(avg_interpurchase_days,2) AS avg_interpurchase_days,
            CAST(has_online_purchases AS INT64) AS has_online_purchases,
            CAST(has_store_purchases AS INT64) AS has_store_purchases,
            ROUND(total_discount_amount,2) AS total_discount_amount,
            ROUND(avg_discount_amount,2) AS avg_discount_amount,
            ROUND(COALESCE(discount_rate,0),3) AS discount_rate,
            CAST(sms_active AS INT64) AS sms_active,
            CAST(email_active AS INT64) AS email_active,
            CAST(is_loyalty_member AS INT64) AS is_loyalty_member,
            CAST(loyalty_points AS INT64) AS loyalty_points
        FROM
            `{config['PROJECT_ID']}.{config['DATASET']}.{config['TABLE']}`
        WHERE
            customer_id IS NOT NULL
            AND cohort_month IS NOT NULL
            AND frequency >= {config['MIN_FREQUENCY']}
            AND total_revenue >= {config['MIN_REVENUE']}
            AND avg_transaction_value >= {config['MIN_TRANSACTION_VALUE']}
            AND cohort_month >= {cohort_month}
            AND last_purchase_date <= {max_date}
            AND loyalty_points >= {config['MIN_LOYALTY_POINTS']}
            AND {channel_filter}
    )
    SELECT
        *
    FROM
        fin
    LIMIT
        {config['LIMIT']}
    """
    
    return query

def get_customer_features_query(config: Dict[str, Any]) -> str:
    """Generate query to fetch/calculate customer features"""
    return f"""
    WITH customer_stats AS (
        SELECT
            customer_id,
            COUNT(*) as frequency,
            MIN(transaction_date) as first_purchase_date,
            MAX(transaction_date) as last_purchase_date,
            SUM(transaction_amount) as total_spend,
            AVG(transaction_amount) as avg_transaction_value,
            COUNT(DISTINCT category) as distinct_categories,
            COUNT(DISTINCT brand) as distinct_brands,
            SUM(loyalty_points) as total_loyalty_points
        FROM
            `{config['project_id']}.{config['dataset']}.{config['table']}`
        WHERE
            transaction_date >= '{config['min_purchase_date']}'
            AND transaction_date <= '{config['max_purchase_date']}'
        GROUP BY
            customer_id
        HAVING
            COUNT(*) >= {config['min_frequency']}
            AND SUM(transaction_amount) >= {config['min_revenue']}
    ),
    
    engagement_metrics AS (
        SELECT
            customer_id,
            MAX(CASE WHEN channel = 'online' THEN 1 ELSE 0 END) as has_online_purchases,
            MAX(CASE WHEN channel = 'store' THEN 1 ELSE 0 END) as has_store_purchases,
            COUNT(DISTINCT DATE_TRUNC(transaction_date, MONTH)) as active_months
        FROM
            `{config['project_id']}.{config['dataset']}.{config['table']}`
        WHERE
            transaction_date >= '{config['min_purchase_date']}'
        GROUP BY
            customer_id
    )
    
    SELECT
        cs.*,
        em.has_online_purchases,
        em.has_store_purchases,
        em.active_months,
        DATE_DIFF(cs.last_purchase_date, cs.first_purchase_date, DAY) as customer_age_days,
        DATE_DIFF(CURRENT_DATE(), cs.last_purchase_date, DAY) as days_since_last_purchase,
        CASE 
            WHEN DATE_TRUNC(cs.first_purchase_date, MONTH) = '{config['cohort_month']}'
            THEN 1 
            ELSE 0 
        END as is_cohort_customer
    FROM
        customer_stats cs
    LEFT JOIN
        engagement_metrics em
    USING
        (customer_id)
    """

def save_predictions_query(
    project_id: str,
    dataset_id: str,
    table_id: str
) -> str:
    """Generate query to save model predictions"""
    return f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}`
    (
        customer_id INT64,
        predicted_clv FLOAT64,
        prediction_lower FLOAT64,
        prediction_upper FLOAT64,
        segment_id INT64,
        prediction_date DATE,
        model_version STRING
    )
    """ 