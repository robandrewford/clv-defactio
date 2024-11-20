import pandas as pd
import numpy as np

# Data processor
class CLVDataProcessor:
    """
    CLV data processing class with data quality checks
    """
    def __init__(self, config=None):
        if config is None:
            import yaml
            with open('src/config/data_processing_config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

    def _build_query(self):
        """Build BigQuery query based on configuration parameters"""
        # Handle date parameters
        max_date = self.config['MAX_PURCHASE_DATE'] or 'CURRENT_DATE()'
        min_date = f"DATE('{self.config['MIN_PURCHASE_DATE']}')"
        cohort_month = f"DATE('{self.config['COHORT_MONTH']}')"

        # Build channel condition
        channel_conditions = []
        if self.config['INCLUDE_ONLINE']:
            channel_conditions.append("has_online_purchases = 1")
        if self.config['INCLUDE_STORE']:
            channel_conditions.append("has_store_purchases = 1")
        channel_filter = f"({' OR '.join(channel_conditions)})" if channel_conditions else "TRUE"

        query = f"""
        WITH
        fin AS (
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
          `{self.config['PROJECT_ID']}.{self.config['DATASET']}.{self.config['TABLE']}`
        WHERE
          customer_id IS NOT NULL
          AND cohort_month IS NOT NULL
          AND frequency >= {self.config['MIN_FREQUENCY']}
          AND total_revenue >= {self.config['MIN_REVENUE']}
          AND avg_transaction_value >= {self.config['MIN_TRANSACTION_VALUE']}
          AND cohort_month >= {cohort_month}
          # AND first_purchase_date >= {min_date}
          AND last_purchase_date <= {max_date}
          AND loyalty_points >= {self.config['MIN_LOYALTY_POINTS']}
          AND {channel_filter}
        )
        SELECT
          *
        FROM
          fin
        LIMIT
          {self.config['LIMIT']}
        """

        return query

    def load_data(self, query=None, project_id=None, csv_path=None):
        """Load data either from BigQuery or CSV file"""
        try:
            if csv_path:
                self.data = pd.read_csv(csv_path)
                print(f"Successfully loaded {len(self.data):,} records from CSV")
            else:
                from google.cloud import bigquery

                client = bigquery.Client(project=project_id or self.config['PROJECT_ID'])

                # Build the query using config parameters
                default_query = self._build_query()

                self.data = client.query(query or default_query).to_dataframe()
                print(f"Successfully loaded {len(self.data):,} records from BigQuery")

            # Convert date columns
            date_columns = ['first_purchase_date', 'last_purchase_date', 'cohort_month']
            for col in date_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_datetime(self.data[col])

            return self

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def process_data(self):
        """
        Main data processing pipeline that handles:
        - Basic cleaning
        - RFM calculation
        - Quality validation
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        try:
            print(f"Starting data processing. Initial shape: {self.data.shape}")

            # 1. Basic cleaning
            self._clean_basic_data()

            # 2. Calculate RFM metrics
            self._calculate_rfm_metrics()

            # 3. Handle outliers and invalid values
            self._clean_monetary_values()
            self._clean_categorical_features()

            # 4. Validate data quality
            self._validate_data_quality()

            # 5. Generate final report
            self._generate_quality_report()

            return self

        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise

    def _clean_basic_data(self):
        """Basic data cleaning operations"""
        print("Cleaning data...")
        initial_count = len(self.data)

        # Remove invalid records
        self.data = self.data[
            (self.data['frequency'] >= self.config['MIN_FREQUENCY']) &
            (self.data['total_revenue'] > 0) &
            (self.data['avg_transaction_value'] > 0)
        ]

        # Drop duplicates
        self.data = self.data.drop_duplicates(subset=['customer_id'])

        # Handle missing values
        self.data['discount_rate'] = self.data['discount_rate'].fillna(0)

        records_removed = initial_count - len(self.data)
        print(f"Records removed: {records_removed}")

    def _calculate_rfm_metrics(self):
        """Calculate Recency, Frequency, Monetary metrics."""
        # Ensure necessary columns exist
        required_columns = ['last_purchase_date', 'first_purchase_date', 'total_revenue', 'frequency']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert dates first to ensure proper subtraction
        self.data['last_purchase_date'] = pd.to_datetime(self.data['last_purchase_date'])
        self.data['first_purchase_date'] = pd.to_datetime(self.data['first_purchase_date'])
        
        # Create a Series of current_date with same index as the dataframe
        current_date = pd.Series(pd.Timestamp.today(), index=self.data.index)
        
        # Now calculate the differences
        self.data['recency'] = (current_date - self.data['last_purchase_date']).dt.days
        self.data['customer_age_days'] = (current_date - self.data['first_purchase_date']).dt.days

        # Handle division by zero
        self.data['avg_transaction_value'] = (self.data['total_revenue'] /
                                            self.data['frequency'].replace(0, np.nan))

        # Replace infinities resulting from division with NaN
        self.data['avg_transaction_value'] = self.data['avg_transaction_value'].replace([np.inf, -np.inf], np.nan)

    def _clean_monetary_values(self):
        """Handle monetary value cleaning and outliers"""
        for col in ['frequency', 'monetary', 'avg_transaction_value', 'total_revenue']:
            if col in self.data.columns:
                # Remove negative values
                self.data = self.data[self.data[col] >= 0]

                # Handle outliers using IQR method
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                self.data = self.data[
                    (self.data[col] >= lower_bound) &
                    (self.data[col] <= upper_bound)
                ]

    def _clean_categorical_features(self):
        """Clean and encode categorical features"""
        categorical_features = {
            'has_online_purchases': 0,
            'has_store_purchases': 0,
            'is_loyalty_member': 0,
            'loyalty_points': 0,
            'sms_active': 0,
            'email_active': 0
        }

        for col, default in categorical_features.items():
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(default).astype(int)

    def _validate_data_quality(self):
        """Validate data quality and create quality flags"""
        # Create quality flags
        self.quality_flags = pd.DataFrame(index=self.data.index)

        # Define validation rules
        validations = {
            'valid_frequency': self.data['frequency'] >= self.config['MIN_FREQUENCY'],
            'valid_recency': self.data['recency'] >= 0,
            'valid_monetary': self.data['avg_transaction_value'] > 0,
            'valid_dates': self.data['last_purchase_date'] >= self.data['first_purchase_date']
        }

        # Apply validations
        for flag_name, condition in validations.items():
            self.quality_flags[flag_name] = condition

        # Overall validation
        self.quality_flags['overall_valid'] = self.quality_flags.all(axis=1)

    def _generate_quality_report(self):
        """Generate final data quality report"""
        report = {
            'record_count': len(self.data),
            'metrics': {
                'frequency_mean': self.data['frequency'].mean(),
                'recency_mean': self.data['recency'].mean(),
                'monetary_mean': self.data['avg_transaction_value'].mean()
            },
            'quality_flags': {
                col: self.quality_flags[col].mean() * 100
                for col in self.quality_flags.columns
            }
        }

        print("\nQuality Report:")
        print(f"Records processed: {report['record_count']:,}")
        print("\nKey Metrics:")
        for metric, value in report['metrics'].items():
            print(f"{metric}: {value:.2f}")
        print("\nQuality Flags (% passing):")
        for flag, pct in report['quality_flags'].items():
            print(f"{flag}: {pct:.1f}%")

    def get_processed_data(self):
        """Return the processed DataFrame"""
        return self.data.copy()