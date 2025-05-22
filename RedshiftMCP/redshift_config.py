
"""
AWS Redshift Configuration

This file contains all configuration for connecting to Redshift.
Sensitive credentials can be overridden using environment variables in .env file.
"""

# AWS Region configuration
DEFAULT_AWS_REGION = "us-east-1"

# Redshift connection details
REDSHIFT_CLUSTER = "sts-use1-mcp-poc-rs-wg.892551050452.us-east-1.redshift-serverless.amazonaws.com"
REDSHIFT_PORT = 5439
REDSHIFT_DATABASE = "dev"
REDSHIFT_USER = ""

# Redshift password - can be overridden by REDSHIFT_PASSWORD environment variable
# For security, consider setting this in .env file instead of here
REDSHIFT_PASSWORD = ""

# AWS Secrets Manager configuration
# Secret name - can be overridden by SECRET_NAME environment variable
SECRETS_MANAGER_SECRET_NAME = ""

# Query configuration
DEFAULT_QUERY_TIMEOUT = 300  # 5 minutes timeout for queries

# Connection preferences
PREFER_SECRETS_MANAGER = True  # Try Secrets Manager first before direct credentials
AUTO_RETRY_CONNECTION = True   # Retry connection if initial attempt fails