#!/usr/bin/env python3
"""
AWS Redshift MCP Server

A Model Context Protocol (MCP) server that provides AI assistants access to AWS Redshift
for querying and exploring data warehouses.

This implementation follows the AWS MCP Server patterns and standards.
"""
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any, Union

import boto3
import redshift_connector
from botocore.exceptions import ClientError
from loguru import logger
from mcp.server.fastmcp import FastMCP

# Configure logging
log_level = os.environ.get("FASTMCP_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=log_level)
logger.add("redshift_mcp_server.log", rotation="10 MB", retention="1 week", level="DEBUG")

logger.info(f"Starting AWS Redshift MCP Server (log level: {log_level})")

# Create the MCP server
mcp = FastMCP("AWS Redshift MCP Server")


# Note: Resource schemas are not supported in this version of FastMCP
# We'll use a simpler approach for resources

# Global state
class RedshiftState:
    """State management for the Redshift MCP server"""

    def __init__(self):
        # AWS clients
        self.aws_session = None
        self.redshift_client = None
        self.redshift_data_client = None
        self.secretsmanager_client = None

        # Redshift connection
        self.connection = None
        self.conn_details = None

        # Load configuration with environment variable overrides
        self._load_configuration()

        # Resources
        self.connection_resource = {
            "cluster": "Not connected",
            "database": "Not connected",
            "port": 5439,
            "region": self.default_region,
            "status": "Not connected",
            "connected": False
        }
        self.schema_resources = []
        self.table_resources = []
        self.common_query_resources = [
            {
                "name": "Count rows by schema",
                "description": "Count the number of rows in each table of a schema",
                "sql": """
SELECT 
    tables.table_name,
    tbl_rows.n_live_tup AS row_count
FROM 
    information_schema.tables
JOIN
    pg_catalog.pg_stat_user_tables tbl_rows 
    ON tables.table_name = tbl_rows.relname
WHERE 
    tables.table_schema = '{schema_name}'
ORDER BY 
    tbl_rows.n_live_tup DESC;
""",
                "tags": ["metadata", "schema", "count", "size"]
            },
            {
                "name": "Table storage analysis",
                "description": "Analyze storage usage for tables in a schema",
                "sql": """
SELECT
    trim(pgdb.datname) AS db_name,
    trim(pgn.nspname) AS schema_name,
    trim(pgc.relname) AS table_name,
    pgc.reltuples AS row_count,
    pg_size_pretty(pg_relation_size(pgc.oid)) AS table_size,
    pg_size_pretty(pg_total_relation_size(pgc.oid)) AS total_size
FROM pg_class pgc
JOIN pg_namespace pgn ON pgc.relnamespace = pgn.oid
JOIN pg_database pgdb ON pgdb.datname = current_database()
WHERE pgc.relkind = 'r' 
    AND pgn.nspname = '{schema_name}'
ORDER BY pg_total_relation_size(pgc.oid) DESC;
""",
                "tags": ["storage", "size", "schema", "optimization"]
            },
            {
                "name": "Column data profiling",
                "description": "Profile data distribution in a specific column",
                "sql": """
SELECT
    {column_name},
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {schema_name}.{table_name}), 2) as percentage
FROM
    {schema_name}.{table_name}
GROUP BY
    {column_name}
ORDER BY
    frequency DESC
LIMIT 20;
""",
                "tags": ["profiling", "column", "distribution", "analysis"]
            },
            {
                "name": "Find NULL values",
                "description": "Find NULL values in a table",
                "sql": """
SELECT 
    column_name,
    COUNT(*) - COUNT({column_name}) AS null_count,
    ROUND((COUNT(*) - COUNT({column_name})) * 100.0 / COUNT(*), 2) AS null_percentage
FROM 
    {schema_name}.{table_name}
GROUP BY 
    column_name
HAVING 
    COUNT(*) - COUNT({column_name}) > 0
ORDER BY 
    null_count DESC;
""",
                "tags": ["data quality", "null", "missing values"]
            },
            {
                "name": "Table join relationships",
                "description": "Find potential join relationships between tables",
                "sql": """
SELECT
    pgnsrc.nspname AS source_schema,
    pgsrc.relname AS source_table,
    pgndest.nspname AS referenced_schema,
    pgdest.relname AS referenced_table,
    pgc.conname AS constraint_name,
    pg_get_constraintdef(pgc.oid) AS constraint_definition
FROM
    pg_constraint pgc
JOIN 
    pg_class pgsrc ON pgc.conrelid = pgsrc.oid
JOIN 
    pg_namespace pgnsrc ON pgsrc.relnamespace = pgnsrc.oid
JOIN 
    pg_class pgdest ON pgc.confrelid = pgdest.oid
JOIN 
    pg_namespace pgndest ON pgdest.relnamespace = pgndest.oid
WHERE 
    pgc.contype = 'f'
    AND pgnsrc.nspname = '{schema_name}'
ORDER BY
    source_schema, source_table;
""",
                "tags": ["relationships", "foreign keys", "joins", "schema"]
            }
        ]

        # Register initial resources
        self._register_initial_resources()

    def _load_configuration(self):
        """Load configuration from redshift_config.py with environment variable overrides"""
        try:
            # Import configuration from redshift_config.py
            from redshift_config import (
                DEFAULT_AWS_REGION,
                REDSHIFT_CLUSTER,
                REDSHIFT_PORT,
                REDSHIFT_DATABASE,
                REDSHIFT_USER,
                REDSHIFT_PASSWORD,
                SECRETS_MANAGER_SECRET_NAME,
                DEFAULT_QUERY_TIMEOUT,
                PREFER_SECRETS_MANAGER,
                AUTO_RETRY_CONNECTION
            )

            # Set configuration values with environment variable overrides
            self.default_region = os.environ.get("AWS_REGION", DEFAULT_AWS_REGION)
            self.default_timeout = int(os.environ.get("QUERY_TIMEOUT", str(DEFAULT_QUERY_TIMEOUT)))

            # Redshift connection details
            self.static_redshift_cluster = REDSHIFT_CLUSTER
            self.static_redshift_port = REDSHIFT_PORT
            self.static_redshift_database = REDSHIFT_DATABASE
            self.static_redshift_user = REDSHIFT_USER

            # Sensitive credentials can be overridden by environment variables
            self.static_redshift_password = os.environ.get("REDSHIFT_PASSWORD", REDSHIFT_PASSWORD)
            self.static_secret_name = os.environ.get("SECRET_NAME", SECRETS_MANAGER_SECRET_NAME)

            # Connection preferences
            self.prefer_secrets_manager = PREFER_SECRETS_MANAGER
            self.auto_retry_connection = AUTO_RETRY_CONNECTION

            logger.info("Loaded configuration from redshift_config.py")
            logger.debug(f"Configuration: region={self.default_region}, cluster={self.static_redshift_cluster}")

        except ImportError as e:
            logger.warning(f"Could not import redshift_config.py: {e}")
            logger.warning("Using environment variables and defaults")

            # Fall back to environment variables and defaults
            self.default_region = os.environ.get("AWS_REGION", "us-east-1")
            self.default_timeout = int(os.environ.get("QUERY_TIMEOUT", "300"))

            self.static_redshift_cluster = os.environ.get("REDSHIFT_CLUSTER", "")
            self.static_redshift_port = int(os.environ.get("REDSHIFT_PORT", "5439"))
            self.static_redshift_database = os.environ.get("REDSHIFT_DATABASE", "")
            self.static_redshift_user = os.environ.get("REDSHIFT_USER", "")
            self.static_redshift_password = os.environ.get("REDSHIFT_PASSWORD", "")
            self.static_secret_name = os.environ.get("SECRET_NAME", "")

            self.prefer_secrets_manager = True
            self.auto_retry_connection = True

    def _register_initial_resources(self):
        """Register the initial resources"""
        # Note: Using simplified resource registration for current FastMCP version
        # Resources will be available as tools instead of formal resources
        pass


# Initialize state
state = RedshiftState()


# Helper functions
def get_aws_session(profile: str = None, region: str = None):
    """Get AWS session with optional profile and region"""
    try:
        logger.debug(f"Creating AWS session with profile={profile}, region={region or state.default_region}")
        if profile:
            state.aws_session = boto3.Session(profile_name=profile, region_name=region or state.default_region)
        else:
            state.aws_session = boto3.Session(region_name=region or state.default_region)

        logger.debug("Successfully created boto3 session")
        return state.aws_session
    except Exception as e:
        logger.error(f"Error creating AWS session: {str(e)}")
        raise e


def get_clients(profile: str = None, region: str = None):
    """Initialize or get AWS service clients"""
    if not state.aws_session:
        logger.debug("No existing session found, creating a new one")
        state.aws_session = get_aws_session(profile, region)

    if not state.redshift_client:
        logger.debug("Initializing Redshift client")
        state.redshift_client = state.aws_session.client('redshift')

    if not state.redshift_data_client:
        logger.debug("Initializing Redshift Data API client")
        state.redshift_data_client = state.aws_session.client('redshift-data')

    if not state.secretsmanager_client:
        logger.debug("Initializing Secrets Manager client")
        state.secretsmanager_client = state.aws_session.client('secretsmanager')

    return state.redshift_client, state.redshift_data_client, state.secretsmanager_client


def connect_to_redshift(conn_details):
    """Connect to Redshift using the provided connection details"""
    try:
        logger.debug(f"Connecting to Redshift cluster: {conn_details.get('cluster_identifier')}")

        # Close existing connection if any
        if state.connection:
            try:
                # Different connection objects have different ways to check if closed
                if hasattr(state.connection, 'closed') and not state.connection.closed:
                    state.connection.close()
                elif hasattr(state.connection, 'close'):
                    state.connection.close()
                logger.debug("Closed existing Redshift connection")
            except Exception as e:
                logger.debug(f"Error closing existing connection: {e}")

        # Connect using redshift_connector
        state.connection = redshift_connector.connect(
            host=conn_details.get('host'),
            database=conn_details.get('database'),
            user=conn_details.get('username'),
            password=conn_details.get('password'),
            port=int(conn_details.get('port', 5439))
        )

        # Enable autocommit
        state.connection.autocommit = True
        logger.debug("Successfully connected to Redshift")

        # Update connection resource
        state.connection_resource = {
            "cluster": conn_details.get('host'),
            "database": conn_details.get('database'),
            "port": conn_details.get('port', 5439),
            "region": state.default_region,
            "status": "Connected",
            "connected": True
        }

        # Store connection details for future use
        state.conn_details = conn_details

        # Load schemas and tables for resources
        load_schema_resources()

        return True
    except Exception as e:
        logger.error(f"Error connecting to Redshift: {str(e)}")

        # Update connection resource with error status
        state.connection_resource["status"] = f"Connection error: {str(e)}"
        state.connection_resource["connected"] = False

        raise e


def load_schema_resources():
    """Load schema and table resources from Redshift"""
    if not state.connection:
        logger.debug("Cannot load schema resources - no connection")
        return

    try:
        # Clear existing resources
        state.schema_resources = []
        state.table_resources = []

        # Get schemas
        cursor = state.connection.cursor()
        cursor.execute("""
            SELECT 
                schema_name,
                schema_owner,
                schema_type,
                pg_catalog.obj_description(oid, 'pg_namespace') as description
            FROM 
                information_schema.schemata s
            JOIN
                pg_catalog.pg_namespace n ON s.schema_name = n.nspname
            WHERE 
                schema_name NOT LIKE 'pg_%' AND
                schema_name NOT IN ('information_schema')
            ORDER BY 
                schema_name
        """)

        schemas = cursor.fetchall()

        # Add schemas to resources
        for schema in schemas:
            schema_name = schema[0]
            owner = schema[1]
            schema_type = schema[2]
            description = schema[3]

            schema_resource = {
                "name": schema_name,
                "owner": owner,
                "type": schema_type,
                "description": description
            }

            state.schema_resources.append(schema_resource)

            # Only load tables for a subset of schemas to avoid overwhelming the model
            if len(state.schema_resources) <= 5:  # Limit to first 5 schemas
                load_table_resources(schema_name)

        logger.debug(f"Loaded {len(state.schema_resources)} schemas as resources")

    except Exception as e:
        logger.error(f"Error loading schema resources: {str(e)}")


def load_table_resources(schema_name):
    """Load table resources for a specific schema"""
    if not state.connection:
        return

    try:
        cursor = state.connection.cursor()

        # Get tables with metadata
        cursor.execute("""
            SELECT 
                t.table_name,
                t.table_type,
                pg_catalog.obj_description(pgc.oid, 'pg_class') as table_comment,
                reldiststyle
            FROM 
                information_schema.tables t
            JOIN 
                pg_catalog.pg_class pgc ON t.table_name = pgc.relname
            JOIN 
                pg_catalog.pg_namespace pgn ON pgc.relnamespace = pgn.oid AND pgn.nspname = t.table_schema
            WHERE 
                t.table_schema = %s
            ORDER BY 
                t.table_name
        """, (schema_name,))

        tables = cursor.fetchall()

        # Map distribution style codes to readable names
        dist_style_map = {0: 'EVEN', 1: 'KEY', 8: 'ALL', 9: 'AUTO(ALL)', 10: 'AUTO(EVEN)', 11: 'AUTO(KEY)'}

        # Add tables to resources (limit to first 10 tables per schema)
        for idx, table in enumerate(tables[:10]):
            table_name = table[0]
            table_type = table[1]
            table_comment = table[2]
            dist_style_code = table[3]
            dist_style = dist_style_map.get(dist_style_code, 'UNKNOWN')

            # Get sort keys
            cursor.execute("""
                SELECT
                    attname
                FROM
                    pg_attribute a
                JOIN
                    pg_class t ON a.attrelid = t.oid
                JOIN
                    pg_namespace n ON t.relnamespace = n.oid
                WHERE
                    n.nspname = %s
                    AND t.relname = %s
                    AND attsortkeyord > 0
                ORDER BY
                    attsortkeyord
            """, (schema_name, table_name))

            sort_keys = [row[0] for row in cursor.fetchall()]

            # Get column information (limit to 20 columns per table)
            cursor.execute("""
                SELECT 
                    c.column_name,
                    c.data_type,
                    pg_catalog.col_description(pgc.oid, c.ordinal_position) as column_comment
                FROM 
                    information_schema.columns c
                JOIN 
                    pg_catalog.pg_class pgc ON pgc.relname = c.table_name
                JOIN 
                    pg_catalog.pg_namespace pgn ON pgc.relnamespace = pgn.oid AND pgn.nspname = c.table_schema
                WHERE 
                    c.table_schema = %s AND c.table_name = %s
                ORDER BY 
                    c.ordinal_position
                LIMIT 20
            """, (schema_name, table_name))

            columns = []
            for col in cursor.fetchall():
                columns.append({
                    "name": col[0],
                    "type": col[1],
                    "description": col[2]
                })

            table_resource = {
                "schema": schema_name,
                "name": table_name,
                "type": table_type,
                "description": table_comment,
                "columns": columns,
                "distribution_style": dist_style,
                "sort_keys": sort_keys
            }

            state.table_resources.append(table_resource)

        logger.debug(f"Loaded {len(tables[:10])} tables from schema {schema_name} as resources")

    except Exception as e:
        logger.error(f"Error loading table resources for schema {schema_name}: {str(e)}")


def format_size(size_bytes):
    """Format size in bytes to a human-readable string"""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_as_table(columns, rows):
    """Format query results as an ASCII table"""
    if not rows:
        return "No rows returned."

    # Convert all values to strings and calculate column widths
    str_rows = []
    col_widths = [len(str(col)) for col in columns]

    for row in rows:
        str_row = [str(val) if val is not None else 'NULL' for val in row]
        str_rows.append(str_row)
        for i, val in enumerate(str_row):
            col_widths[i] = max(col_widths[i], len(val))

    # Format the header
    header = " | ".join(col.ljust(col_widths[i]) for i, col in enumerate(columns))
    separator = "-+-".join("-" * width for width in col_widths)

    # Format the rows
    formatted_rows = [header, separator]
    for row in str_rows:
        formatted_rows.append(" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))

    return "\n".join(formatted_rows)


# MCP Tools
@mcp.tool()
def initialize_aws_session(profile: str = None, region: str = None) -> str:
    """Initialize AWS session with optional profile and region

    Args:
        profile: AWS CLI profile name (e.g., 'default' or SSO profile)
        region: AWS region (e.g., 'us-east-1')
    """
    try:
        logger.debug(f"Tool called: initialize_aws_session(profile={profile}, region={region})")
        get_clients(profile, region)

        # Verify that session is working by checking AWS account ID
        sts_client = state.aws_session.client('sts')
        caller_identity = sts_client.get_caller_identity()
        account_id = caller_identity.get('Account', 'Unknown')
        user_arn = caller_identity.get('Arn', 'Unknown')

        logger.debug(f"Successfully initialized AWS session as: {user_arn}")
        return f"Successfully initialized AWS session with {profile or 'default credentials'} in {region or state.default_region}\nConnected to AWS Account: {account_id}\nUser ARN: {user_arn}"
    except Exception as e:
        logger.error(f"Error initializing AWS session: {str(e)}")
        return f"Error initializing AWS session: {str(e)}"


@mcp.tool()
def connect_with_static_resources() -> str:
    """Connect to Redshift using predefined static connection details

    This uses the static Redshift cluster and credentials that are configured in redshift_config.py.
    No parameters needed - automatically connects to the preconfigured cluster.
    """
    if not state.aws_session:
        try:
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        # Try to connect using the static secret first if configured and preferred
        if state.static_secret_name and state.prefer_secrets_manager:
            logger.debug(f"Retrieving static secret: {state.static_secret_name}")
            try:
                secret_response = state.secretsmanager_client.get_secret_value(SecretId=state.static_secret_name)
                secret_string = secret_response['SecretString']
                secret_data = json.loads(secret_string)

                # Extract connection details from the secret
                conn_details = {
                    'host': secret_data.get('host', state.static_redshift_cluster),
                    'port': secret_data.get('port', state.static_redshift_port),
                    'database': secret_data.get('dbname') or secret_data.get('database',
                                                                             state.static_redshift_database),
                    'username': secret_data.get('username') or secret_data.get('user', state.static_redshift_user),
                    'password': secret_data.get('password', state.static_redshift_password),
                    'cluster_identifier': secret_data.get('dbClusterIdentifier', state.static_redshift_cluster)
                }

                # Connect to Redshift
                connect_to_redshift(conn_details)

                return f"Successfully connected to Redshift cluster {conn_details['cluster_identifier']} using static secret configuration"

            except Exception as secret_e:
                logger.warning(f"Could not connect using static secret: {str(secret_e)}")
                if not state.auto_retry_connection:
                    return f"Failed to connect using secret: {str(secret_e)}"
                logger.warning("Falling back to direct static credentials...")

        # Fall back to direct static credentials if secret access fails or isn't configured
        if not state.static_redshift_cluster or not state.static_redshift_database or not state.static_redshift_user:
            return "Static connection details not configured. Please set connection details in redshift_config.py or provide environment variables."

        conn_details = {
            'host': state.static_redshift_cluster,
            'port': state.static_redshift_port,
            'database': state.static_redshift_database,
            'username': state.static_redshift_user,
            'password': state.static_redshift_password,
            'cluster_identifier': state.static_redshift_cluster
        }

        # Connect to Redshift
        connect_to_redshift(conn_details)

        return f"Successfully connected to static Redshift cluster {state.static_redshift_cluster} using direct credentials"
    except Exception as e:
        logger.error(f"Error connecting with static resources: {str(e)}")
        return f"Error connecting with static resources: {str(e)}"


@mcp.tool()
def connect_with_secret(secret_id: str) -> str:
    """Connect to Redshift using a secret stored in AWS Secrets Manager

    Args:
        secret_id: The ID of the secret in AWS Secrets Manager
    """
    if not state.secretsmanager_client:
        try:
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        logger.debug(f"Retrieving secret: {secret_id}")
        secret_response = state.secretsmanager_client.get_secret_value(SecretId=secret_id)
        secret_string = secret_response['SecretString']
        secret_data = json.loads(secret_string)

        # Extract connection details from the secret
        conn_details = {
            'host': secret_data.get('host', 'localhost'),
            'port': secret_data.get('port', 5439),
            'database': secret_data.get('dbname') or secret_data.get('database', 'dev'),
            'username': secret_data.get('username') or secret_data.get('user'),
            'password': secret_data.get('password'),
            'cluster_identifier': secret_data.get('dbClusterIdentifier', secret_id)
        }

        # Connect to Redshift
        connect_to_redshift(conn_details)

        return f"Successfully connected to Redshift cluster {conn_details['cluster_identifier']} using secret {secret_id}"
    except Exception as e:
        logger.error(f"Error connecting with secret: {str(e)}")
        return f"Error connecting with secret: {str(e)}"


@mcp.tool()
def connect_with_credentials(
        host: str,
        database: str,
        username: str,
        password: str,
        port: int = 5439,
        cluster_identifier: str = None
) -> str:
    """Connect to Redshift using direct credentials

    Args:
        host: Redshift endpoint hostname
        database: Database name
        username: Database username
        password: Database password
        port: Database port (default: 5439)
        cluster_identifier: Optional name to identify the cluster
    """
    try:
        logger.debug(f"Connecting to Redshift at {host}:{port}/{database} as {username}")

        # Prepare connection details
        conn_details = {
            'host': host,
            'port': port,
            'database': database,
            'username': username,
            'password': password,
            'cluster_identifier': cluster_identifier or host
        }

        # Connect to Redshift
        connect_to_redshift(conn_details)

        return f"Successfully connected to Redshift at {host}:{port}/{database}"
    except Exception as e:
        logger.error(f"Error connecting with credentials: {str(e)}")
        return f"Error connecting with credentials: {str(e)}"


@mcp.tool()
def list_clusters() -> str:
    """List all Redshift clusters in the current AWS account"""
    if not state.redshift_client:
        try:
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        logger.debug("Listing Redshift clusters")
        response = state.redshift_client.describe_clusters()
        clusters = response.get('Clusters', [])

        if not clusters:
            return "No Redshift clusters found in this AWS account."

        results = [f"Found {len(clusters)} Redshift clusters:"]

        for cluster in clusters:
            cluster_id = cluster.get('ClusterIdentifier', 'Unknown')
            status = cluster.get('ClusterStatus', 'Unknown')
            node_type = cluster.get('NodeType', 'Unknown')
            node_count = cluster.get('NumberOfNodes', 0)
            db_name = cluster.get('DBName', 'Unknown')
            endpoint = cluster.get('Endpoint', {})
            address = endpoint.get('Address', 'Unknown')
            port = endpoint.get('Port', 'Unknown')

            results.append(f"- {cluster_id}: {status}, {node_count}x {node_type} nodes")
            results.append(f"  Database: {db_name}")
            results.append(f"  Endpoint: {address}:{port}")

        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error listing clusters: {str(e)}")
        return f"Error listing clusters: {str(e)}"


@mcp.tool()
def list_schemas() -> str:
    """List all schemas in the connected Redshift database"""
    if not state.connection:
        return "Not connected to Redshift. Please connect first using connect_with_credentials or connect_with_secret."

    try:
        logger.debug("Listing schemas in Redshift database")
        cursor = state.connection.cursor()
        cursor.execute("""
            SELECT 
                schema_name,
                schema_owner,
                schema_type
            FROM 
                information_schema.schemata
            WHERE 
                schema_name NOT LIKE 'pg_%' AND
                schema_name NOT IN ('information_schema')
            ORDER BY 
                schema_name
        """)

        schemas = cursor.fetchall()

        if not schemas:
            return "No user-defined schemas found in the database."

        results = [f"Found {len(schemas)} schemas:"]

        for schema in schemas:
            schema_name = schema[0]
            owner = schema[1]
            schema_type = schema[2]

            results.append(f"- {schema_name} (Owner: {owner}, Type: {schema_type})")

        cursor.close()

        # Refresh schema resources
        load_schema_resources()

        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error listing schemas: {str(e)}")
        return f"Error listing schemas: {str(e)}"


@mcp.tool()
def list_tables(schema: str = 'public') -> str:
    """List all tables in a specified schema

    Args:
        schema: Schema name (default: public)
    """
    if not state.connection:
        return "Not connected to Redshift. Please connect first using connect_with_credentials or connect_with_secret."

    try:
        logger.debug(f"Listing tables in schema: {schema}")
        cursor = state.connection.cursor()

        # Query for tables with additional metadata
        cursor.execute("""
            SELECT 
                t.table_name,
                t.table_type,
                pg_catalog.obj_description(pgc.oid, 'pg_class') as table_comment,
                pg_catalog.pg_table_size(pgc.oid) as table_size_bytes
            FROM 
                information_schema.tables t
            JOIN 
                pg_catalog.pg_class pgc ON t.table_name = pgc.relname
            JOIN 
                pg_catalog.pg_namespace pgn ON pgc.relnamespace = pgn.oid AND pgn.nspname = t.table_schema
            WHERE 
                t.table_schema = %s
            ORDER BY 
                t.table_name
        """, (schema,))

        tables = cursor.fetchall()

        if not tables:
            return f"No tables found in schema '{schema}'."

        results = [f"Found {len(tables)} tables in schema '{schema}':"]

        for table in tables:
            table_name = table[0]
            table_type = table[1]
            table_comment = table[2] or 'No comment'
            # Convert size to human-readable format
            size_bytes = table[3] or 0
            size_display = format_size(size_bytes)

            results.append(f"- {table_name} ({table_type}, {size_display})")
            results.append(f"  Comment: {table_comment}")

        cursor.close()

        # Refresh table resources for this schema
        load_table_resources(schema)

        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}")
        return f"Error listing tables: {str(e)}"


@mcp.tool()
def describe_table(schema: str, table: str) -> str:
    """Get detailed information about a specific table

    Args:
        schema: Schema name
        table: Table name
    """
    if not state.connection:
        return "Not connected to Redshift. Please connect first using connect_with_credentials or connect_with_secret."

    try:
        logger.debug(f"Describing table: {schema}.{table}")
        cursor = state.connection.cursor()

        # Get table comment
        cursor.execute("""
            SELECT 
                pg_catalog.obj_description(pgc.oid, 'pg_class') as table_comment
            FROM 
                pg_catalog.pg_class pgc
            JOIN 
                pg_catalog.pg_namespace pgn ON pgc.relnamespace = pgn.oid
            WHERE 
                pgn.nspname = %s AND pgc.relname = %s
        """, (schema, table))

        table_comment_result = cursor.fetchone()
        table_comment = table_comment_result[0] if table_comment_result else 'No comment'

        # Get column information with comments
        cursor.execute("""
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                pg_catalog.col_description(pgc.oid, c.ordinal_position) as column_comment
            FROM 
                information_schema.columns c
            JOIN 
                pg_catalog.pg_class pgc ON pgc.relname = c.table_name
            JOIN 
                pg_catalog.pg_namespace pgn ON pgc.relnamespace = pgn.oid AND pgn.nspname = c.table_schema
            WHERE 
                c.table_schema = %s AND c.table_name = %s
            ORDER BY 
                c.ordinal_position
        """, (schema, table))

        columns = cursor.fetchall()

        if not columns:
            return f"Table '{schema}.{table}' not found or has no columns."

        # Get distribution and sort keys
        cursor.execute("""
            SELECT
                reldiststyle,
                attname as distkey,
                relkind,
                attsortkeyord
            FROM
                pg_class t
            JOIN pg_namespace n ON (n.oid = t.relnamespace)
            LEFT JOIN pg_attribute a ON (a.attrelid = t.oid)
            WHERE
                n.nspname = %s
                AND t.relname = %s
                AND (attsortkeyord > 0 OR attisdistkey IS TRUE)
            ORDER BY
                attsortkeyord
        """, (schema, table))

        dist_sort_info = cursor.fetchall()

        # Collect distribution style and keys
        dist_style_map = {0: 'EVEN', 1: 'KEY', 8: 'ALL', 9: 'AUTO(ALL)', 10: 'AUTO(EVEN)', 11: 'AUTO(KEY)'}
        dist_style = None
        dist_keys = []
        sort_keys = []

        for row in dist_sort_info:
            if dist_style is None:
                dist_style = dist_style_map.get(row[0], 'UNKNOWN')

            if row[1] and row[0] in (1, 11):  # If it's a distribution key
                dist_keys.append(row[1])

            if row[3] > 0:  # If it's a sort key
                sort_keys.append((row[1], row[3]))  # column name and sort key order

        # Sort the sort keys by order
        sort_keys.sort(key=lambda x: x[1])
        sort_key_names = [k[0] for k in sort_keys]

        # Build the result
        results = [f"Table: {schema}.{table}"]
        results.append(f"Comment: {table_comment}")

        # Add distribution and sort key information
        results.append(f"Distribution Style: {dist_style}")
        if dist_keys:
            results.append(f"Distribution Key(s): {', '.join(dist_keys)}")
        if sort_key_names:
            results.append(f"Sort Key(s): {', '.join(sort_key_names)}")

        # Add columns
        results.append("\nColumns:")
        for col in columns:
            col_name = col[0]
            col_type = col[1]
            nullable = "NULL" if col[2] == "YES" else "NOT NULL"
            default = f"DEFAULT {col[3]}" if col[3] else ""
            comment = col[4] or 'No comment'

            results.append(f"- {col_name} ({col_type} {nullable} {default})")
            results.append(f"  Comment: {comment}")

        # Get table statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as row_count
            FROM 
                {schema}.{table}
        """.format(schema=schema, table=table))

        row_count = cursor.fetchone()[0]
        results.append(f"\nRow Count: {row_count:,}")

        cursor.close()
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error describing table: {str(e)}")
        return f"Error describing table: {str(e)}"


@mcp.tool()
def execute_query(query: str, timeout: int = None) -> str:
    """Execute a SQL query on the connected Redshift database

    Args:
        query: SQL query to execute
        timeout: Query timeout in seconds (default: uses server configuration)
    """
    if not state.connection:
        return "Not connected to Redshift. Please connect first using connect_with_credentials or connect_with_secret."

    try:
        logger.debug(f"Executing query: {query}")
        cursor = state.connection.cursor()

        # Set statement timeout if provided
        if timeout:
            cursor.execute(f"SET statement_timeout TO {timeout * 1000}")
        else:
            cursor.execute(f"SET statement_timeout TO {state.default_timeout * 1000}")

        # Execute the query
        start_time = time.time()
        cursor.execute(query)
        execution_time = time.time() - start_time

        # Check if there are results to fetch
        if cursor.description:
            # Format column headers
            columns = [desc[0] for desc in cursor.description]

            # Fetch the results
            rows = cursor.fetchall()
            row_count = len(rows)

            # Limit results if there are too many
            if row_count > 1000:
                logger.debug(f"Query returned {row_count} rows, truncating to 1000")
                rows = rows[:1000]
                truncated = True
            else:
                truncated = False

            # Format the results as a table
            result = format_as_table(columns, rows)

            if truncated:
                result += f"\n\n(Showing 1000 of {row_count:,} rows)"

            # Add execution summary
            result += f"\n\nQuery executed in {execution_time:.2f} seconds, returned {row_count:,} rows."
        else:
            # For non-SELECT queries
            row_count = cursor.rowcount
            result = f"Query executed successfully in {execution_time:.2f} seconds."

            if row_count >= 0:
                result += f" Affected {row_count:,} rows."

        cursor.close()
        return result
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return f"Error executing query: {str(e)}"


@mcp.tool()
def search_tables(search_term: str, schema: str = None) -> str:
    """Search for tables containing the search term in their name or comment

    Args:
        search_term: Text to search for
        schema: Optional schema to limit the search
    """
    if not state.connection:
        return "Not connected to Redshift. Please connect first using connect_with_credentials or connect_with_secret."

    try:
        logger.debug(f"Searching for tables with term: {search_term}")
        cursor = state.connection.cursor()

        # Build the query
        query = """
            SELECT 
                t.table_schema,
                t.table_name,
                t.table_type,
                pg_catalog.obj_description(pgc.oid, 'pg_class') as table_comment
            FROM 
                information_schema.tables t
            JOIN 
                pg_catalog.pg_class pgc ON t.table_name = pgc.relname
            JOIN 
                pg_catalog.pg_namespace pgn ON pgc.relnamespace = pgn.oid AND pgn.nspname = t.table_schema
            WHERE 
                (t.table_name ILIKE %s OR 
                 pg_catalog.obj_description(pgc.oid, 'pg_class') ILIKE %s)
        """

        params = [f"%{search_term}%", f"%{search_term}%"]

        # Add schema filter if provided
        if schema:
            query += " AND t.table_schema = %s"
            params.append(schema)

        # Exclude system schemas
        query += """
            AND t.table_schema NOT LIKE 'pg_%' 
            AND t.table_schema != 'information_schema'
            ORDER BY t.table_schema, t.table_name
        """

        cursor.execute(query, params)
        tables = cursor.fetchall()

        if not tables:
            return f"No tables found matching '{search_term}'."

        results = [f"Found {len(tables)} tables matching '{search_term}':"]

        for table in tables:
            schema_name = table[0]
            table_name = table[1]
            table_type = table[2]
            table_comment = table[3] or 'No comment'

            results.append(f"- {schema_name}.{table_name} ({table_type})")
            results.append(f"  Comment: {table_comment}")

        cursor.close()
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error searching tables: {str(e)}")
        return f"Error searching tables: {str(e)}"


@mcp.tool()
def search_columns(search_term: str, schema: str = None) -> str:
    """Search for columns containing the search term in their name or comment

    Args:
        search_term: Text to search for
        schema: Optional schema to limit the search
    """
    if not state.connection:
        return "Not connected to Redshift. Please connect first using connect_with_credentials or connect_with_secret."

    try:
        logger.debug(f"Searching for columns with term: {search_term}")
        cursor = state.connection.cursor()

        # Build the query
        query = """
            SELECT 
                c.table_schema,
                c.table_name,
                c.column_name,
                c.data_type,
                pg_catalog.col_description(pgc.oid, c.ordinal_position) as column_comment
            FROM 
                information_schema.columns c
            JOIN 
                pg_catalog.pg_class pgc ON pgc.relname = c.table_name
            JOIN 
                pg_catalog.pg_namespace pgn ON pgc.relnamespace = pgn.oid AND pgn.nspname = c.table_schema
            WHERE 
                (c.column_name ILIKE %s OR 
                 pg_catalog.col_description(pgc.oid, c.ordinal_position) ILIKE %s)
        """

        params = [f"%{search_term}%", f"%{search_term}%"]

        # Add schema filter if provided
        if schema:
            query += " AND c.table_schema = %s"
            params.append(schema)

        # Exclude system schemas
        query += """
            AND c.table_schema NOT LIKE 'pg_%' 
            AND c.table_schema != 'information_schema'
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """

        cursor.execute(query, params)
        columns = cursor.fetchall()

        if not columns:
            return f"No columns found matching '{search_term}'."

        results = [f"Found {len(columns)} columns matching '{search_term}':"]

        for column in columns:
            schema_name = column[0]
            table_name = column[1]
            column_name = column[2]
            data_type = column[3]
            column_comment = column[4] or 'No comment'

            results.append(f"- {schema_name}.{table_name}.{column_name} ({data_type})")
            results.append(f"  Comment: {column_comment}")

        cursor.close()
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error searching columns: {str(e)}")
        return f"Error searching columns: {str(e)}"


@mcp.tool()
def get_connection_info() -> str:
    """Get current connection information"""
    info = state.connection_resource
    return f"""Connection Status:
- Cluster: {info['cluster']}
- Database: {info['database']}
- Port: {info['port']}
- Region: {info['region']}
- Status: {info['status']}
- Connected: {info['connected']}"""


@mcp.tool()
def get_common_queries() -> str:
    """Get list of common SQL queries for Redshift analysis"""
    results = ["Available common query templates:"]

    for query in state.common_query_resources:
        results.append(f"\n**{query['name']}**")
        results.append(f"Description: {query['description']}")
        results.append(f"Tags: {', '.join(query['tags'])}")
        results.append("SQL Template:")
        results.append("```sql")
        results.append(query['sql'].strip())
        results.append("```")

    return "\n".join(results)


@mcp.tool()
def get_table_sample(schema: str, table: str, limit: int = 10) -> str:
    """Get a sample of rows from a table

    Args:
        schema: Schema name
        table: Table name
        limit: Maximum number of rows to return (default: 10)
    """
    if not state.connection:
        return "Not connected to Redshift. Please connect first using connect_with_credentials or connect_with_secret."

    try:
        logger.debug(f"Getting sample from table: {schema}.{table} (limit: {limit})")

        # Validate limit to prevent excessive data return
        if limit > 100:
            limit = 100

        query = f"SELECT * FROM {schema}.{table} LIMIT {limit}"
        return execute_query(query)
    except Exception as e:
        logger.error(f"Error getting table sample: {str(e)}")
        return f"Error getting table sample: {str(e)}"


@mcp.prompt()
def example_prompt(query: str) -> str:
    return f"""You need to interact with AWS Redshift to answer this question:

{query}

First, check if we're connected to Redshift. If not, connect using either connect_with_static_resources(), connect_with_credentials(), or connect_with_secret().
Then, explore the database using list_schemas(), list_tables(), and describe_table() to understand the data structure.
You can search for tables and columns using search_tables() and search_columns().
To answer specific questions, formulate SQL queries and execute them with execute_query().

Example workflow:
1. Connect to Redshift (if not already connected)
2. Explore schemas and tables to find relevant data
3. Examine table structures using describe_table()
4. Formulate a SQL query to answer the question
5. Execute the query and interpret the results

Ensure your SQL queries follow Redshift's SQL syntax and best practices."""


# Try to load AWS environment on startup
if __name__ == "__main__":
    logger.info("Starting AWS Redshift MCP server...")
    logger.info(f"Default region: {state.default_region}")
    logger.info(f"Default timeout: {state.default_timeout} seconds")

    if state.static_redshift_cluster:
        logger.info(f"Static Redshift cluster configured: {state.static_redshift_cluster}")

    if state.static_secret_name:
        logger.info(f"Static secret name configured: {state.static_secret_name}")

    # Check if AWS credentials are available
    try:
        default_session = boto3.Session()
        # Just check if we can get the caller identity
        sts_client = default_session.client('sts')
        sts_client.get_caller_identity()
        logger.info("AWS credentials available for default session")

        # Try to connect with static resources
        try:
            # Get AWS clients
            if not state.aws_session:
                state.aws_session = default_session
                state.redshift_client = state.aws_session.client('redshift')
                state.redshift_data_client = state.aws_session.client('redshift-data')
                state.secretsmanager_client = state.aws_session.client('secretsmanager')

            # Try connecting with static resources
            if state.static_secret_name or state.static_redshift_cluster:
                logger.info("Attempting to connect with static resources at startup...")

                if state.static_secret_name:
                    try:
                        logger.info(f"Trying to connect using secret: {state.static_secret_name}")
                        connect_with_secret(state.static_secret_name)
                    except Exception as secret_err:
                        logger.warning(f"Failed to connect using secret: {str(secret_err)}")

                # If secret connection failed, try direct connection
                if not state.connection and state.static_redshift_cluster and state.static_redshift_database:
                    try:
                        logger.info(f"Trying to connect using direct credentials to: {state.static_redshift_cluster}")
                        connect_with_credentials(
                            host=state.static_redshift_cluster,
                            port=state.static_redshift_port,
                            database=state.static_redshift_database,
                            username=state.static_redshift_user,
                            password=state.static_redshift_password
                        )
                    except Exception as conn_err:
                        logger.warning(f"Failed to connect using direct credentials: {str(conn_err)}")
        except Exception as static_err:
            logger.warning(f"Error during static connection attempt: {str(static_err)}")
            logger.info("You will need to manually connect during the session")

    except Exception as e:
        logger.warning(f"No AWS credentials available for default session: {str(e)}")
        logger.info("You will need to initialize a session with a profile when using this server")

    # Initialize and run the server
    logger.info("Starting MCP server with stdio transport")
    mcp.run(transport="stdio")