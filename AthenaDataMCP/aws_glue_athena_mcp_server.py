#!/usr/bin/env python3
"""
MCP Server for AWS Glue Catalog, Athena and Lake Formation integration.
Hybrid implementation using boto3 and AWS Data Wrangler.
"""
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any

import boto3
import awswrangler as wr
from botocore.exceptions import ClientError
from loguru import logger
from mcp.server.fastmcp import FastMCP

# Configure detailed logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.add("mcp_server.log", rotation="10 MB", retention="1 week", level="DEBUG")

logger.debug("Starting AWS Glue/Athena MCP Server with hybrid boto3 and AWS Data Wrangler")

# Create an MCP server
mcp = FastMCP("AWS Glue/Athena MCP Server")

# Global variables for clients and session
session = None
glue_client = None
athena_client = None
lakeformation_client = None
s3_client = None

# Configuration
DEFAULT_REGION = "us-east-1"
DEFAULT_S3_OUTPUT = "s3://sts-use1-sample-data-athena-output/"  # Updated with user's bucket
DEFAULT_WORKGROUP = "primary"
DEFAULT_CTAS_APPROACH = True  # Use CTAS for better Athena performance
DEFAULT_TIMEOUT = 300  # 5 minutes timeout for Athena queries


def get_aws_session(profile: str = None, region: str = None):
    """Get AWS session with optional profile and region"""
    global session
    try:
        logger.debug(f"Creating AWS session with profile={profile}, region={region or DEFAULT_REGION}")
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region or DEFAULT_REGION)
        else:
            session = boto3.Session(region_name=region or DEFAULT_REGION)

        # Note: AWS Data Wrangler doesn't need explicit session config
        # We'll pass the session directly to each function call instead
        logger.debug("Successfully created boto3 session")

        return session
    except Exception as e:
        logger.error(f"Error creating AWS session: {str(e)}")
        raise e


def get_clients(profile: str = None, region: str = None):
    """Initialize or get AWS service clients"""
    global glue_client, athena_client, lakeformation_client, s3_client, session

    if not session:
        logger.debug("No existing session found, creating a new one")
        session = get_aws_session(profile, region)

    if not glue_client:
        logger.debug("Initializing Glue client")
        glue_client = session.client('glue')

    if not athena_client:
        logger.debug("Initializing Athena client")
        athena_client = session.client('athena')

    if not lakeformation_client:
        logger.debug("Initializing Lake Formation client")
        lakeformation_client = session.client('lakeformation')

    if not s3_client:
        logger.debug("Initializing S3 client")
        s3_client = session.client('s3')

    return glue_client, athena_client, lakeformation_client, s3_client


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
        sts_client = session.client('sts')
        caller_identity = sts_client.get_caller_identity()
        account_id = caller_identity.get('Account', 'Unknown')
        user_arn = caller_identity.get('Arn', 'Unknown')

        logger.debug(f"Successfully initialized AWS session as: {user_arn}")
        return f"Successfully initialized AWS session with {profile or 'default credentials'} in {region or DEFAULT_REGION}\nConnected to AWS Account: {account_id}\nUser ARN: {user_arn}"
    except Exception as e:
        logger.error(f"Error initializing AWS session: {str(e)}")
        return f"Error initializing AWS session: {str(e)}"


@mcp.tool()
def verify_aws_access() -> str:
    """Verify access to AWS Glue, Athena, and Lake Formation services"""
    logger.debug("Tool called: verify_aws_access()")

    if not session:
        try:
            logger.debug("No existing session, creating a new one")
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    results = []

    # Check AWS account info
    try:
        sts_client = session.client('sts')
        caller_identity = sts_client.get_caller_identity()
        account_id = caller_identity.get('Account', 'Unknown')
        user_arn = caller_identity.get('Arn', 'Unknown')
        results.append(f"AWS Account ID: {account_id}")
        results.append(f"User ARN: {user_arn}")
        results.append("")
    except Exception as e:
        results.append(f"Error getting AWS identity: {str(e)}")

    # Check Glue access
    results.append("Checking AWS Glue access:")
    try:
        glue_result = glue_client.get_catalog_import_status()
        results.append("✅ AWS Glue service is accessible")
    except Exception as e:
        if "AccessDeniedException" in str(e):
            results.append("❌ AWS Glue access denied. Check your IAM permissions.")
        else:
            results.append(f"❌ AWS Glue error: {str(e)}")

    # Check Athena access
    results.append("\nChecking AWS Athena access:")
    try:
        athena_result = athena_client.list_workgroups(MaxResults=10)
        workgroups = [wg['Name'] for wg in athena_result.get('WorkGroups', [])]
        results.append(f"✅ AWS Athena service is accessible")
        results.append(f"Available workgroups: {', '.join(workgroups) if workgroups else 'None found'}")
    except Exception as e:
        if "AccessDeniedException" in str(e):
            results.append("❌ AWS Athena access denied. Check your IAM permissions.")
        else:
            results.append(f"❌ AWS Athena error: {str(e)}")

    # Check Lake Formation access
    results.append("\nChecking AWS Lake Formation access:")
    try:
        lf_result = lakeformation_client.list_resources(MaxResults=10)
        resources = lf_result.get('ResourceInfoList', [])
        resources_count = len(resources)
        results.append(f"✅ AWS Lake Formation service is accessible")
        results.append(f"Resources registered with Lake Formation: {resources_count}")
    except Exception as e:
        if "AccessDeniedException" in str(e):
            results.append("❌ AWS Lake Formation access denied. Check your IAM permissions.")
        else:
            results.append(f"❌ AWS Lake Formation error: {str(e)}")

    # Check S3 access
    results.append("\nChecking AWS S3 access:")
    try:
        s3_result = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in s3_result['Buckets']]
        results.append(f"✅ AWS S3 service is accessible")
        results.append(f"Available buckets: {', '.join(buckets[:5]) + (', ...' if len(buckets) > 5 else '')}")

        # Check specific Athena output bucket
        output_bucket = DEFAULT_S3_OUTPUT.replace("s3://", "").split("/")[0]
        try:
            s3_client.head_bucket(Bucket=output_bucket)
            results.append(f"✅ Athena output bucket '{output_bucket}' is accessible")
        except Exception as s3_e:
            results.append(f"❌ Cannot access Athena output bucket '{output_bucket}': {str(s3_e)}")

    except Exception as e:
        if "AccessDeniedException" in str(e):
            results.append("❌ AWS S3 access denied. Check your IAM permissions.")
        else:
            results.append(f"❌ AWS S3 error: {str(e)}")

    # Check permissions
    results.append("\nPermissions check:")
    try:
        # Test IAM access
        iam_client = session.client('iam')
        try:
            iam_result = iam_client.get_user()
            results.append("✅ You have IAM user access")
        except Exception as iam_e:
            if "cannot get user" in str(iam_e).lower():
                results.append("ℹ️ You're using an IAM role (normal for SSO or assumed roles)")
            elif "AccessDenied" in str(iam_e):
                results.append("⚠️ Limited IAM access (expected for regular users)")
            else:
                results.append(f"ℹ️ IAM access note: {str(iam_e)}")
    except Exception as e:
        results.append(f"ℹ️ Could not check IAM permissions: {str(e)}")

    results.append("\nRecommended IAM permissions needed:")
    results.append("- glue:GetDatabases, glue:GetTables, glue:GetTable")
    results.append("- athena:StartQueryExecution, athena:GetQueryExecution, athena:GetQueryResults")
    results.append("- lakeformation:ListPermissions, lakeformation:ListResources")
    results.append("- s3:ListBucket, s3:GetObject, s3:PutObject (for Athena query results)")

    return "\n".join(results)


@mcp.tool()
def list_databases(catalog_id: str = None) -> str:
    """List all databases in the Glue Catalog

    Args:
        catalog_id: The AWS account ID of the Glue Catalog owner
    """
    logger.debug(f"Tool called: list_databases(catalog_id={catalog_id})")

    if not session:
        try:
            logger.debug("No existing session, creating a new one")
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        # First try with AWS Data Wrangler
        try:
            logger.debug("Using AWS Data Wrangler to list databases")

            # Handle the catalog_id parameter - ensure it's a proper string
            if catalog_id and catalog_id.lower() == 'null':
                catalog_id = None

            params = {}
            if catalog_id:
                params['catalog_id'] = catalog_id

            # Try with AWS Data Wrangler first
            databases_generator = wr.catalog.get_databases(**params)

            # Convert generator to list and extract database names
            databases = list(databases_generator)
            logger.debug(f"Found {len(databases)} databases using AWS Data Wrangler")

            if not databases:
                return "No databases found."

            # Format results
            results = [f"Found {len(databases)} databases:"]
            for db in databases:
                db_name = db.get('Name', 'Unknown')
                description = db.get('Description', 'No description')
                location = db.get('LocationUri', 'No location')
                results.append(f"- {db_name}: {description} (Location: {location})")

            return "\n".join(results)

        except Exception as wrangler_error:
            # If AWS Data Wrangler fails, fall back to boto3
            logger.warning(f"AWS Data Wrangler failed: {str(wrangler_error)}, falling back to boto3")

            # Fall back to direct boto3 call
            params = {}
            if catalog_id and catalog_id.lower() != 'null':
                params['CatalogId'] = catalog_id

            response = glue_client.get_databases(**params)

            databases = response.get('DatabaseList', [])
            results = [f"Found {len(databases)} databases:"]

            for db in databases:
                db_name = db.get('Name', 'Unknown')
                description = db.get('Description', 'No description')
                location = db.get('LocationUri', 'No location')
                results.append(f"- {db_name}: {description} (Location: {location})")

            # Handle pagination if needed
            next_token = response.get('NextToken')
            while next_token:
                params['NextToken'] = next_token
                response = glue_client.get_databases(**params)
                databases = response.get('DatabaseList', [])

                for db in databases:
                    db_name = db.get('Name', 'Unknown')
                    description = db.get('Description', 'No description')
                    location = db.get('LocationUri', 'No location')
                    results.append(f"- {db_name}: {description} (Location: {location})")

                next_token = response.get('NextToken')

            return "\n".join(results)

    except Exception as e:
        logger.error(f"Error listing databases: {str(e)}")
        return f"Error listing databases: {str(e)}"


@mcp.tool()
def list_tables(database_name: str, catalog_id: str = None) -> str:
    """List all tables in a Glue database

    Args:
        database_name: Name of the Glue database
        catalog_id: The AWS account ID of the Glue Catalog owner
    """
    logger.debug(f"Tool called: list_tables(database_name={database_name}, catalog_id={catalog_id})")

    if not session:
        try:
            logger.debug("No existing session, creating a new one")
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        # First try with AWS Data Wrangler
        try:
            logger.debug(f"Using AWS Data Wrangler to list tables in database {database_name}")

            # Handle the catalog_id parameter
            if catalog_id and catalog_id.lower() == 'null':
                catalog_id = None

            params = {
                'database': database_name
            }
            if catalog_id:
                params['catalog_id'] = catalog_id

            # Get tables as a generator and convert to list
            tables_generator = wr.catalog.get_tables(**params)
            tables = list(tables_generator)

            logger.debug(f"Found {len(tables)} tables using AWS Data Wrangler")

            if not tables:
                return f"No tables found in database '{database_name}'."

            # Format results
            results = [f"Found {len(tables)} tables in database '{database_name}':"]

            for table in tables:
                table_name = table.get('Name', 'Unknown')
                description = table.get('Description', 'No description')
                table_type = table.get('TableType', 'Unknown')

                # Check for table properties that indicate data size if available
                table_props = table.get('Parameters', {})
                size_info = ""
                if isinstance(table_props, dict):
                    if 'recordCount' in table_props:
                        size_info = f", Records: {table_props['recordCount']}"
                    elif 'numRows' in table_props:
                        size_info = f", Rows: {table_props['numRows']}"

                results.append(f"- {table_name}: {description} (Type: {table_type}{size_info})")

            return "\n".join(results)

        except Exception as wrangler_error:
            # If AWS Data Wrangler fails, fall back to boto3
            logger.warning(f"AWS Data Wrangler failed: {str(wrangler_error)}, falling back to boto3")

            # Use direct boto3 call
            params = {'DatabaseName': database_name}
            if catalog_id and catalog_id.lower() != 'null':
                params['CatalogId'] = catalog_id

            response = glue_client.get_tables(**params)

            tables = response.get('TableList', [])
            results = [f"Found {len(tables)} tables in database '{database_name}':"]

            for table in tables:
                table_name = table.get('Name', 'Unknown')
                description = table.get('Description', 'No description')
                table_type = table.get('TableType', 'Unknown')
                results.append(f"- {table_name}: {description} (Type: {table_type})")

            # Handle pagination if needed
            next_token = response.get('NextToken')
            while next_token:
                params['NextToken'] = next_token
                response = glue_client.get_tables(**params)
                tables = response.get('TableList', [])

                for table in tables:
                    table_name = table.get('Name', 'Unknown')
                    description = table.get('Description', 'No description')
                    table_type = table.get('TableType', 'Unknown')
                    results.append(f"- {table_name}: {description} (Type: {table_type})")

                next_token = response.get('NextToken')

            return "\n".join(results)

    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}")
        return f"Error listing tables: {str(e)}"


@mcp.tool()
def get_table_schema(database_name: str, table_name: str, catalog_id: str = None) -> str:
    """Get the schema of a Glue table including column descriptions

    Args:
        database_name: Name of the Glue database
        table_name: Name of the Glue table
        catalog_id: The AWS account ID of the Glue Catalog owner
    """
    logger.debug(
        f"Tool called: get_table_schema(database_name={database_name}, table_name={table_name}, catalog_id={catalog_id})")

    if not session:
        try:
            logger.debug("No existing session, creating a new one")
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        # Try with direct boto3 call for more reliability
        logger.debug(f"Using boto3 to get schema for {database_name}.{table_name}")

        params = {
            'DatabaseName': database_name,
            'Name': table_name
        }
        if catalog_id and catalog_id.lower() != 'null':
            params['CatalogId'] = catalog_id

        response = glue_client.get_table(**params)

        table = response.get('Table', {})
        storage_descriptor = table.get('StorageDescriptor', {})
        columns = storage_descriptor.get('Columns', [])

        # Build the result
        results = [f"Schema for {database_name}.{table_name}:"]

        # Add columns
        results.append("\nColumns:")
        if columns:
            for col in columns:
                col_name = col.get('Name', 'Unknown')
                col_type = col.get('Type', 'Unknown')
                comment = col.get('Comment', 'No comment')
                results.append(f"- {col_name} ({col_type}): {comment}")
        else:
            results.append("- No columns found")

        # Add partition keys
        partition_keys = table.get('PartitionKeys', [])
        if partition_keys:
            results.append("\nPartition Keys:")
            for col in partition_keys:
                col_name = col.get('Name', 'Unknown')
                col_type = col.get('Type', 'Unknown')
                comment = col.get('Comment', 'No comment')
                results.append(f"- {col_name} ({col_type}): {comment}")

        # Add storage format info
        sd = table.get('StorageDescriptor', {})
        input_format = sd.get('InputFormat', 'Not specified')
        if input_format:
            # Determine user-friendly format name
            format_name = "Unknown"
            if "parquet" in input_format.lower():
                format_name = "Parquet"
            elif "orc" in input_format.lower():
                format_name = "ORC"
            elif "text" in input_format.lower() or "csv" in input_format.lower():
                format_name = "CSV/Text"
            elif "json" in input_format.lower():
                format_name = "JSON"
            elif "avro" in input_format.lower():
                format_name = "Avro"

            results.append(f"\nStorage Format: {format_name}")

        # Add table properties
        table_properties = table.get('Parameters', {})
        if table_properties:
            results.append("\nTable Properties:")
            # Filter to the most useful properties
            important_props = [
                "classification", "compressionType", "recordCount",
                "averageRecordSize", "numRows", "numFiles",
                "totalSize", "timeToLiveMillis", "comment"
            ]

            # First show important properties
            for key in important_props:
                if key in table_properties:
                    results.append(f"- {key}: {table_properties[key]}")

            # Then show any others
            for key, value in table_properties.items():
                if key not in important_props:
                    results.append(f"- {key}: {value}")

        # Add location
        location = storage_descriptor.get('Location', 'Not specified')
        results.append(f"\nLocation: {location}")

        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        return f"Error getting table schema: {str(e)}"


@mcp.tool()
def check_lake_formation_permissions(database_name: str, table_name: str = None, catalog_id: str = None) -> str:
    """Check Lake Formation permissions for a database or table

    Args:
        database_name: Name of the Glue database
        table_name: Optional name of the Glue table
        catalog_id: The AWS account ID of the Glue Catalog owner
    """
    if not lakeformation_client:
        try:
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        # First, get the caller identity to know whose permissions to check
        sts_client = session.client('sts')
        caller_identity = sts_client.get_caller_identity()
        principal = caller_identity.get('Arn', '')

        # Set up params for the request
        params = {
            'Principal': {'DataLakePrincipalIdentifier': principal},
            'Resource': {}
        }

        if catalog_id and catalog_id.lower() != 'null':
            params['CatalogId'] = catalog_id

        # Set the resource type based on whether a table name was provided
        if table_name:
            params['Resource']['Table'] = {
                'DatabaseName': database_name,
                'Name': table_name
            }
            resource_str = f"table {database_name}.{table_name}"
        else:
            params['Resource']['Database'] = {
                'Name': database_name
            }
            resource_str = f"database {database_name}"

        # Get permissions
        response = lakeformation_client.list_permissions(**params)
        permissions = response.get('PrincipalResourcePermissions', [])

        if not permissions:
            return f"No permissions found for {resource_str}"

        results = [f"Permissions for {resource_str}:"]

        for perm in permissions:
            permission = perm.get('Permissions', [])
            permission_with_grant = perm.get('PermissionsWithGrantOption', [])

            results.append(f"- Permissions: {', '.join(permission)}")
            if permission_with_grant:
                results.append(f"- Permissions with grant option: {', '.join(permission_with_grant)}")

        # Handle pagination if needed
        next_token = response.get('NextToken')
        while next_token:
            params['NextToken'] = next_token
            response = lakeformation_client.list_permissions(**params)
            permissions = response.get('PrincipalResourcePermissions', [])

            for perm in permissions:
                permission = perm.get('Permissions', [])
                permission_with_grant = perm.get('PermissionsWithGrantOption', [])

                results.append(f"- Permissions: {', '.join(permission)}")
                if permission_with_grant:
                    results.append(f"- Permissions with grant option: {', '.join(permission_with_grant)}")

            next_token = response.get('NextToken')

        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error checking Lake Formation permissions: {str(e)}")
        return f"Error checking Lake Formation permissions: {str(e)}"


@mcp.tool()
def execute_athena_query(
        query: str,
        database: str,
        output_location: str = DEFAULT_S3_OUTPUT,
        workgroup: str = DEFAULT_WORKGROUP,
        wait_for_results: bool = True,
        max_wait_seconds: int = DEFAULT_TIMEOUT,
        use_ctas: bool = DEFAULT_CTAS_APPROACH
) -> str:
    """Execute an Athena query using AWS Data Wrangler

    Args:
        query: The SQL query to execute
        database: The database to use for the query
        output_location: S3 location for query results
        workgroup: Athena workgroup to use
        wait_for_results: Whether to wait for query completion
        max_wait_seconds: Maximum time to wait for results
        use_ctas: Whether to use CTAS approach for better performance
    """
    logger.debug(f"Tool called: execute_athena_query(db={database}, workgroup={workgroup}, wait={wait_for_results})")
    logger.debug(f"Query: {query}")

    if not session:
        try:
            logger.debug("No existing session, creating a new one")
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        # Check if we should wait for results or just submit
        if not wait_for_results:
            # Use boto3 directly to submit without waiting
            logger.debug("Submitting query without waiting for results")
            response = athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': database
                },
                ResultConfiguration={
                    'OutputLocation': output_location
                },
                WorkGroup=workgroup
            )

            query_execution_id = response['QueryExecutionId']
            logger.debug(f"Started Athena query with ID: {query_execution_id}")
            return f"Query submitted successfully. Execution ID: {query_execution_id}"

        # Try to use AWS Data Wrangler for queries where we want results
        try:
            logger.debug("Using AWS Data Wrangler to execute query with results")

            # Execute the query with AWS Data Wrangler (without setting timeout config)
            df = wr.athena.read_sql_query(
                sql=query,
                database=database,
                ctas_approach=use_ctas,
                s3_output=output_location,
                workgroup=workgroup,
                boto3_session=session  # Pass session explicitly
            )

            logger.debug(f"Query returned a DataFrame with shape: {df.shape}")

            # Check if DataFrame is empty
            if df.empty:
                return "Query executed successfully but returned no results."

            # Handle large result sets
            if len(df) > 1000:
                logger.debug(f"Large result set with {len(df)} rows, returning first 1000 rows")
                df = df.head(1000)
                result_text = df.to_string(index=False)
                return f"Query returned {len(df)} rows. Showing first 1000:\n\n{result_text}\n\n(Result truncated, query returned more rows)"
            else:
                # Format the DataFrame as string
                result_text = df.to_string(index=False)
                return result_text

        except Exception as wrangler_error:
            # If pandas mode fails, fall back to raw SQL with boto3
            logger.warning(f"AWS Data Wrangler pandas mode failed: {str(wrangler_error)}")
            logger.debug("Falling back to boto3 for query execution")

            # Start the query execution with boto3
            response = athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': database
                },
                ResultConfiguration={
                    'OutputLocation': output_location
                },
                WorkGroup=workgroup
            )

            query_execution_id = response['QueryExecutionId']
            logger.debug(f"Started Athena query with ID: {query_execution_id}")

            # Wait for the query to complete
            state = 'RUNNING'
            start_time = time.time()

            logger.debug("Waiting for query to complete...")
            while state in ['RUNNING', 'QUEUED'] and (time.time() - start_time) < max_wait_seconds:
                response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
                state = response['QueryExecution']['Status']['State']

                if state in ['RUNNING', 'QUEUED']:
                    time.sleep(2)  # Poll every 2 seconds

            # Check the final state
            if state == 'SUCCEEDED':
                logger.debug("Query succeeded, fetching results")
                # Get the results
                results_paginator = athena_client.get_paginator('get_query_results')
                results_iterator = results_paginator.paginate(
                    QueryExecutionId=query_execution_id,
                    PaginationConfig={'MaxItems': 1000}  # Limit to 1000 rows max
                )

                all_rows = []
                header_row = None

                for results_page in results_iterator:
                    rows = results_page['ResultSet']['Rows']

                    # The first row of the first page contains the column names
                    if not header_row and rows:
                        header_row = [col['VarCharValue'] for col in rows[0]['Data']]
                        all_rows.append(header_row)
                        rows = rows[1:]  # Skip the header for processing

                    for row in rows:
                        # Handle the various data types that can be returned
                        processed_row = []
                        for col in row['Data']:
                            # Each value will have one of these fields
                            if 'VarCharValue' in col:
                                processed_row.append(col['VarCharValue'])
                            else:
                                # Handle other Athena data types if necessary
                                processed_row.append("NULL")

                        all_rows.append(processed_row)

                # Format the results as a table
                if all_rows:
                    col_widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(all_rows[0]))]
                    formatted_rows = []

                    # Add header
                    header = " | ".join(str(all_rows[0][i]).ljust(col_widths[i]) for i in range(len(all_rows[0])))
                    formatted_rows.append(header)

                    # Add separator
                    separator = "-+-".join("-" * col_widths[i] for i in range(len(all_rows[0])))
                    formatted_rows.append(separator)

                    # Add data rows
                    for row in all_rows[1:]:
                        formatted_row = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
                        formatted_rows.append(formatted_row)

                    result_text = "\n".join(formatted_rows)

                    # Add a note if results were truncated
                    if len(all_rows) >= 1000:
                        result_text += "\n\n(Results truncated, query returned more rows)"

                    return result_text
                else:
                    return "Query executed successfully but returned no results."
            elif state == 'FAILED':
                error_info = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                logger.error(f"Query failed: {error_info}")
                return f"Query failed: {error_info}"
            elif state == 'CANCELLED':
                logger.warning("Query was cancelled")
                return "Query was cancelled"
            else:
                logger.warning(f"Query timed out after {max_wait_seconds} seconds. Current state: {state}")
                return f"Query timed out after {max_wait_seconds} seconds. Current state: {state}. Execution ID: {query_execution_id}"
    except Exception as e:
        logger.error(f"Error executing Athena query: {str(e)}")
        return f"Error executing Athena query: {str(e)}"


@mcp.tool()
def get_query_execution_status(query_execution_id: str) -> str:
    """Get the status of a running Athena query

    Args:
        query_execution_id: The ID of the query execution to check
    """
    if not athena_client:
        try:
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        execution = response['QueryExecution']

        state = execution['Status']['State']
        submission_time = execution['Status'].get('SubmissionDateTime', 'Unknown')
        completion_time = execution['Status'].get('CompletionDateTime', 'Not completed')

        if state == 'FAILED':
            error_info = execution['Status'].get('StateChangeReason', 'Unknown error')
            return f"Query {query_execution_id} failed: {error_info}"

        # Get query statistics if available
        stats = {}
        if 'Statistics' in execution:
            stats = {
                'Data scanned': f"{execution['Statistics'].get('DataScannedInBytes', 0) / (1024 * 1024):.2f} MB",
                'Execution time': f"{execution['Statistics'].get('EngineExecutionTimeInMillis', 0) / 1000:.2f} seconds",
                'Total time': f"{execution['Statistics'].get('TotalExecutionTimeInMillis', 0) / 1000:.2f} seconds"
            }

        results = [
            f"Query {query_execution_id} status: {state}",
            f"Submitted: {submission_time}",
            f"Completed: {completion_time if state in ['SUCCEEDED', 'FAILED', 'CANCELLED'] else 'Not completed'}"
        ]

        if stats:
            results.append("\nStatistics:")
            for key, value in stats.items():
                results.append(f"- {key}: {value}")

        # Add output location if available
        if 'ResultConfiguration' in execution and 'OutputLocation' in execution['ResultConfiguration']:
            results.append(f"\nOutput location: {execution['ResultConfiguration']['OutputLocation']}")

        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error getting query status: {str(e)}")
        return f"Error getting query status: {str(e)}"


@mcp.tool()
def generate_create_table_sql(database_name: str, table_name: str, catalog_id: str = None) -> str:
    """Generate CREATE TABLE SQL statement for an existing Glue table

    Args:
        database_name: Name of the Glue database
        table_name: Name of the Glue table
        catalog_id: The AWS account ID of the Glue Catalog owner
    """
    if not glue_client:
        try:
            get_clients()
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return f"Error initializing AWS clients: {str(e)}"

    try:
        params = {
            'DatabaseName': database_name,
            'Name': table_name
        }
        if catalog_id and catalog_id.lower() != 'null':
            params['CatalogId'] = catalog_id

        response = glue_client.get_table(**params)

        table = response.get('Table', {})
        storage_descriptor = table.get('StorageDescriptor', {})
        columns = storage_descriptor.get('Columns', [])

        # Build the CREATE TABLE statement
        sql_parts = [f"CREATE EXTERNAL TABLE `{database_name}`.`{table_name}` ("]

        # Add columns
        column_defs = []
        for col in columns:
            col_name = col.get('Name', 'Unknown')
            col_type = col.get('Type', 'string')
            comment = col.get('Comment')

            if comment:
                column_defs.append(f"  `{col_name}` {col_type} COMMENT '{comment}'")
            else:
                column_defs.append(f"  `{col_name}` {col_type}")

        sql_parts.append(",\n".join(column_defs))
        sql_parts.append(")")

        # Add partition columns if available
        partition_keys = table.get('PartitionKeys', [])
        if partition_keys:
            partition_defs = []
            sql_parts.append("PARTITIONED BY (")

            for col in partition_keys:
                col_name = col.get('Name', 'Unknown')
                col_type = col.get('Type', 'string')
                comment = col.get('Comment')

                if comment:
                    partition_defs.append(f"  `{col_name}` {col_type} COMMENT '{comment}'")
                else:
                    partition_defs.append(f"  `{col_name}` {col_type}")

            sql_parts.append(",\n".join(partition_defs))
            sql_parts.append(")")

        # Add storage format
        input_format = storage_descriptor.get('InputFormat')
        output_format = storage_descriptor.get('OutputFormat')
        serde_info = storage_descriptor.get('SerdeInfo', {})

        # Detect common formats
        if input_format and 'parquet' in input_format.lower():
            sql_parts.append("STORED AS PARQUET")
        elif input_format and 'orc' in input_format.lower():
            sql_parts.append("STORED AS ORC")
        elif input_format:
            sql_parts.append(f"STORED AS INPUTFORMAT '{input_format}'")
            sql_parts.append(f"OUTPUTFORMAT '{output_format}'")

            if serde_info and 'SerializationLibrary' in serde_info:
                sql_parts.append(f"ROW FORMAT SERDE '{serde_info['SerializationLibrary']}'")

        # Add location
        location = storage_descriptor.get('Location')
        if location:
            sql_parts.append(f"LOCATION '{location}'")

        # Add table properties
        table_properties = table.get('Parameters', {})
        if table_properties:
            sql_parts.append("TBLPROPERTIES (")
            prop_parts = []

            for key, value in table_properties.items():
                prop_parts.append(f"  '{key}'='{value}'")

            sql_parts.append(",\n".join(prop_parts))
            sql_parts.append(")")

        # Add final semicolon
        sql_parts.append(";")

        return "\n".join(sql_parts)
    except Exception as e:
        logger.error(f"Error generating CREATE TABLE SQL: {str(e)}")
        return f"Error generating CREATE TABLE SQL: {str(e)}"


@mcp.prompt()
def example_prompt(query: str) -> str:
    return f"""You need to interact with AWS Glue, Athena, and Lake Formation to answer this question:

{query}

First, initialize the AWS session with initialize_aws_session(), then explore databases with list_databases().
Use list_tables() to browse tables in a database, and get_table_schema() to examine columns and their descriptions.
You can check permissions with check_lake_formation_permissions() and run queries with execute_athena_query().

Example workflow:
1. Initialize AWS session (use SSO profile if needed)
2. List available databases
3. Explore tables in a database
4. Check table schema and permissions
5. Execute Athena query and return results"""


if __name__ == "__main__":
    logger.info("Starting AWS Glue/Athena MCP server...")
    logger.info(f"Default region: {DEFAULT_REGION}")
    logger.info(f"Default S3 output: {DEFAULT_S3_OUTPUT}")
    logger.info(f"Default workgroup: {DEFAULT_WORKGROUP}")
    logger.info(f"Default timeout: {DEFAULT_TIMEOUT} seconds")
    logger.info(f"Using CTAS approach by default: {DEFAULT_CTAS_APPROACH}")
    logger.info(f"AWS Data Wrangler version: {wr.__version__}")
    logger.info(f"Boto3 version: {boto3.__version__}")

    # Check if AWS credentials are available
    try:
        default_session = boto3.Session()
        # Just check if we can get the caller identity
        sts_client = default_session.client('sts')
        sts_client.get_caller_identity()
        logger.info("AWS credentials available for default session")
    except Exception as e:
        logger.warning(f"No AWS credentials available for default session: {str(e)}")
        logger.info("You will need to initialize a session with a profile when using this server")

    # Initialize and run the server
    logger.info("Starting MCP server with stdio transport")
    mcp.run(transport="stdio")