#!/usr/bin/env python3
"""
MCP Server with dynamic pandas query generation for police shooting data.
"""
import json
import os
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger
from mcp.server.fastmcp import FastMCP

# Configuration - Update with your actual file paths
JSON_PATH = r"localdrive\dataset_metadata.json"
CSV_PATH = r"localdrive\dataset.csv"

# Create an MCP server
mcp = FastMCP("Dynamic Query Data Demo")

# Load data at server startup
try:
    with open(JSON_PATH, 'r') as f:
        metadata = json.load(f)
    logger.info(f"Loaded metadata from {JSON_PATH}")
except Exception as e:
    logger.error(f"Error loading metadata: {str(e)}")
    metadata = {}

try:
    df = pd.read_csv(CSV_PATH)
    # Convert date columns to datetime if they exist
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    logger.info(f"Loaded data from {CSV_PATH}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Data columns: {df.columns.tolist()}")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    df = pd.DataFrame()


@mcp.tool()
def get_metadata() -> str:
    """Get the metadata for the dataset"""
    return json.dumps(metadata, indent=2)


@mcp.tool()
def search_metadata(query: str) -> str:
    """Search the metadata for relevant information

    Args:
        query: The search term to look for in metadata
    """
    logger.info(f"Searching metadata for: {query}")

    result = {}
    query_lower = query.lower()

    for key, value in metadata.items():
        if isinstance(value, str) and query_lower in value.lower():
            result[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, str) and query_lower in sub_value.lower():
                    if key not in result:
                        result[key] = {}
                    result[key][sub_key] = sub_value

    if not result:
        return "No matching metadata found."

    return json.dumps(result, indent=2)


@mcp.tool()
def get_dataframe_info() -> str:
    """Get information about the DataFrame structure"""
    if df.empty:
        return "No data available."

    # Create a buffer to hold the string representation
    buffer = io.StringIO()

    # Write basic dataframe info
    buffer.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
    buffer.write("Columns:\n")

    # Write column info
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        buffer.write(f"- {col}: type={dtype}, nulls={null_count}, unique values={unique_count}\n")

    # Add sample values for each column (first 5 unique values)
    buffer.write("\nSample Values:\n")
    for col in df.columns:
        sample_values = df[col].dropna().unique()[:5]
        sample_str = ", ".join([str(val) for val in sample_values])
        buffer.write(f"- {col}: {sample_str}\n")

    return buffer.getvalue()


@mcp.tool()
def execute_pandas_query(query_code: str) -> str:
    """Execute a pandas query on the dataset

    Args:
        query_code: Python code using pandas to query the dataframe (uses 'df' as the dataframe name)
    """
    logger.info(f"Executing pandas query: {query_code}")

    if df.empty:
        return "No data available to query."

    try:
        # Create a local namespace with the dataframe
        local_namespace = {"df": df, "pd": pd}

        # Execute the query code in the local namespace
        exec(f"result = {query_code}", {"pd": pd}, local_namespace)

        # Get the result
        result = local_namespace.get("result")

        # Format the result based on its type
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return "Query returned an empty DataFrame."

            # If it's a large dataframe, limit the output
            if len(result) > 50:
                return f"Query returned {len(result)} rows. Here are the first 50:\n\n{result.head(50).to_string()}"
            else:
                return result.to_string()

        elif isinstance(result, pd.Series):
            return f"Series result:\n{result.to_string()}"

        else:
            return f"Result: {result}"

    except Exception as e:
        logger.error(f"Error executing pandas query: {str(e)}")
        return f"Error executing query: {str(e)}"


@mcp.tool()
def get_sample_data(rows: int = 5) -> str:
    """Get a sample of rows from the dataset

    Args:
        rows: Number of rows to sample (default: 5)
    """
    if df.empty:
        return "No data available."

    return df.sample(min(rows, len(df))).to_string()


@mcp.prompt()
def example_prompt(query: str) -> str:
    return f"""You need to analyze police shooting data to answer this question:

{query}

First, understand the data structure using get_metadata() or get_dataframe_info(). 
Then create a pandas query using execute_pandas_query() to answer the question."""


if __name__ == "__main__":
    print("Starting Dynamic Query MCP server...")
    print(f"JSON metadata: {JSON_PATH}")
    print(f"CSV data: {CSV_PATH}")
    # Initialize and run the server
    mcp.run(transport="stdio")