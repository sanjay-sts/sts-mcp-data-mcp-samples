import psycopg2
from psycopg2 import sql
from loguru import logger
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("PostgreSQL Demo")


@mcp.tool()
def query_postgres(sql_query: str, connection_params: dict = None) -> str:
    """Execute SQL queries on a PostgreSQL database

    Args:
        sql_query: The SQL query to execute
        connection_params: Optional dictionary with connection parameters.
                          If not provided, defaults will be used.
    """
    logger.info(f"Executing PostgreSQL query: {sql_query}")

    # Default connection parameters
    default_params = {
        "host": "localhost",
        "database": "dvdrental",
        "user": "user",
        "password": "password",
        "port": "5432"
    }

    # Use provided connection params or defaults
    params = connection_params if connection_params else default_params

    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=params["host"],
            database=params["database"],
            user=params["user"],
            password=params["password"],
            port=params["port"]
        )

        # Create a cursor
        cursor = conn.cursor()

        # Execute the query
        cursor.execute(sql_query)

        # Check if the query is a SELECT query or similar that returns data
        if cursor.description:
            result = cursor.fetchall()
            # Format results as a string
            column_names = [desc[0] for desc in cursor.description]
            formatted_results = [", ".join(column_names)]

            for row in result:
                formatted_results.append(", ".join(str(value) for value in row))

            return "\n".join(formatted_results)
        else:
            # For non-SELECT queries (INSERT, UPDATE, DELETE)
            conn.commit()
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if 'conn' in locals():
            if 'cursor' in locals():
                cursor.close()
            conn.close()


@mcp.tool()
def list_tables() -> str:
    """List all tables in the PostgreSQL database"""
    list_tables_query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name;
    """
    return query_postgres(list_tables_query)


@mcp.tool()
def describe_table(table_name: str) -> str:
    """Get the structure of a specific table

    Args:
        table_name: The name of the table to describe
    """
    describe_query = f"""
    SELECT column_name, data_type, character_maximum_length
    FROM information_schema.columns
    WHERE table_name = '{table_name}'
    ORDER BY ordinal_position;
    """
    return query_postgres(describe_query)


# @mcp.prompt()
# def example_prompt(code: str) -> str:
#     return f"Please review this code:\n\n{code}"


if __name__ == "__main__":
    print("Starting PostgreSQL MCP server...")
    # Initialize and run the server
    mcp.run(transport="stdio")