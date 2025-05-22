#!/usr/bin/env python3
"""
MCP Client for AWS Glue/Athena/Lake Formation using stdio transport.
Includes debug-level logging to match server capabilities.
"""
import os
import sys
import asyncio
import argparse
from dataclasses import dataclass, field
from typing import Union, cast, Dict, List, Optional

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolUnionParam, ToolUseBlock
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress
from loguru import logger

# Configure basic logging initially - with no handlers yet
logger.remove()  # Make sure we start with no handlers
logger.add(sys.stderr, level="INFO")  # Default console logging
logger.add("mcp_client.log", rotation="10 MB", retention="1 week", level="DEBUG")

logger.debug("Initial logger configuration complete")

# Load environment variables from .env file
load_dotenv()
logger.debug("Loaded environment variables from .env file")

# Configure the console for rich text output
console = Console()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AWS Glue/Athena/Lake Formation MCP Client")
    parser.add_argument('--server', default='./aws_glue_athena_mcp_server.py',
                        help='Path to the MCP server script')
    parser.add_argument('--profile', default=None,
                        help='AWS CLI profile name (e.g., SSO profile)')
    parser.add_argument('--region', default=None,
                        help='AWS region (e.g., us-east-1)')
    parser.add_argument('--model', default='claude-3-7-sonnet-latest',
                        help='Claude model to use (if API key is available)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level for console output')
    parser.add_argument('--log-file', default='mcp_client.log',
                        help='Path to log file')

    args = parser.parse_args()

    # Configure logging based on arguments - properly remove all handlers first
    logger.remove()  # Remove all existing handlers

    # Add new handlers with configured log levels
    logger.add(sys.stderr, level=args.log_level)
    logger.add(args.log_file, rotation="10 MB", retention="1 week", level="DEBUG")

    logger.debug(f"Command line arguments: {args}")
    return args


# Parse arguments first
args = parse_arguments()

# Now check for Claude API key
try:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    use_claude = True
    logger.info("Claude API integration enabled")
    console.print(Panel("[bold green]Claude API integration enabled[/bold green]"))
    anthropic_client = anthropic.AsyncAnthropic()
    logger.debug(f"Using Claude model: {args.model}")
except KeyError:
    use_claude = False
    logger.warning("Claude API key not found in environment variables - falling back to manual mode")
    console.print(Panel(
        "[bold yellow]Claude API key not found in environment variables.\n"
        "Falling back to manual mode. Add ANTHROPIC_API_KEY to your .env file to enable Claude.[/bold yellow]"
    ))


# Create a factory function to return the parsed args
def get_parsed_args():
    """Factory function to return the parsed arguments"""
    global args
    return args


# Simple factory to create message list
def get_empty_messages():
    """Factory function to return an empty message list"""
    return []


@dataclass
class Chat:
    """Handles the chat interface between the user and MCP server"""
    messages: list[MessageParam] = field(default_factory=get_empty_messages)
    args: argparse.Namespace = field(default_factory=get_parsed_args)

    system_prompt: str = """You are a helpful assistant for analyzing data using AWS Glue Catalog, Athena, and Lake Formation.
Your job is to help the user explore and query data stored in Amazon S3 and managed via AWS Glue and Lake Formation.

Available tools:
- initialize_aws_session: Connect to AWS with optional profile and region
- list_databases: List all databases in the Glue Catalog
- list_tables: List all tables in a specific Glue database
- get_table_schema: Get detailed schema information including column descriptions
- check_lake_formation_permissions: Check permissions for a database or table
- execute_athena_query: Run SQL queries using Amazon Athena
- get_query_execution_status: Check the status of a running query
- generate_create_table_sql: Generate CREATE TABLE SQL for a Glue table

Your workflow should be:
1. Always initialize the AWS session first using initialize_aws_session()
2. Help the user explore their data catalog using list_databases() and list_tables()
3. Examine table schemas with get_table_schema() before querying
4. Check permissions with check_lake_formation_permissions() if needed
5. Craft and execute SQL queries using execute_athena_query()

When writing SQL:
- Use proper Athena/Presto SQL syntax (similar to ANSI SQL)
- Include appropriate table qualifiers (database.table)
- Use double quotes for identifiers with special characters
- Handle partitions efficiently

Always explain the SQL you're writing in clear terms and interpret query results for the user.
For table exploration, first check the schema before suggesting queries.

Note that the server uses a hybrid implementation with boto3 and AWS Data Wrangler to provide optimal performance.
AWS Data Wrangler is used for data operations (tables, schemas, querying) while boto3 handles initialization, security, and management operations.
"""

    async def process_query(self, session: ClientSession, query: str) -> None:
        """Process a user query using the MCP session and Claude (if available)"""
        logger.debug(f"Processing query: {query}")

        # List available tools
        logger.debug("Listing available tools from MCP server")
        response = await session.list_tools()
        available_tools: list[ToolUnionParam] = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        logger.debug(f"Found {len(available_tools)} available tools")
        for tool in available_tools:
            logger.debug(f"Tool available: {tool['name']}")

        # Show that we're processing
        with console.status("[bold green]Processing query...[/bold green]"):
            if use_claude:
                # Use Claude API
                logger.info(f"Using Claude API ({self.args.model}) to process query")
                try:
                    res = await anthropic_client.messages.create(
                        model=self.args.model,
                        system=self.system_prompt,
                        max_tokens=8000,
                        messages=self.messages,
                        tools=available_tools,
                    )
                    logger.debug("Received initial response from Claude")

                    # Process Claude's responses recursively until no more tools are called
                    await self._process_response(session, res, available_tools)
                except Exception as e:
                    logger.error(f"Error calling Claude API: {str(e)}")
                    console.print(f"[bold red]Error calling Claude API: {str(e)}[/bold red]")
                    console.print("Falling back to manual mode for this query.")
                    await self._handle_manual_mode(session, query, available_tools)
            else:
                # Use manual mode without Claude
                logger.info("Using manual mode to process query (no Claude API)")
                await self._handle_manual_mode(session, query, available_tools)

    async def _handle_manual_mode(self, session: ClientSession, query: str,
                                  available_tools: list[ToolUnionParam]) -> None:
        """Handle queries in manual mode (without Claude)"""
        logger.info("Running in manual mode (no Claude API)")
        console.print("\n[bold yellow]Running in manual mode (no Claude API)[/bold yellow]")

        # Display available tools
        console.print("\n[bold]Available Tools:[/bold]")
        tools_table = Table(title="MCP Tools")
        tools_table.add_column("Tool Name", style="cyan")
        tools_table.add_column("Description", style="green")

        for tool in available_tools:
            tools_table.add_row(tool["name"], tool["description"])

        console.print(tools_table)

        # Ask user which tool to use
        console.print("\n[bold]Choose a tool to execute:[/bold]")
        tool_names = [tool["name"] for tool in available_tools]

        for i, name in enumerate(tool_names):
            console.print(f"{i + 1}. {name}")

        selection = console.input("\nEnter tool number (or 'q' to quit): ")

        if selection.lower() == 'q':
            logger.debug("User quit manual mode")
            return

        try:
            tool_index = int(selection) - 1
            selected_tool = tool_names[tool_index]
            logger.info(f"User selected tool: {selected_tool}")

            # Get tool parameters
            tool_schema = [tool["input_schema"] for tool in available_tools if tool["name"] == selected_tool][0]
            required_params = tool_schema.get("required", [])
            properties = tool_schema.get("properties", {})

            # Collect parameter values
            params = {}
            for param_name, param_details in properties.items():
                is_required = param_name in required_params
                param_type = param_details.get("type", "string")
                description = param_details.get("description", "")

                prompt = f"\n{param_name}"
                if is_required:
                    prompt += " (required)"
                if description:
                    prompt += f": {description}"
                prompt += ": "

                value = console.input(prompt)
                logger.debug(f"Parameter {param_name}: {value}")

                # Skip optional empty parameters
                if value or is_required:
                    # Convert types if necessary
                    if param_type == "integer":
                        params[param_name] = int(value)
                    elif param_type == "boolean":
                        params[param_name] = value.lower() in ("yes", "true", "t", "1")
                    elif param_type == "object" and value:
                        # Assume JSON string for objects
                        import json
                        params[param_name] = json.loads(value)
                    else:
                        params[param_name] = value

            # Execute the tool
            console.print(f"\n[bold]Executing {selected_tool}...[/bold]")
            logger.info(f"Executing tool: {selected_tool} with params: {params}")

            with Progress() as progress:
                task = progress.add_task("[cyan]Executing tool...", total=1)

                try:
                    # Call the tool
                    result = await session.call_tool(selected_tool, params)
                    progress.update(task, completed=1)
                    logger.debug("Tool execution completed successfully")
                except Exception as e:
                    progress.update(task, completed=1)
                    error_msg = f"Error executing tool: {str(e)}"
                    logger.error(error_msg)
                    console.print(f"[bold red]{error_msg}[/bold red]")
                    return

            # Display the result
            tool_result_text = getattr(result.content[0], "text", "")
            logger.debug(f"Tool returned {len(tool_result_text)} characters of output")

            if selected_tool == "execute_athena_query":
                # Format SQL results as table if possible
                console.print("\n[bold green]Query Results:[/bold green]")
                try:
                    # Try to parse as a table
                    lines = tool_result_text.strip().split("\n")
                    if len(lines) >= 3 and "-+-" in lines[1]:
                        headers = [h.strip() for h in lines[0].split("|")]
                        result_table = Table()

                        for header in headers:
                            result_table.add_column(header)

                        for row_line in lines[2:]:
                            row_values = [cell.strip() for cell in row_line.split("|")]
                            result_table.add_row(*row_values)

                        console.print(result_table)
                        logger.debug(f"Formatted SQL results as table with {len(lines) - 2} rows")
                    else:
                        # Fall back to plain text
                        console.print(tool_result_text)
                        logger.debug("Displayed SQL results as plain text (not in table format)")
                except Exception as e:
                    # If parsing fails, just print the text
                    console.print(tool_result_text)
                    logger.warning(f"Failed to parse SQL results as table: {str(e)}")
            elif selected_tool == "generate_create_table_sql":
                # Format SQL nicely
                console.print("\n[bold green]Generated SQL:[/bold green]")
                console.print(Syntax(tool_result_text, "sql", theme="monokai"))
                logger.debug("Displayed generated SQL with syntax highlighting")
            else:
                # General result display
                console.print("\n[bold green]Result:[/bold green]")
                console.print(tool_result_text)
                logger.debug("Displayed general tool result")

            # Ask if user wants to execute another tool
            if console.input("\nExecute another tool? (y/n): ").lower() == 'y':
                logger.debug("User chose to execute another tool")
                await self._handle_manual_mode(session, query, available_tools)
            else:
                logger.debug("User completed manual mode session")

        except (ValueError, IndexError) as e:
            error_msg = f"Error in tool selection: {str(e)}"
            logger.error(error_msg)
            console.print(f"[bold red]{error_msg}[/bold red]")
            await self._handle_manual_mode(session, query, available_tools)

    async def _process_response(self, session: ClientSession, res, available_tools):
        """Process a response from Claude, handling any tool calls recursively"""
        logger.debug("Processing Claude response")
        assistant_message_content = []
        contains_tool_use = False

        for content in res.content:
            if content.type == "text":
                logger.debug("Processing text content from Claude")
                assistant_message_content.append(content)
                console.print(Markdown(content.text))
            elif content.type == "tool_use":
                logger.debug(f"Processing tool use: {content.name}")
                contains_tool_use = True
                tool_name = content.name
                tool_args = content.input

                console.print(f"\nExecuting tool: [bold]{tool_name}[/bold]")

                # Special handling for different tools
                if tool_name == "execute_athena_query" and "query" in tool_args:
                    logger.debug(f"Executing SQL query: {tool_args['query']}")
                    console.print("SQL Query:")
                    console.print(Syntax(tool_args["query"], "sql", theme="monokai"))
                elif tool_args:
                    formatted_args = "\n".join([f"  - {k}: {v}" for k, v in tool_args.items()])
                    console.print(f"Arguments:\n{formatted_args}")
                    logger.debug(f"Tool arguments: {tool_args}")

                # Execute tool call with progress indicator and error handling
                with Progress() as progress:
                    task = progress.add_task("[cyan]Executing tool...", total=1)
                    try:
                        start_time = asyncio.get_event_loop().time()
                        result = await session.call_tool(tool_name, cast(dict, tool_args))
                        elapsed_time = asyncio.get_event_loop().time() - start_time
                        logger.debug(f"Tool execution completed in {elapsed_time:.2f} seconds")
                        progress.update(task, completed=1)
                    except Exception as e:
                        progress.update(task, completed=1)
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        logger.error(error_msg)
                        console.print(f"[bold red]{error_msg}[/bold red]")

                        # Create an error result to pass back to Claude
                        result_content = [{"text": f"Error: {str(e)}", "type": "text"}]
                        result = type('obj', (object,), {'content': result_content})

                tool_result_text = getattr(result.content[0], "text", "")
                logger.debug(f"Tool returned {len(tool_result_text)} characters of output")

                # Add the tool use to the assistant's message
                assistant_message_content.append(content)

                # Save the current assistant message
                self.messages.append(
                    {"role": "assistant", "content": assistant_message_content}
                )

                # Add tool result to messages
                tool_result = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": tool_result_text,
                        }
                    ],
                }
                self.messages.append(tool_result)

                # Display tool results appropriately based on tool type
                if tool_name == "execute_athena_query":
                    # Try to format SQL results as a table
                    console.print("[bold green]Query Results:[/bold green]")
                    try:
                        # Check if result looks like a formatted table
                        lines = tool_result_text.strip().split("\n")
                        if len(lines) >= 3 and "-+-" in lines[1]:
                            # Looks like a table, print as-is but with some styling
                            console.print(Syntax(tool_result_text, "text", theme="default"))
                            logger.debug(f"Formatted SQL results as table with {len(lines)} lines")
                        else:
                            # Not a table format, just print as text
                            console.print(tool_result_text)
                            logger.debug("Displayed SQL results as plain text")
                    except Exception as e:
                        # If parsing fails, just print the text
                        console.print(tool_result_text)
                        logger.warning(f"Failed to parse SQL results: {str(e)}")
                else:
                    # For other tools, show preview of results
                    if len(tool_result_text) > 500:
                        preview = tool_result_text[:500] + "..."
                        console.print(f"Tool result preview: [dim]{preview}[/dim]")
                        console.print("(Full result sent to Claude)")
                        logger.debug(f"Showed preview of long result ({len(tool_result_text)} chars)")
                    else:
                        console.print(f"Tool result: [dim]{tool_result_text}[/dim]")
                        logger.debug("Showed full tool result")

                # Get next response from Claude with the tool result
                logger.debug("Getting follow-up response from Claude")
                try:
                    follow_up_res = await anthropic_client.messages.create(
                        model=self.args.model,
                        max_tokens=8000,
                        messages=self.messages,
                        tools=available_tools,
                    )

                    # Process the follow-up response recursively
                    await self._process_response(session, follow_up_res, available_tools)
                except Exception as e:
                    error_msg = f"Error getting follow-up response from Claude: {str(e)}"
                    logger.error(error_msg)
                    console.print(f"\n[bold red]{error_msg}[/bold red]")
                    console.print("The tool was executed, but Claude couldn't process the result.")

                return

        # If we got here, there were no tool uses in this response, so save it
        if not contains_tool_use and assistant_message_content:
            logger.debug("No tool uses in response, saving final assistant message")
            self.messages.append(
                {"role": "assistant", "content": assistant_message_content}
            )

    async def chat_loop(self, session: ClientSession):
        """Main chat loop"""
        logger.info("Starting chat loop")
        banner_text = f"""
[bold]AWS Glue/Athena/Lake Formation MCP Client[/bold]
Profile: {self.args.profile or 'default'}
Region: {self.args.region or 'default'}
Mode: {'Claude AI Assistant' if use_claude else 'Manual Mode'}
Log Level: {self.args.log_level}
Log File: {self.args.log_file}
"""
        console.print(Panel.fit(banner_text, border_style="green"))

        # Initialize AWS session with profile/region from args if specified
        if self.args.profile or self.args.region:
            logger.info(f"Initializing AWS session with profile={self.args.profile}, region={self.args.region}")
            console.print("\n[bold]Initializing AWS session...[/bold]")
            init_params = {}
            if self.args.profile:
                init_params["profile"] = self.args.profile
            if self.args.region:
                init_params["region"] = self.args.region

            try:
                result = await session.call_tool("initialize_aws_session", init_params)
                result_text = getattr(result.content[0], "text", "")
                console.print(f"[green]{result_text}[/green]")
                logger.info("AWS session initialized successfully")
            except Exception as e:
                error_msg = f"Error initializing AWS session: {str(e)}"
                logger.error(error_msg)
                console.print(f"[bold red]{error_msg}[/bold red]")
                console.print("You may need to manually initialize the session during the chat.")

        console.print("\nType [bold]'exit'[/bold] to quit, [bold]'help'[/bold] for example commands.")

        while True:
            query = console.input("\n[bold blue]You:[/bold blue] ").strip()

            if query.lower() == 'exit':
                logger.info("User requested to exit")
                console.print("[bold yellow]Exiting...[/bold yellow]")
                break

            if query.lower() == 'help':
                logger.debug("User requested help")
                self._print_help()
                continue

            if not query:
                continue

            logger.info(f"User query: {query}")

            # Add query to message history
            self.messages.append(
                MessageParam(
                    role="user",
                    content=query,
                )
            )

            # Process the query
            try:
                await self.process_query(session, query)
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                logger.error(error_msg)
                console.print(f"\n[bold red]{error_msg}[/bold red]")

    def _print_help(self):
        """Print help information"""
        logger.debug("Displaying help information")
        help_text = """
# Example Commands

## Basic Exploration
- "Initialize my AWS session with profile 'data-analyst'"
- "List all databases in the Glue Catalog"
- "Show me the tables in the 'analytics' database"
- "What's the schema of the 'users' table in the 'customers' database?"
- "Check my Lake Formation permissions for the 'sales' database"

## Queries
- "Run a query to count rows in analytics.daily_sales"
- "Query the top 10 customers by order value"
- "Find all transactions from the last 30 days"
- "Generate a CREATE TABLE statement for analytics.user_sessions"

## Analysis Examples
- "Analyze the distribution of values in the 'category' column"
- "Compare sales figures between regions"
- "Find correlated columns in the customer_behavior table"
- "Show me a trend of daily sales over time"

## Logging Options
- You can adjust logging with --log-level [DEBUG|INFO|WARNING|ERROR]
- Logs are written to {self.args.log_file}
"""
        console.print(Markdown(help_text))

    async def run(self):
        """Run the chat client"""
        logger.info(f"Starting client with server: {self.args.server}")
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",  # Executable
            args=[self.args.server],  # Server script
            env=None,  # Optional environment variables
        )

        logger.debug("Connecting to MCP server")
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the connection
                    logger.debug("Initializing MCP session")
                    await session.initialize()
                    logger.info("MCP session initialized successfully")

                    # Start the chat loop
                    await self.chat_loop(session)
        except Exception as e:
            error_msg = f"Error connecting to MCP server: {str(e)}"
            logger.error(error_msg)
            console.print(f"[bold red]{error_msg}[/bold red]")
            console.print("Please check that the server script exists and is executable.")


if __name__ == "__main__":
    logger.info("Starting AWS Glue/Athena MCP Client")
    try:
        chat = Chat()
        asyncio.run(chat.run())
    except KeyboardInterrupt:
        logger.info("Client terminated by keyboard interrupt")
        console.print("\n[bold yellow]Client terminated by keyboard interrupt[/bold yellow]")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
    finally:
        logger.info("Client shutdown complete")
        # Ensure all log messages are flushed
        for handler in logger._core.handlers:
            try:
                handler.flush()
            except:
                pass