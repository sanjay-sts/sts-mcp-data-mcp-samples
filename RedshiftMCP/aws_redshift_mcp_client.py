#!/usr/bin/env python3
"""
AWS Redshift MCP Client

A Model Context Protocol (MCP) client that connects to the AWS Redshift MCP server
and integrates with Claude 3.7 to provide AI-powered data analysis.

This implementation follows the AWS MCP client patterns and standards.
"""
import asyncio
import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, cast

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolUnionParam, ToolUseBlock
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress
from rich.prompt import Prompt
from loguru import logger

# Load environment variables from .env file
load_dotenv(override=True)

# Configure basic logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=log_level)
logger.add("redshift_mcp_client.log", rotation="10 MB", retention="1 week", level="DEBUG")

logger.info(f"AWS Redshift MCP Client starting (log level: {log_level})")

# Configure the console for rich text output
console = Console()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AWS Redshift MCP Client")
    parser.add_argument('--server', default='./aws_redshift_mcp_server.py',
                        help='Path to the MCP server script')
    parser.add_argument('--profile', default=None,
                        help='AWS CLI profile name (e.g., SSO profile)')
    parser.add_argument('--region', default=None,
                        help='AWS region (e.g., us-east-1)')
    parser.add_argument('--model', default='claude-3-7-sonnet-latest',
                        help='Claude model to use')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level for console output')
    parser.add_argument('--log-file', default='redshift_mcp_client.log',
                        help='Path to log file')
    parser.add_argument('--auto-connect', action='store_true',
                        help='Automatically connect to Redshift using static resources')

    args = parser.parse_args()

    # Configure logging based on arguments
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add(args.log_file, rotation="10 MB", retention="1 week", level="DEBUG")

    logger.debug(f"Command line arguments: {args}")
    return args


# Parse command line arguments
args = parse_arguments()

# Check for Claude API key
try:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    use_claude = True
    logger.info("Claude API integration enabled")
    console.print(Panel("[bold green]Claude API integration enabled[/bold green]"))
except (KeyError, Exception) as e:
    use_claude = False
    logger.warning(f"Claude API integration disabled: {str(e)}")
    console.print(Panel(
        "[bold yellow]Claude API key not found or invalid.\n"
        "Falling back to manual mode. Add ANTHROPIC_API_KEY to your .env file to enable Claude.[/bold yellow]"
    ))


@dataclass
class Chat:
    """Handles the chat interface between the user and MCP server"""
    messages: List[MessageParam] = field(default_factory=list)
    args: argparse.Namespace = field(default_factory=lambda: args)

    system_prompt: str = """You are a helpful assistant for analyzing data in AWS Redshift.
Your job is to help the user explore and query data stored in Amazon Redshift.

Available tools:
- initialize_aws_session: Connect to AWS with optional profile and region
- connect_with_static_resources: Connect to Redshift using predefined static connection (recommended)
- connect_with_secret: Connect to Redshift using a secret from AWS Secrets Manager
- connect_with_credentials: Connect to Redshift using direct credentials
- list_clusters: List all Redshift clusters in the current AWS account
- list_schemas: List all schemas in the connected Redshift database
- list_tables: List all tables in a specified schema
- describe_table: Get detailed information about a specific table
- execute_query: Run SQL queries on the connected Redshift database
- search_tables: Search for tables containing a specific term
- search_columns: Search for columns containing a specific term
- get_table_sample: Get a sample of rows from a table
- get_connection_info: Get current connection status and details
- get_common_queries: Get list of common SQL query templates

This version uses tools instead of formal MCP resources for providing context information.
You can use get_connection_info() to check connection status and get_common_queries() to see example SQL templates.

Your workflow should be:
1. Always check if we're connected to Redshift first (preferably use connect_with_static_resources)
2. Help the user explore their database using list_schemas() and list_tables()
3. Examine table schemas with describe_table() before querying
4. Craft and execute SQL queries using execute_query()

When writing SQL:
- Use proper Redshift SQL syntax (similar to PostgreSQL but with Redshift-specific features)
- Include appropriate schema qualifiers (schema.table)
- Use double quotes for identifiers with special characters
- Consider using Redshift's distribution and sort keys for query optimization
- Refer to the common_query resources for examples of well-formed queries

Always explain the SQL you're writing in clear terms and interpret query results for the user.
For table exploration, first check the schema before suggesting queries.

Pay special attention to table and column comments, as they often contain valuable information about the data.
Use search_tables() and search_columns() to find relevant data when the user isn't sure where to look.
You can refer to table_info resources to see table structure without making tool calls.

Remember that Redshift is column-oriented and designed for analytics, so optimize queries accordingly.
"""

    async def process_query(self, session: ClientSession, query: str) -> None:
        """Process a user query using the MCP session and Claude (if available)"""
        logger.debug(f"Processing query: {query}")

        # List available tools
        logger.debug("Listing available tools from MCP server")
        tools_response = await session.list_tools()
        available_tools: list[ToolUnionParam] = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in tools_response.tools
        ]

        logger.debug(f"Found {len(available_tools)} available tools")
        for tool in available_tools:
            logger.debug(f"Tool available: {tool['name']}")

        # Get available resources (if supported)
        logger.debug("Checking for available resources from MCP server")
        resources = []

        try:
            resources_response = await session.list_resources()
            if resources_response and resources_response.resources:
                logger.debug(f"Found {len(resources_response.resources)} resource types")

                for resource_type in resources_response.resources:
                    resource_type_name = resource_type.name
                    logger.debug(f"Getting resources for type: {resource_type_name}")

                    try:
                        resource_instances = await session.get_all_resources(resource_type_name)
                        if resource_instances and resource_instances.resources:
                            logger.debug(f"Found {len(resource_instances.resources)} instances of {resource_type_name}")
                            resources.append({
                                "type": resource_type_name,
                                "resources": resource_instances.resources
                            })
                    except Exception as e:
                        logger.error(f"Error getting resources for type {resource_type_name}: {str(e)}")
        except Exception as e:
            logger.debug(f"Resources not supported or available: {str(e)}")
            # Resources not supported in this MCP version, continue without them

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
                    await self._handle_manual_mode(session, query, available_tools, resources)
            else:
                # Use manual mode without Claude
                logger.info("Using manual mode to process query (no Claude API)")
                await self._handle_manual_mode(session, query, available_tools, resources)

    async def _handle_manual_mode(self, session: ClientSession, query: str,
                                  available_tools: list[ToolUnionParam],
                                  resources: list[Dict[str, Any]]) -> None:
        """Handle queries in manual mode (without Claude)"""
        logger.info("Running in manual mode (no Claude API)")
        console.print("\n[bold yellow]Running in manual mode (no Claude API)[/bold yellow]")

        # Show available resources
        if resources:
            console.print("\n[bold]Available Resources:[/bold]")
            for resource_type in resources:
                resource_count = len(resource_type["resources"])
                console.print(f"[cyan]{resource_type['type']}[/cyan]: {resource_count} resources available")

                # Ask if user wants to see resource details
                if console.input(f"View {resource_type['type']} resources? (y/n): ").lower() == 'y':
                    resource_table = Table(title=f"{resource_type['type']} Resources")

                    # Get all potential keys across resources
                    all_keys = set()
                    for resource in resource_type["resources"]:
                        all_keys.update(resource.keys())

                    # Add columns for the table
                    for key in all_keys:
                        resource_table.add_column(key, overflow="fold")

                    # Add rows to the table
                    for resource in resource_type["resources"]:
                        row = [str(resource.get(key, "")) for key in all_keys]
                        resource_table.add_row(*row)

                    console.print(resource_table)

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

            if selected_tool == "execute_query":
                # Format SQL results as table if possible
                console.print("\n[bold green]Query Results:[/bold green]")
                # Just print the formatted result as-is
                console.print(tool_result_text)
                logger.debug("Displayed SQL results")
            elif selected_tool == "list_schemas" or selected_tool == "list_tables" or selected_tool == "list_clusters":
                # General result display
                console.print("\n[bold green]Result:[/bold green]")
                console.print(tool_result_text)
                logger.debug("Displayed general tool result")
            else:
                # General result display
                console.print("\n[bold green]Result:[/bold green]")
                console.print(tool_result_text)
                logger.debug("Displayed general tool result")

            # Ask if user wants to execute another tool
            if console.input("\nExecute another tool? (y/n): ").lower() == 'y':
                logger.debug("User chose to execute another tool")
                await self._handle_manual_mode(session, query, available_tools, resources)
            else:
                logger.debug("User completed manual mode session")

        except (ValueError, IndexError) as e:
            error_msg = f"Error in tool selection: {str(e)}"
            logger.error(error_msg)
            console.print(f"[bold red]{error_msg}[/bold red]")
            await self._handle_manual_mode(session, query, available_tools, resources)

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
                if tool_name == "execute_query" and "query" in tool_args:
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
                if tool_name == "execute_query":
                    # Display query results as-is (they're already formatted by the server)
                    console.print("[bold green]Query Results:[/bold green]")
                    console.print(tool_result_text)
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
[bold]AWS Redshift MCP Client[/bold]
Profile: {self.args.profile or 'default'}
Region: {self.args.region or 'default'}
Mode: {'Claude AI Assistant' if use_claude else 'Manual Mode'}
Claude Model: {self.args.model if use_claude else 'N/A'}
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

                # Try connecting with static resources if auto-connect enabled
                if self.args.auto_connect:
                    console.print("\n[bold]Connecting to Redshift using static resources...[/bold]")
                    try:
                        result = await session.call_tool("connect_with_static_resources", {})
                        result_text = getattr(result.content[0], "text", "")
                        console.print(f"[green]{result_text}[/green]")
                        logger.info("Connected to Redshift using static resources")
                    except Exception as e:
                        logger.warning(f"Unable to connect with static resources: {str(e)}")
                        console.print(
                            "[yellow]Could not connect using static resources. You'll need to connect manually.[/yellow]")
            except Exception as e:
                error_msg = f"Error initializing AWS session: {str(e)}"
                logger.error(error_msg)
                console.print(f"[bold red]{error_msg}[/bold red]")
                console.print("You may need to manually initialize the session during the chat.")
        else:
            # Even without profile/region specified, try to connect with static resources if auto-connect is enabled
            if self.args.auto_connect:
                console.print("\n[bold]Connecting to Redshift using static resources...[/bold]")
                try:
                    result = await session.call_tool("connect_with_static_resources", {})
                    result_text = getattr(result.content[0], "text", "")
                    console.print(f"[green]{result_text}[/green]")
                    logger.info("Connected to Redshift using static resources")
                except Exception as e:
                    logger.warning(f"Unable to connect with static resources: {str(e)}")
                    console.print(
                        "[yellow]Could not connect using static resources. You'll need to connect manually.[/yellow]")

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

## Connection and Setup
- "Initialize my AWS session with profile 'data-analyst'"
- "Connect to my Redshift cluster using static resources"
- "Connect to my Redshift cluster with secret 'redshift/analytics'"
- "Connect to Redshift with host analytics-cluster.example.com, database 'analytics', username 'analyst'"
- "List available Redshift clusters in my account"

## Database Exploration
- "List all schemas in the database"
- "Show me the tables in the 'public' schema"
- "Describe the 'customers' table in the 'sales' schema"
- "Search for tables related to 'order'"
- "Find columns containing 'date' across all tables"
- "Get a sample of data from the customers table"

## Queries
- "How many customers do we have by region?"
- "What are the top 10 products by revenue?"
- "Show me the daily sales trend over the last month"
- "Calculate the average order size by customer segment"
- "Find duplicate records in the 'transactions' table"

## Analysis Help
- "What tables would I need to analyze customer purchasing patterns?"
- "Help me optimize a slow query for customer order history"
- "What's the best way to identify our most valuable customers?"
- "Create a query to find transactions with suspicious activity"

## Options
- Client command-line options:
  --server: Path to the MCP server script (default: ./aws_redshift_mcp_server.py)
  --profile: AWS CLI profile to use
  --region: AWS region to use
  --model: Claude model to use (default: claude-3-7-sonnet-latest)
  --log-level: Logging level (DEBUG, INFO, WARNING, ERROR)
  --log-file: Log file path (default: redshift_mcp_client.log)
  --auto-connect: Automatically connect to Redshift using static resources
"""
        console.print(Markdown(help_text))

    async def run(self):
        """Run the chat client"""
        logger.info(f"Starting client with server: {self.args.server}")
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",  # Executable
            args=[self.args.server],  # Server script
            env={"FASTMCP_LOG_LEVEL": self.args.log_level},  # Environment variables
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
    logger.info("Starting AWS Redshift MCP Client")
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