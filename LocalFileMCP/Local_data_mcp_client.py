#!/usr/bin/env python3
"""
MCP Client for dynamic pandas query generation using stdio transport.
"""
import asyncio
import os
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

load_dotenv()  # Load .env file if it exists

# Configure the console
console = Console()

# Check if ANTHROPIC_API_KEY is set
try:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    use_claude = True
    console.print(Panel("[bold green]Claude API integration enabled[/bold green]"))
    anthropic_client = anthropic.AsyncAnthropic()
except KeyError:
    use_claude = False
    console.print(Panel(
        "[bold yellow]Claude API key not found in environment variables.\n"
        "Falling back to simulated responses.[/bold yellow]"
    ))

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["./Local_data_mcp_server.py"],  # Server script
    env=None,  # Optional environment variables
)


@dataclass
class Chat:
    messages: list[MessageParam] = field(default_factory=list)

    system_prompt: str = """You are a data analysis assistant for a police shooting dataset.
    Your job is to analyze the dataset by writing pandas code to answer the user's questions.

    Available tools:
    - get_metadata: Get the full metadata for the dataset
    - search_metadata: Search for specific information in the metadata
    - get_dataframe_info: Get information about the dataframe structure including column types and sample values
    - execute_pandas_query: Execute a pandas query (you will supply Python code that uses 'df' as the dataframe name)
    - get_sample_data: Get a sample of rows from the dataset

    IMPORTANT: Your workflow should typically be:
    1. First, understand the dataset structure using get_metadata() and get_dataframe_info()
    2. Then, write a pandas query to answer the user's question
    3. Finally, execute the query using execute_pandas_query() and explain the results

    When writing pandas queries:
    - Always use 'df' as the name of the dataframe
    - Keep queries concise but readable
    - Use proper pandas methods like .query(), .groupby(), .value_counts(), etc.
    - Return aggregated/summarized data when appropriate
    - For more complex queries, build them up step by step
    - Format results appropriately (e.g., sort data for readability)

    Example queries:
    - df['race'].value_counts()
    - df.groupby('state')['id'].count().sort_values(ascending=False).head(10)
    - df[df['armed'] == 'gun'].groupby('race')['id'].count()
    - df.query("age > 25 and signs_of_mental_illness == True").shape[0]

    Always explain the pandas code you're using and interpret the results in plain language.
    If a query fails, debug the issue and try a different approach.
    """

    async def simulate_llm_response(self, query: str, tool_results: List[Dict]) -> Dict:
        """Simulate an LLM response when Claude API is not available"""
        # Create a basic response based on the tool results
        response_parts = ["I'll help you analyze the police shooting data.\n\n"]
        pandas_code = None
        pandas_result = None

        # Process the tool results to extract information
        for result in tool_results:
            tool_name = result.get("tool_name", "")
            tool_result = result.get("result", "")

            # Skip empty results
            if not tool_result or tool_result.strip() == "":
                continue

            if tool_name == "get_metadata" or tool_name == "search_metadata":
                response_parts.append("I've looked at the metadata for this dataset.\n\n")

            elif tool_name == "get_dataframe_info":
                response_parts.append("I've examined the structure of the dataset. ")
                if "DataFrame Shape:" in tool_result:
                    shape_line = tool_result.split("\n")[0]
                    response_parts.append(f"{shape_line}\n\n")

            elif tool_name == "execute_pandas_query":
                pandas_result = tool_result

            elif tool_name == "get_sample_data":
                response_parts.append("Here's a sample of the data to understand its structure:\n\n")
                response_parts.append("```\n" + tool_result + "\n```\n\n")

        # Generate a pandas query based on the user's question
        if "race" in query.lower() or "ethnicity" in query.lower():
            pandas_code = "df['race'].value_counts()"
            if not pandas_result:
                response_parts.append("Let me analyze the racial distribution in the dataset.\n\n")
                response_parts.append("I'll use this pandas code to count occurrences by race:\n\n")
                response_parts.append("```python\n" + pandas_code + "\n```\n\n")

        elif "state" in query.lower():
            pandas_code = "df['state'].value_counts().head(10)"
            if not pandas_result:
                response_parts.append("Let me look at the geographical distribution of incidents.\n\n")
                response_parts.append("I'll use this pandas code to count incidents by state:\n\n")
                response_parts.append("```python\n" + pandas_code + "\n```\n\n")

        elif "age" in query.lower():
            pandas_code = "df['age'].describe()"
            if not pandas_result:
                response_parts.append("Let me analyze the age distribution of victims.\n\n")
                response_parts.append("I'll use this pandas code to get age statistics:\n\n")
                response_parts.append("```python\n" + pandas_code + "\n```\n\n")

        elif "gender" in query.lower() or "sex" in query.lower():
            pandas_code = "df['gender'].value_counts()"
            if not pandas_result:
                response_parts.append("Let me analyze the gender distribution in the dataset.\n\n")
                response_parts.append("I'll use this pandas code to count by gender:\n\n")
                response_parts.append("```python\n" + pandas_code + "\n```\n\n")

        elif "mental" in query.lower() or "illness" in query.lower():
            pandas_code = "df['signs_of_mental_illness'].value_counts()"
            if not pandas_result:
                response_parts.append("Let me analyze data on mental illness indicators.\n\n")
                response_parts.append("I'll use this pandas code to get the distribution:\n\n")
                response_parts.append("```python\n" + pandas_code + "\n```\n\n")

        elif "time" in query.lower() or "year" in query.lower():
            pandas_code = "df['date'].dt.year.value_counts().sort_index()"
            if not pandas_result:
                response_parts.append("Let me analyze the time distribution of incidents.\n\n")
                response_parts.append("I'll use this pandas code to count incidents by year:\n\n")
                response_parts.append("```python\n" + pandas_code + "\n```\n\n")

        elif "weapon" in query.lower() or "armed" in query.lower():
            pandas_code = "df['armed'].value_counts().head(10)"
            if not pandas_result:
                response_parts.append("Let me analyze what victims were armed with.\n\n")
                response_parts.append("I'll use this pandas code to count by armed status:\n\n")
                response_parts.append("```python\n" + pandas_code + "\n```\n\n")

        else:
            # Default general analysis
            pandas_code = "df.describe(include='all').T"
            if not pandas_result:
                response_parts.append("Let me provide a general analysis of the dataset.\n\n")
                response_parts.append("I'll use this pandas code to get a statistical summary:\n\n")
                response_parts.append("```python\n" + pandas_code + "\n```\n\n")

        # Add pandas results if available
        if pandas_result:
            response_parts.append("Here are the results:\n\n")
            response_parts.append("```\n" + pandas_result + "\n```\n\n")

            # Add some interpretation
            if "race" in query.lower():
                response_parts.append("This shows the distribution of victims by race in the dataset.\n\n")
            elif "state" in query.lower():
                response_parts.append("This shows which states had the most incidents.\n\n")
            elif "age" in query.lower():
                response_parts.append("This shows the statistical distribution of victim ages.\n\n")
            elif "gender" in query.lower():
                response_parts.append("This shows the gender distribution of victims.\n\n")
            elif "armed" in query.lower():
                response_parts.append("This shows what victims were armed with, if anything.\n\n")
            elif "time" in query.lower() or "year" in query.lower():
                response_parts.append("This shows how incidents were distributed over time.\n\n")

        # Add a closing statement
        response_parts.append("Is there anything specific about this data you'd like me to analyze further?")

        return {
            "text": "".join(response_parts),
            "pandas_code": pandas_code
        }

    async def process_query(self, session: ClientSession, query: str) -> None:
        """Process a query using the MCP session"""
        # List available tools
        response = await session.list_tools()
        available_tools: list[ToolUnionParam] = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        # Show that we're processing
        with console.status("[bold green]Processing query...[/bold green]"):
            if use_claude:
                # Use Claude API
                res = await anthropic_client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    system=self.system_prompt,
                    max_tokens=8000,
                    messages=self.messages,
                    tools=available_tools,
                )

                # Process Claude's responses recursively until no more tools are called
                await self._process_response(session, res, available_tools)
            else:
                # Simulate Claude's behavior with a simplified approach
                await self._simulate_claude(session, query, available_tools)

    async def _simulate_claude(self, session: ClientSession, query: str, available_tools: list[ToolUnionParam]) -> None:
        """Simulate Claude's behavior when API is not available"""
        tool_results = []

        # Always start by getting dataframe info to understand the structure
        console.print("\nExecuting tool: [bold]get_dataframe_info[/bold]")
        result = await session.call_tool("get_dataframe_info", {})
        tool_result_text = getattr(result.content[0], "text", "")
        tool_results.append({"tool_name": "get_dataframe_info", "result": tool_result_text})
        console.print(f"\nTool result: [dim]{tool_result_text[:200]}...[/dim]")

        # Generate a simulated response to determine what pandas code to execute
        simulated_response = await self.simulate_llm_response(query, tool_results)
        pandas_code = simulated_response.get("pandas_code")

        if pandas_code:
            # Execute the pandas query
            console.print(f"\nExecuting tool: [bold]execute_pandas_query[/bold]")
            console.print(f"Generated pandas code: [bold]{pandas_code}[/bold]")

            result = await session.call_tool("execute_pandas_query", {"query_code": pandas_code})
            tool_result_text = getattr(result.content[0], "text", "")
            tool_results.append({"tool_name": "execute_pandas_query", "result": tool_result_text})
            console.print(f"\nTool result: [dim]{tool_result_text[:200]}...[/dim]")

            # Generate final response with the query results
            final_response = await self.simulate_llm_response(query, tool_results)

            # Print the response
            console.print("\n[bold green]Assistant:[/bold green]")
            console.print(Markdown(final_response["text"]))

            # Add to message history - simplified for simulation
            self.messages.append({"role": "assistant", "content": [{"type": "text", "text": final_response["text"]}]})
        else:
            # Just use the initial response if no pandas code was generated
            console.print("\n[bold green]Assistant:[/bold green]")
            console.print(Markdown(simulated_response["text"]))

            # Add to message history
            self.messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": simulated_response["text"]}]})

    async def _process_response(self, session: ClientSession, res, available_tools):
        """Process a response from Claude, handling any tool calls recursively"""
        assistant_message_content = []
        contains_tool_use = False

        for content in res.content:
            if content.type == "text":
                assistant_message_content.append(content)
                console.print(Markdown(content.text))
            elif content.type == "tool_use":
                contains_tool_use = True
                tool_name = content.name
                tool_args = content.input

                console.print(f"\nExecuting tool: [bold]{tool_name}[/bold]")

                # Special handling for execute_pandas_query to show the code
                if tool_name == "execute_pandas_query" and "query_code" in tool_args:
                    code = tool_args["query_code"]
                    console.print("Pandas Code:")
                    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))
                elif tool_args:
                    console.print(f"Arguments: {tool_args}")

                # Execute tool call
                result = await session.call_tool(tool_name, cast(dict, tool_args))
                tool_result_text = getattr(result.content[0], "text", "")

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

                # Show a preview of the result
                if len(tool_result_text) > 500:
                    console.print(f"\nTool result: [dim]{tool_result_text[:500]}...[/dim]")
                else:
                    console.print(f"\nTool result: [dim]{tool_result_text}[/dim]")

                # Get next response from Claude with the tool result
                follow_up_res = await anthropic_client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=8000,
                    messages=self.messages,
                    tools=available_tools,
                )

                # Process the follow-up response recursively
                await self._process_response(session, follow_up_res, available_tools)
                return

        # If we got here, there were no tool uses in this response, so save it
        if not contains_tool_use and assistant_message_content:
            self.messages.append(
                {"role": "assistant", "content": assistant_message_content}
            )

    async def chat_loop(self, session: ClientSession):
        """Main chat loop"""
        console.print(Panel.fit(
            "[bold]Police Shooting Data Analysis[/bold]\n"
            "Ask questions about the police shooting dataset for analysis with dynamic pandas queries.",
            border_style="green"
        ))
        console.print("Type [bold]'exit'[/bold] to quit, [bold]'help'[/bold] for example questions.")

        while True:
            query = console.input("\n[bold blue]You:[/bold blue] ").strip()

            if query.lower() == 'exit':
                console.print("[bold yellow]Exiting chat...[/bold yellow]")
                break

            if query.lower() == 'help':
                self._print_help()
                continue

            if not query:
                continue

            # Add query to message history
            self.messages.append(
                MessageParam(
                    role="user",
                    content=query,
                )
            )

            # Process the query
            await self.process_query(session, query)

    def _print_help(self):
        """Print help information"""
        help_text = """
# Example Questions

You can ask questions like:

- "What is the racial breakdown of victims in the dataset?"
- "Which states have the highest number of incidents?"
- "What is the age distribution of victims?"
- "Show me the gender breakdown of victims"
- "How many victims showed signs of mental illness?"
- "What weapons were people armed with?"
- "How did the number of incidents change over time?"
- "What percentage of victims were fleeing?"
- "Compare the age distribution between different racial groups"
- "What's the relationship between mental illness and being armed?"
- "Show me incidents where body cameras were used"
- "Which month had the most shootings?"
- "Calculate the average age of victims by state"
        """
        console.print(Markdown(help_text))

    async def run(self):
        """Run the chat client"""
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # Start the chat loop
                await self.chat_loop(session)


if __name__ == "__main__":
    chat = Chat()
    asyncio.run(chat.run())