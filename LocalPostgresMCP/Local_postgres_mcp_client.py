import asyncio
from dataclasses import dataclass, field
from typing import Union, cast

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolUnionParam, ToolUseBlock
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

anthropic_client = anthropic.AsyncAnthropic()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["./Local_postgres_mcp_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)


@dataclass
class Chat:
    messages: list[MessageParam] = field(default_factory=list)

    system_prompt: str = """You are a master PostgreSQL assistant. 
    Your job is to use the tools at your disposal to execute SQL queries on a PostgreSQL database and provide the results to the user.

    Available tools:
    - query_postgres: Execute any PostgreSQL query
    - list_tables: List all tables in the database
    - describe_table: Get the structure of a specific table (requires table_name parameter)

    When using describe_table, you MUST specify a table_name parameter.

    Always help the user explore and understand their data. If they need to connect to a different database,
    you can use the connection_params parameter in the query_postgres tool.

    Always format your SQL queries properly. When asked about a table's contents, first describe the table using describe_table, 
    then use query_postgres to show the data."""

    async def process_query(self, session: ClientSession, query: str) -> None:
        response = await session.list_tools()
        available_tools: list[ToolUnionParam] = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        # Initial Claude API call
        res = await anthropic_client.messages.create(
            model="claude-3-7-sonnet-latest",
            system=self.system_prompt,
            max_tokens=8000,
            messages=self.messages,
            tools=available_tools,
        )

        # Process Claude's responses recursively until no more tools are called
        await self._process_response(session, res, available_tools)

    async def _process_response(self, session: ClientSession, res, available_tools):
        """Process a response from Claude, handling any tool calls recursively"""
        assistant_message_content = []
        contains_tool_use = False

        for content in res.content:
            if content.type == "text":
                assistant_message_content.append(content)
                print(content.text)
            elif content.type == "tool_use":
                contains_tool_use = True
                tool_name = content.name
                tool_args = content.input

                print(f"\nExecuting tool: {tool_name}")
                if tool_name == "describe_table" and "table_name" not in tool_args:
                    print("Error: Missing table_name parameter for describe_table")
                    tool_args["table_name"] = "language"  # Default to language table
                    print(f"Defaulting to table: language")

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

                print(f"\nTool result: {tool_result_text}")

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
        print("\nPostgreSQL Query Assistant (Type 'exit' to quit)")
        print("------------------------------------------------")

        while True:
            query = input("\nQuery: ").strip()

            if query.lower() == 'exit':
                print("Exiting chat...")
                break

            self.messages.append(
                MessageParam(
                    role="user",
                    content=query,
                )
            )

            await self.process_query(session, query)

    async def run(self):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                await self.chat_loop(session)


if __name__ == "__main__":
    chat = Chat()
    asyncio.run(chat.run())