# LangGraph MCP Server Example

This repository demonstrates a quick and easy way to set up a Model Context Protocol (MCP) server in Cursor IDE. The example showcases how to integrate LangGraph documentation querying capabilities using a Language Model (LLM) instead of traditional IDE features.

Made based on the langchain tutorial: "https://mirror-feeling-d80.notion.site/MCP-From-Scratch-1b9808527b178040b5baf83a991ed3b2"

## Overview

This project implements a simple MCP server that allows querying LangGraph documentation using embeddings and vector search. It serves as an example of how to build AI-powered documentation tools that can be integrated with various development environments.

## Features

- Easy-to-setup MCP server implementation
- LangGraph documentation querying using embeddings
- Vector store integration for efficient document retrieval
- Simple API for querying documentation

## Inspecting the MCP Server

To inspect the MCP server and its capabilities, you can use the MCP Inspector tool:

```bash
npx @modelcontextprotocol/inspector
```

This command will help you explore the available endpoints and test the functionality of the MCP server.

## Getting Started


## Implementation Details

The server is implemented using FastMCP and includes:
- Vector store-based document retrieval
- OpenAI embeddings integration
- Custom resource endpoints for accessing documentation

This example demonstrates how LLMs can be used to enhance the development experience by providing intelligent documentation access, serving as an alternative to traditional IDE-based documentation lookup. 