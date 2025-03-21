import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

# Load environment variables from .env file
load_dotenv()

# Define common path to the repo locally
PATH = "/Users/wout_vp/Code/test-mcp/"

# Create an MCP server
mcp = FastMCP('Langgraph-Docs-MCP-Server')

# Adding a tool to query the LangGraph documentation
@mcp.tool()
def langgraph_query_tool(query: str):
    """
    Query the Langgraph Documentation using a retriever

    Args:
        query (str): The question to query the documentation with

    Returns:
        str: A str of the retrieved documents
    """

    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY")),
        persist_path=PATH + "sklearn_vectorstore.parquet",
        serializer='parquet').as_retriever(search_kwargs={"k": 3}
    )

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return formatted_context


# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
@mcp.resource("docs://langgraph/full")
def get_all_langgraph_docs() -> str:
    """
    Get all the LangGraph documentation. Returns the contents of the file llms_full.txt,
    which contains a curated set of LangGraph documentation (~300k tokens). This is useful
    for a comprehensive response to questions about LangGraph.

    Args: None

    Returns:
        str: The contents of the LangGraph documentation
    """

    # Local path to the LangGraph documentation
    doc_path = PATH + "llms_full.txt"
    try:
        with open(doc_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading log file: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
