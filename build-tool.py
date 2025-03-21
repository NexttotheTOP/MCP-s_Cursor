import re, os
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
import tiktoken
from bs4 import BeautifulSoup

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables from .env file
load_dotenv()

def count_tokens(text, model="cl100k_base"):
    """
    Count the number of tokens in the text using tiktoken.
    """
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))

def bs4_extractor(html: str) -> str:
    """Extract text content from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "lxml")
    main_content = soup.find("article", class_="md-content__inner")
    content = main_content.get_text() if main_content else soup.text
    content = re.sub(r"\n\n+", "\n\n", content).strip()
    return content

def load_langgraph_docs():
    """Load LangGraph documentation from the official website."""
    print("Loading LangGraph documentation...")

    urls = ["https://langchain-ai.github.io/langgraph/concepts/",
     "https://langchain-ai.github.io/langgraph/how-tos/",
     "https://langchain-ai.github.io/langgraph/tutorials/workflows/",  
     "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
     "https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/"]

    docs = []
    for url in urls:
        loader = RecursiveUrlLoader(
            url,
            max_depth=5,
            extractor=bs4_extractor,
        )
        docs_lazy = loader.lazy_load()
        for d in docs_lazy:
            docs.append(d)

    print(f"Loaded {len(docs)} documents from LangGraph documentation.")
    
    total_tokens = 0
    tokens_per_doc = []
    for doc in docs:
        total_tokens += count_tokens(doc.page_content)
        tokens_per_doc.append(count_tokens(doc.page_content))
    
    print(f"Total tokens in loaded documents: {total_tokens}")
    return docs, tokens_per_doc

def save_llms_full(documents):
    """Save the documents to a file."""
    output_filename = "llms_full.txt"
    with open(output_filename, "w") as f:
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Unknown URL')
            f.write(f"DOCUMENT {i+1}\n")
            f.write(f"SOURCE: {source}\n")
            f.write("CONTENT:\n")
            f.write(doc.page_content)
            f.write("\n\n" + "="*80 + "\n\n")
    print(f"Documents concatenated into {output_filename}")

def split_documents(documents):
    """Split documents into smaller chunks for improved retrieval."""
    print("Splitting documents...")
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=8000,  
        chunk_overlap=500  
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} chunks from documents.")
    
    total_tokens = 0
    for doc in split_docs:
        total_tokens += count_tokens(doc.page_content)
    
    print(f"Total tokens in split documents: {total_tokens}")
    return split_docs

def create_vectorstore(splits):
    """Create a vector store from document chunks using SKLearnVectorStore."""
    print("Creating SKLearnVectorStore...")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    persist_path = os.getcwd()+"/sklearn_vectorstore.parquet"
    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet",
    )
    print("SKLearnVectorStore created successfully.")
    
    vectorstore.persist()
    print("SKLearnVectorStore was persisted to", persist_path)
    return vectorstore

@tool
def langgraph_query_tool(query: str) -> str:
    """
    Query the LangGraph documentation using a retriever.
    
    Args:
        query (str): The question to query the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY")),
        persist_path=os.getcwd()+"/sklearn_vectorstore.parquet",
        serializer="parquet"
    ).as_retriever(search_kwargs={"k": 1})

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return formatted_context

def print_memory_contents(memory: ConversationBufferMemory):
    """Helper function to print the current contents of memory."""
    print("\n=== MEMORY CONTENTS ===")
    messages = memory.chat_memory.messages
    for i, msg in enumerate(messages):
        print(f"\n[Message {i+1}]")
        print(f"Type: {type(msg).__name__}")
        print(f"Content: {msg.content}")
    print("\n=====================")

def create_langgraph_agent():
    """Create an agent that can answer questions about LangGraph using the documentation."""
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful expert on LangGraph who answers questions based on the LangGraph documentation.
        Use the langgraph_query_tool to search the documentation when needed.
        Always base your answers on the retrieved documentation.
        If you don't find relevant information in the documentation, say so.
        Be concise but thorough in your explanations.
        
        When referring to previous conversation, use the chat history to maintain context."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_openai_functions_agent(llm, [langgraph_query_tool], prompt)
    
    # Create the agent executor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[langgraph_query_tool],
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )
    
    return agent_executor, memory

def main():
    """Main function to set up the vectorstore and create the agent."""
    # Only run these steps if the vectorstore doesn't exist
    if not os.path.exists("sklearn_vectorstore.parquet"):
        documents, tokens_per_doc = load_langgraph_docs()
        save_llms_full(documents)
        split_docs = split_documents(documents)
        vectorstore = create_vectorstore(split_docs)
    
    # Create the agent and get memory reference
    agent, memory = create_langgraph_agent()
    
    # Interactive loop
    print("\nLangGraph Documentation Assistant Ready!")
    print("Ask questions about LangGraph (type 'quit' to exit)")
    print("Type 'memory' to see current memory contents")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['quit', 'exit']:
            break
        
        # Check if user wants to see memory contents
        if question.lower() == 'memory':
            print_memory_contents(memory)
            continue
            
        # Get response from agent
        result = agent.invoke({
            "input": question
        })
        
        # Update chat history  -----   not needed since we are using the memory from langgraph
        #chat_history.extend([
        #    HumanMessage(content=question),
        #    AIMessage(content=result['output'])
        #])
        
        # Print the response
        print("\nAssistant:", result['output'])
        
        # Print updated memory after each interaction
        print("\nMemory state after interaction:")
        print_memory_contents(memory)

if __name__ == "__main__":
    main()