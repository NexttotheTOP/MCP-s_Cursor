import re, os
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
import tiktoken
from bs4 import BeautifulSoup
import speech_recognition as sr
import pyttsx3
import time
import sounddevice as sd
import numpy as np
import soundfile as sf
from io import BytesIO
from openai import OpenAI
import requests
from datetime import datetime
from tavily import TavilyClient
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

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

elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

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
    Query the LangGraph documentation for specific information.
    Use this tool when you need to find information about LangGraph or anything related to it.
    
    Args:
        query (str): The specific LangGraph-related question to search for

    Returns:
        str: Relevant documentation content
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

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current information about any topic.
    Use this tool when you need up-to-date information or when the information isn't available in other tools.
    
    Args:
        query (str): The search query
    
    Returns:
        str: Search results from the web
    """
    try:
        client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        
        # Perform the search with Tavily
        search_result = client.search(
            query=query,
            search_depth="advanced",  # Can be "basic" or "advanced"
            max_results=5,
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        
        # Format the results
        formatted_results = []
        
        # Add the AI-generated answer if available
        if search_result.get('answer'):
            formatted_results.append(
                f"Summary: {search_result['answer']}\n"
            )
        
        # Add individual search results
        for result in search_result.get('results', [])[:5]:
            formatted_results.append(
                f"Title: {result.get('title', 'N/A')}\n"
                f"Content: {result.get('content', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Score: {result.get('score', 'N/A')}\n"  # Tavily provides relevance scores
            )
        
        return "\n\n".join(formatted_results) if formatted_results else "No results found."
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def print_memory_contents(memory: ConversationBufferMemory):
    """Helper function to print the current contents of memory."""
    print("\n=== MEMORY CONTENTS ===")
    messages = memory.chat_memory.messages
    for i, msg in enumerate(messages):
        print(f"\n[Message {i+1}]")
        print(f"Type: {type(msg).__name__}")
        print(f"Content: {msg.content}")
    print("\n=====================")

def create_assistant():
    """Create a general-purpose assistant with multiple tools."""
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,  # Slightly higher temperature for more natural conversation
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and conversational AI assistant with access to multiple tools:
        - LangGraph documentation for specific LangGraph questions
        - Web search for current information about any topic

        Always communicate your actions clearly. Before using any tool:
        1. Tell the user what you're about to do
        2. Explain which tool you'll use and why
        3. Share what you find in a conversational way

        For example:
        "Let me search for that information... I'll check the web to get the latest details."
        or
        "I'll look that up in the LangGraph documentation to give you the most accurate answer."

        Maintain a natural conversation while being informative and helpful. If you're going to use
        multiple tools, explain your process: "First, I'll check X, then I'll look up Y..."

        When sharing information:
        - Be concise but thorough
        - Use a conversational tone
        - Cite your sources (whether from documentation or web search)
        - If you're not sure about something, say so

        Remember to speak your actions before using tools, making the interaction more natural and transparent."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent with both tools
    agent = create_openai_functions_agent(
        llm, 
        [langgraph_query_tool, web_search_tool], 
        prompt
    )
    
    # Create the agent executor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[langgraph_query_tool, web_search_tool],
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )
    
    return agent_executor, memory

class SpeechRecognizer:
    """Handle speech recognition using OpenAI's Whisper API."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.client = OpenAI()
        
        # Configure recognition settings for initial audio capture
        self.recognizer.energy_threshold = 500  # Lower threshold for easier activation
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.5
        self.recognizer.phrase_threshold = 0.1
    
    def recognize(self, audio_data):
        """Recognize speech using Whisper API."""
        # Save audio to temporary file
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            # Get WAV data with specific parameters for better quality
            wav_data = audio_data.get_wav_data(
                convert_rate=16000,  # Whisper prefers 16kHz
                convert_width=2  # 16-bit audio
            )
            f.write(wav_data)
        
        try:
            # Use Whisper API with optimized parameters
            with open(temp_file, "rb") as audio:
                transcript = self.client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio,
                    language="en",
                    temperature=0.3,  # Lower temperature for more accurate transcription
                    response_format="text"
                )
            return transcript
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

class VoiceInterface:
    """Handle voice input and output for the agent."""
    
    def __init__(self):
        # Initialize speech recognizer and OpenAI client
        self.speech_recognizer = SpeechRecognizer()
        self.client = OpenAI()
        
        # TTS settings for ElevenLabs
        self.voice = "pNInz6obpgDQGcFmaJgB"  # You can change this to any ElevenLabs voice ID
        
        # Define commands with more variations
        self.commands = {
            'text mode': ['text mode', 'switch to text', 'enable text mode', 'use text', 'type mode'],
            'voice mode': ['voice mode', 'switch to voice', 'enable voice mode', 'use voice', 'speak mode'],
            'memory': ['memory', 'show memory', 'display memory', 'what do you remember', 'conversation history'],
            'quit': ['quit', 'exit', 'goodbye', 'bye', 'stop', 'end', 'finish']
        }
    
    def is_command(self, text: str) -> tuple[bool, str]:
        """Check if the input text matches any command."""
        text = text.lower().strip()
        for command, variations in self.commands.items():
            if any(var in text for var in variations):
                return True, command
        return False, ""
    
    def listen(self) -> str:
        """Listen for voice input and convert to text using Whisper."""
        with sr.Microphone(sample_rate=16000) as source:  # 16kHz for Whisper
            print("\nListening... (speak now)")
            
            # Brief adjustment for ambient noise
            self.speech_recognizer.recognizer.adjust_for_ambient_noise(
                source, duration=0.5
            )
            
            try:
                print("Listening...")
                # Listen for audio input with more flexible timing
                audio = self.speech_recognizer.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=20,  # Longer phrase time for complete thoughts
                )
                
                print("Processing...")
                # Try recognition with Whisper
                text = self.speech_recognizer.recognize(audio)
                print(f"\nYou said: {text}")
                return text
                
            except sr.WaitTimeoutError:
                print("\nNo speech detected within timeout period.")
                self.speak("I didn't hear anything. Please try again.")
                return ""
            except sr.UnknownValueError:
                print("\nCould not understand the audio.")
                self.speak("I didn't understand that. Could you please repeat?")
                return ""
            except Exception as e:
                print(f"\nAn error occurred during voice recognition: {str(e)}")
                self.speak("I encountered an error with voice recognition. Please try again.")
                return ""
    
    def speak(self, text: str):
        """Convert text to speech using ElevenLabs TTS API."""
        try:
            print(f"\nAssistant speaking: {text}")
            
            # Generate audio using ElevenLabs
            audio_generator = elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=self.voice,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_22050_32",
                voice_settings=VoiceSettings(
                    stability=0.0, 
                    similarity_boost=1.0, 
                    style=0.0, 
                    use_speaker_boost=True,
                ),
            )
            
            # Collect all audio data from the generator
            audio_data = BytesIO()
            for chunk in audio_generator:
                audio_data.write(chunk)
            audio_data.seek(0)
            
            # Read and play the audio
            data, samplerate = sf.read(audio_data)
            
            # Ensure audio data is in the correct format (float32)
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # If stereo, convert to mono by averaging channels
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            sd.play(data, samplerate)
            sd.wait()  # Wait until audio is done playing
            
        except Exception as e:
            print(f"\nText-to-speech error: {str(e)}")
            print("Falling back to text-only output")

def main():
    """Main function to set up the assistant and run the interactive loop."""
    # Only run these steps if the vectorstore doesn't exist
    if not os.path.exists("sklearn_vectorstore.parquet"):
        documents, tokens_per_doc = load_langgraph_docs()
        save_llms_full(documents)
        split_docs = split_documents(documents)
        vectorstore = create_vectorstore(split_docs)
    
    # Create the assistant and get memory reference
    assistant, memory = create_assistant()
    
    # Initialize voice interface
    voice = VoiceInterface()
    
    # Interactive loop
    print("\nAI Assistant Ready!")
    print("Start speaking to ask questions (say 'quit' or 'exit' to end)")
    print("Available commands:")
    print("- 'text mode' or 'switch to text': Switch to text input")
    print("- 'voice mode' or 'switch to voice': Switch to voice input")
    print("- 'memory' or 'show memory': Display memory contents")
    print("- 'quit', 'exit', or 'goodbye': End the session")
    
    voice_mode = True  # Start in voice mode
    
    while True:
        try:
            # Get input (either voice or text)
            if voice_mode:
                question = voice.listen()
            else:
                question = input("\nYour question: ")
            
            # Handle empty input
            if not question:
                continue
            
            # Check for commands
            is_cmd, cmd = voice.is_command(question)
            if is_cmd:
                if cmd == 'text mode':
                    voice_mode = False
                    print("\nSwitched to text input mode")
                    if voice_mode:
                        voice.speak("Switching to text input mode")
                    continue
                elif cmd == 'voice mode':
                    voice_mode = True
                    print("\nSwitched to voice input mode")
                    voice.speak("Voice input mode activated")
                    continue
                elif cmd == 'memory':
                    print_memory_contents(memory)
                    if voice_mode:
                        voice.speak("I've displayed the memory contents on screen")
                    continue
                elif cmd == 'quit':
                    if voice_mode:
                        voice.speak("Goodbye!")
                    print("\nGoodbye!")
                    break
            
            # Get response from assistant
            result = assistant.invoke({
                "input": question
            })
            
            # Output response
            if voice_mode:
                voice.speak(result['output'])
            print("\nAssistant:", result['output'])
            
            # Print updated memory after each interaction
            print("\nMemory state after interaction:")
            print_memory_contents(memory)
            
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            if voice_mode:
                voice.speak("I encountered an error. Please try again.")
            continue

if __name__ == "__main__":
    main()