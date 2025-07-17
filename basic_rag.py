!pip install -q -U \
     Sentence-transformers \
     langchain \
     langchain-groq \
     langchain-community \
     langchain-huggingface \
     einops \
     faiss_cpu

"""# Import related libraries related to Langchain, HuggingfaceEmbedding"""

!pip install google-search-results

from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from serpapi import GoogleSearch
import os
from google.colab import userdata # Import userdata
from langchain_huggingface import HuggingFaceEndpoint # Import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub # Keep for now if other cells use it

# Load environment variables from Colab secrets
serpapi_key = userdata.get("SERPAPI_API_KEY")
huggingface_api_key = userdata.get("HF_TOKEN") # Assuming HF_TOKEN is also in secrets

# Validate keys
if not serpapi_key:
    print("SERPAPI_API_KEY not found in environment variables. Please set it in Colab secrets.")
    serpapi_key = None # Set to None to handle gracefully
if not huggingface_api_key:
    print("HF_TOKEN not found in environment variables. Please set it in Colab secrets.")
    huggingface_api_key = None # Set to None to handle gracefully


# Define search query (using the query from the previous cell for consistency)
query = "What is machine learning?" # Or use the query defined in this cell if preferred

params = {
    "q": query,
    "api_key": serpapi_key,
    "engine": "google",
    "num": 5
}

# Fetch search results
documents = [] # Initialize documents list
if serpapi_key: # Only fetch if serpapi_key is available
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        documents = [
            Document(
                page_content=result.get("snippet", ""),
                metadata={"source": result.get("link")}
            )
            for result in organic_results
        ]
    except Exception as e:
        print(f"Error fetching SerpApi results: {e}")


# Chunk documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vectorstore
if chunks and huggingface_api_key: # Only create vectorstore and LLM if there are chunks and HF key
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Initialize HuggingFaceEndpoint LLM
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-large",
        huggingfacehub_api_token=huggingface_api_key,
        temperature=0.7,
        max_new_tokens=256
    )

    # Prompt template
    template = """Use the following context to answer the question:
    {context}
    Question: {question}
    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # 👈 this enables full output
    )

    # Run the QA chain
    try:
        output = qa_chain({"query": query})
        print("\nAnswer:\n", output["result"])
        print("\nRetrieved Documents:\n")
        for i, doc in enumerate(output["source_documents"]):
            print(f"{i+1}. {doc.metadata.get('source')}\n{doc.page_content[:300]}...\n")
    except Exception as e:
        print(f"Error running QA chain: {e}")
else:
    if not serpapi_key:
        print("Skipping SerpApi search due to missing SERPAPI_API_KEY.")
    if not huggingface_api_key:
        print("Skipping LLM initialization and QA chain due to missing HF_TOKEN.")
    if not chunks and serpapi_key: # Only print this if search was attempted but no chunks were created
         print("No documents were retrieved or chunked, cannot create vectorstore or run QA chain.")

docs = vectorstore.similarity_search("What is machine learning?", k=3)
for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---\n{doc.page_content}\nSource: {doc.metadata.get('source')}")