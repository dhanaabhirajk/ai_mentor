import logging
import sys

from dotenv import load_dotenv

load_dotenv()

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import textwrap
from llama_index.core import Settings

Settings.chunk_size = 512
Settings.chunk_overlap = 50


documents = SimpleDirectoryReader("./data/").load_data()

vector_store = LanceDBVectorStore(
    uri="./lancedb", mode="overwrite", query_type="hybrid"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

retriever = index.as_retriever(similarity_top_k=1)
response = retriever.retrieve("What is 3B parameters?")

print(response[0].get_content())
