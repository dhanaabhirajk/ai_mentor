import chainlit as cl

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

SIMPLE_RAG_PROMPT = """You are AI mentor to help user with their user query using the availabkle information.

Available Information to answer user query:
{available_info}

User Query:
{user_query}

Answer:
"""

SIMPLE_RAG_PROMPT_V1 = """You are AI mentor to help user with their user query using the below Source Information only.

Instructions:
1. Your goal is to answer the user query with only the below available information.
2. Don't include extra information other than 
2. If the provided Source Information does not contain the necessary details to answer the query, respond with: "I'm sorry, I can't help with that."

Source Information:
{available_info}

User Query:
{user_query}

Answer:
"""

SIMPLE_RAG_PROMPT_TEMPLATE = PromptTemplate(SIMPLE_RAG_PROMPT)

vector_store = LanceDBVectorStore(
    uri="./lancedb", mode="overwrite", query_type="hybrid"
)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
)
rag_llm = OpenAI(model="gpt-4o-mini")


@cl.step(type="tool")
async def retrieve(user_query: str):
    # Fake tool
    retriever = index.as_retriever(similarity_top_k=1)
    response = retriever.retrieve(user_query)

    source_text = response[0].get_content()
    return source_text


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """
    user_query = message.content
    # Call the tool
    tool_res = await retrieve(user_query)

    response = rag_llm.chat(
        [
            ChatMessage(
                role="system",
                content=SIMPLE_RAG_PROMPT_TEMPLATE.format(
                    user_query=user_query,
                    available_info=tool_res,
                ),
            )
        ]
    )

    print(response)
    await cl.Message(content=response.message.content).send()