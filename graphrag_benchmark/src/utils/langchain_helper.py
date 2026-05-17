from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3
    )


def run_chain(query: str, context: str):

    llm = create_llm()

    template = """
You are an entertainment assistant.

Context:
{context}

Question:
{query}

Answer clearly and accurately.
"""

    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=template
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "query": query
    })

    return response.content