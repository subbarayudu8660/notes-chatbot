import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model='llama3-70b-8192',
        temperature=0
    )


    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key='answer', # Crucial: tells memory which part of the output to save
        max_token_limit=1000
    )

    # 3. The "Judge" Prompt (for map_rerank)
    # This is what the AI uses to score each note chunk individually.
    rerank_template = """
    Use the following piece of context to answer the user's question. 
    If you cannot answer the question based on this context, give it a score of 0.
    If the context is perfectly relevant, give it a score of 100.

    **Context**: {context}
    **Question**: {question}

    **Instructions**:
    1. Provide the answer in a clear, factual, and respectful tone.
    2. If the answer isn't in the context, say "I don't have that in my notes."
    3. End your response with a score of your confidence (0-100).

    Helpful Answer: [Your answer]
    Score: [Your score]
    """
    
    RERANK_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=rerank_template
    )

    # 4. Tie it all together in a Conversational Chain
    # This is the "Manager" that handles the rewriting and the ranking.
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        # This tells the chain to use our 'map_rerank' logic for the final answer
        combine_docs_chain_kwargs={
            "chain_type": "map_rerank",
            "prompt": RERANK_PROMPT
        },
        return_source_documents=True,
        verbose=True # Set to True so you can see the "Score" and "Rewriting" in your terminal
    )