import json
from groq import Groq
from classes import ChatResponse
from rank_bm25 import BM25Okapi

groq_client = Groq()

def get_chat_response(user_question: str, chunks: list) -> ChatResponse:
    # 1. Tokenize chunks for keyword matching
    tokenized_corpus = [chunk.text.lower().split(" ") for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 2. Find the top 3 most relevant chunks to the question
    tokenized_query = user_question.lower().split(" ")
    top_chunks = bm25.get_top_n(tokenized_query, chunks, n=3)
    
    # 3. Format those top chunks as context evidence for the LLM
    context_str = ""
    for idx, c in enumerate(top_chunks):
        context_str += f"\n[Source {idx} | Page {c.page_number} | Section {c.section_name}]\nText: {c.text}\n"
        
    # 4. Craft the prompt forcing strict adherence to the provided context
    prompt = f"""
    You are an expert academic AI assistant. Answer the user's question using ONLY the provided paper text snippets below.
    For every claim you make, you must reference which 'Source X' it came from.
    
    Respond STRICTLY in this JSON format:
    {{
        "answer": "Your clear, direct answer to the user.",
        "citations": [
            {{"page_number": 1, "section_name": "Introduction", "excerpt": "Exact text snippet supporting your answer"}}
        ]
    }}
    
    Context from Paper:
    {context_str}
    
    User Question: {user_question}
    """
    
    # 5. Get structured answer from LLM
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192", 
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return ChatResponse(**json.loads(response.choices[0].message.content))