import json
from groq import Groq

from classes import VerifiedSummary

groq_client = Groq()

def generate_verified_summary_with_coverage(chunks: list) -> VerifiedSummary:
    # 1. Prepare text context mapped with IDs so the AI can cite them
    context_pool = ""
    for idx, chunk in enumerate(chunks):
        context_pool += f"\n[Chunk ID: {idx} | Section: {chunk.section_name} | Page: {chunk.page_number}]\nText: {chunk.text}\n"

    # PASS 1: Initial Grounded Extraction
    pass1_prompt = f"""
    Analyze the following research paper context. Extract the most critical findings, methods, and metrics.
    For EVERY bullet point you create, you MUST cite the exact Chunk ID, Section, Page, and provide a direct quote.
    
    Respond strictly in this JSON format:
    {{
        "bullet_points": [
            {{"statement": "Summary bullet", "section_name": "Section", "page_number": 1, "source_quote": "exact text"}}
        ]
    }}

    Context:
    {context_pool[:12000]}
    """
    
    response1 = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": pass1_prompt}],
        response_format={"type": "json_object"}
    )
    
    pass1_data = json.loads(response1.choices[0].message.content)

    # PASS 2: Coverage Checker (The Self-Correction Pass)
    coverage_prompt = f"""
    You are a meticulous peer reviewer. Review this initial summary of the paper:
    {json.dumps(pass1_data, indent=2)}

    Now read the full context again. Did the initial summary miss any core performance numbers, datasets, architectural constraints, or major conclusions?
    Identify what was missed, extract those missing points with strict citations, and append them to the existing list.

    Return the final combined list matching the same JSON format.
    
    Full Context:
    {context_pool[:12000]}
    """

    final_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": coverage_prompt}],
        response_format={"type": "json_object"}
    )
    
    return VerifiedSummary(**json.loads(final_response.choices[0].message.content))