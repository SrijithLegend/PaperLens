import json
from groq import Groq

from classes import CrossPaperEvaluation

groq_client = Groq()

def evaluate_and_compare_papers(paper1_chunks: list, paper2_chunks: list) -> CrossPaperEvaluation:
    # 1. Compile focused summaries of both papers to fit token windows comfortably
    p1_summary = " ".join([c.text for c in paper1_chunks if c.section_name in ["Abstract", "Introduction", "Results"]])[:6000]
    p2_summary = " ".join([c.text for c in paper2_chunks if c.section_name in ["Abstract", "Introduction", "Results"]])[:6000]
    
    prompt = f"""
    You are an elite academic peer reviewer. Critically compare these two research papers.
    
    Paper 1 Data:
    {p1_summary}
    
    Paper 2 Data:
    {p2_summary}
    
    Construct a complete cross-evaluation. For the 'matrix_table', build a crisp single-row synthesis for each paper. 
    For the verdicts, pass direct judgment on which paper breaks genuinely new ground vs which paper displays more rigorous testing/baselines.
    
    Respond strictly in this JSON format:
    {{
        "matrix_table": [
            {{"title": "Paper 1 Title", "problem": "...", "method": "...", "dataset": "...", "metric": "...", "main_result": "...", "limitation": "..."}},
            {{"title": "Paper 2 Title", "problem": "...", "method": "...", "dataset": "...", "metric": "...", "main_result": "...", "limitation": "..."}}
        ],
        "problem_comparison": "Deep comparison of their targeted issues",
        "method_comparison": "Technical breakdown of structural differences",
        "result_comparison": "Side by side empirical outcome tracking",
        "novelty_verdict": "Clear verdict stating which is more novel and why",
        "rigor_verdict": "Clear verdict stating which is more empirically rigorous and why"
    }}
    """
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return CrossPaperEvaluation(**json.loads(response.choices[0].message.content))


from fastapi.responses import StreamingResponse
import io

def generate_markdown_export(data: CrossPaperEvaluation) -> io.BytesIO:
    md_content = f"# Paper Analysis Summary: {data.get('title', 'Research Paper')}\n\n"
    md_content += f"## Problem Statement\n{data.get('problem', 'N/A')}\n\n"
    md_content += f"## Core Methodology\n{data.get('method', 'N/A')}\n\n"
    md_content += f"## Main Results\n{data.get('results', 'N/A')}\n\n"
    md_content += f"## Limitations\n{data.get('limitations', 'N/A')}\n"
    
    return io.BytesIO(md_content.encode("utf-8"))