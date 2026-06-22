import json
from groq import Groq

from classes import BaselineComparison, NoveltyAssessment

groq_client = Groq()

# --- 24. Novelty Detector Engine (Focuses on Intro/Methodology) ---
def detect_paper_novelty(chunks: list) -> NoveltyAssessment:
    context = " ".join([c.text for c in chunks if c.section_name in ["Abstract", "Introduction", "Methodology"]])[:8000]
    
    prompt = f"""
    Analyze this research paper text. Critique its novelty by separating it into three categories:
    1. Real Novelty: What is fundamentally new or breakthrough?
    2. Standard Components Reused: What architectures, datasets, or practices are standardly reused from prior work?
    3. Incremental Improvements: What elements are just tiny, safe, or minor tweaks?
    
    Respond strictly in this JSON format:
    {{
        "real_novelty": "description",
        "standard_components_reused": ["item 1", "item 2"],
        "incremental_improvements": "description"
    }}
    
    Text:
    {context}
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return NoveltyAssessment(**json.loads(response.choices[0].message.content))


# --- 25. Baseline Comparison Extractor (Focuses on Experiments/Results) ---
def extract_baseline_comparisons(chunks: list) -> BaselineComparison:
    context = " ".join([c.text for c in chunks if c.section_name in ["Experiments", "Results"]])[:8000]
    
    prompt = f"""
    Analyze the results text of this paper. Identify the baseline models they compared against, 
    the core metrics used, the scores for both the baseline and the proposed method, and calculate or extract the improvement.
    Also, summarize whether these jumps are substantial or statistically minor.
    
    Respond strictly in this JSON format:
    {{
        "benchmarks": [
            {{
                "baseline_model": "BERT", 
                "metric_name": "Accuracy", 
                "baseline_score": "87.1", 
                "proposed_score": "89.0", 
                "improvement": "+1.9"
            }}
        ],
        "significance_summary": "Summary of whether improvements are large or tiny."
    }}
    
    Text:
    {context}
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return BaselineComparison(**json.loads(response.choices[0].message.content))