from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
import shutil
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import fitz
from groq import Groq
import json
import os
from dotenv import load_dotenv
import re
from classes import KeyContributions, LimitationsExtractor, MethodologyBreakdown, ProblemStatement, ResultsSummary, SectionSummaries , PaperMetadata, DocumentChunk, SimpleSummary, TechnicalSummary, CompletePaperSummary
from chunk import process_pdf_into_chunks, split_text_into_chunks , extract_metadata_from_pdf
from summary import generate_section_summaries, extract_key_contributions , generate_paper_summary

load_dotenv()
groq_client = Groq()

def extract_problem_statement(chunks: list) -> ProblemStatement:
    context = " ".join([c.text for c in chunks if c.section_name in ["Abstract", "Introduction"]])[:6000]
    prompt = f"""
    Analyze this text and extract the problem statements. 
    You must respond strictly in valid JSON format matching this structure:
    {{
        "research_gap": "description of the gap in research",
        "previous_method_failures": "why prior approaches failed",
        "exact_target_problem": "the explicit target problem addressed"
    }}

    Text:
    {context}
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return ProblemStatement(**json.loads(response.choices[0].message.content))

# --- 11. Extract Methodology Breakdown (Targets Methodology/Proposed Method) ---
def extract_methodology(chunks: list) -> MethodologyBreakdown:
    context = " ".join([c.text for c in chunks if "Method" in c.section_name or "Architecture" in c.section_name])[:8000]
    prompt = f"""
    Analyze the methodology text of this research paper.
    You must respond strictly in valid JSON format matching this exact structure:
    {{
        "main_pipeline": "overall description of main pipeline",
        "components_modules": ["component 1", "component 2"],
        "architecture_details": "technical design specifications or null",
        "step_by_step_workflow": ["step 1", "step 2"],
        "simplified_algorithms": "plain-english description of equations or logic"
    }}

    Text:
    {context}
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return MethodologyBreakdown(**json.loads(response.choices[0].message.content))

# --- 12. Extract Results Summary (Targets Experiments/Results) ---
def extract_results(chunks: list) -> ResultsSummary:
    context = " ".join([c.text for c in chunks if c.section_name in ["Experiments", "Results"]])[:8000]
    prompt = f"""
    Extract the results data from this research paper text.
    You must respond strictly in valid JSON format matching this exact structure:
    {{
        "key_performance_numbers": ["metric 1", "metric 2"],
        "baseline_comparison": "how it compared to baselines",
        "datasets_used": ["dataset 1", "dataset 2"],
        "improvements_and_tradeoffs": "where it improved vs where it did not"
    }}

    Text:
    {context}
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return ResultsSummary(**json.loads(response.choices[0].message.content))

# --- 13. Extract Limitations (Targets Limitations/Conclusion/Discussion) ---
def extract_limitations(chunks: list) -> LimitationsExtractor:
    context = " ".join([c.text for c in chunks if c.section_name in ["Limitations", "Conclusion", "Discussion"]])[:6000]
    prompt = f"""
    Extract all flaws, constraints, and limitations from this text.
    You must respond strictly in valid JSON format matching this exact structure:
    {{
        "explicit_limitations": ["admission 1", "admission 2"],
        "method_constraints": ["constraint 1", "constraint 2"],
        "failure_cases": ["edge case 1", "edge case 2"],
        "dataset_and_cost_limitations": "compute or data limits description or null"
    }}

    Text:
    {context}
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return LimitationsExtractor(**json.loads(response.choices[0].message.content))