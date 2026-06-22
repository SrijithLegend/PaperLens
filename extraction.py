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
from classes import KeyContributions, LimitationsExtractor, MethodologyBreakdown, ProblemStatement, ProblemStatement, ResultsSummary, SectionSummaries , PaperMetadata, DocumentChunk, SimpleSummary, TechnicalSummary, CompletePaperSummary
from chunk import process_pdf_into_chunks, split_text_into_chunks , extract_metadata_from_pdf
from summary import generate_section_summaries, extract_key_contributions , generate_paper_summary

load_dotenv()
groq_client = Groq()

def extract_problem_statement(chunks: list) -> ProblemStatement:
    context = " ".join([c.text for c in chunks if c.section_name in ["Abstract", "Introduction"]])[:6000]
    prompt = f"Analyze this text and extract: 1. The research gap, 2. Why previous methods failed, 3. The exact target problem.\n\nText:\n{context}"
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return ProblemStatement(**json.loads(response.choices[0].message.content))

# --- 11. Extract Methodology Breakdown (Targets Methodology/Proposed Method) ---
def extract_methodology(chunks: list) -> MethodologyBreakdown:
    context = " ".join([c.text for c in chunks if "Method" in c.section_name or "Architecture" in c.section_name])[:8000]
    prompt = f"Analyze the methodology text. Provide the main pipeline, individual components, architecture notes, a step-by-step workflow, and a simple explanation of any algorithms/equations.\n\nText:\n{context}"
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return MethodologyBreakdown(**json.loads(response.choices[0].message.content))

# --- 12. Extract Results Summary (Targets Experiments/Results) ---
def extract_results(chunks: list) -> ResultsSummary:
    context = " ".join([c.text for c in chunks if c.section_name in ["Experiments", "Results"]])[:8000]
    prompt = f"Extract results data: Key metrics/numbers, benchmark datasets used, how it compared to baselines, and where it improved vs where it did not.\n\nText:\n{context}"
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return ResultsSummary(**json.loads(response.choices[0].message.content))

# --- 13. Extract Limitations (Targets Limitations/Conclusion/Discussion) ---
def extract_limitations(chunks: list) -> LimitationsExtractor:
    context = " ".join([c.text for c in chunks if c.section_name in ["Limitations", "Conclusion", "Discussion"]])[:6000]
    prompt = f"Extract all flaws/constraints. Include explicit author admissions, method constraints, edge-case failure modes, and dataset/compute cost limits.\n\nText:\n{context}"
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return LimitationsExtractor(**json.loads(response.choices[0].message.content))