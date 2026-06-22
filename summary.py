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
from classes import KeyContributions, SectionSummaries , PaperMetadata, DocumentChunk, SimpleSummary, TechnicalSummary, CompletePaperSummary
from chunk import process_pdf_into_chunks, split_text_into_chunks

load_dotenv()
groq_client = Groq()

def generate_section_summaries(chunks: list) -> SectionSummaries:

    sections_map = {}
    for chunk in chunks:
        if chunk.section_name not in sections_map:
            sections_map[chunk.section_name] = []
        sections_map[chunk.section_name].append(chunk.text)
    
    final_summaries = {}
    
    # 2. Loop through each section and summarize it individually
    for section_title, text_list in sections_map.items():
        # Combine text for this section (limit to avoid token issues if a section is massive)
        section_text = " ".join(text_list)[:4000] 
        
        prompt = f"Summarize the following '{section_title}' section of a research paper in 2-3 concise sentences focusing on the core details:\n\n{section_text}"
        
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192", # Fast, smaller model works perfectly for short sections
            messages=[{"role": "user", "content": prompt}]
        )
        
        final_summaries[section_title] = response.choices[0].message.content.strip()
        
    return SectionSummaries(summaries=final_summaries)


def extract_key_contributions(chunks: list) -> KeyContributions:
    # 1. Isolate text from Abstract or Introduction where contributions live
    intro_text = " ".join([
        chunk.text for chunk in chunks 
        if chunk.section_name in ["Abstract", "Introduction"]
    ])[:6000] # Safeguard limit
    
    prompt = f"""
    Analyze the introduction/abstract text of this research paper. 
    Extract the main contributions of this work as a list of short, direct bullet points (e.g., 'Proposes a new model architecture', 'Improves inference speed').
    
    Respond strictly in this JSON format:
    {{
        "contributions": ["bullet 1", "bullet 2", "bullet 3"]
    }}
    
    Text:
    {intro_text}
    """
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192", # Higher reasoning model for extracting precise outcomes
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result_json = json.loads(response.choices[0].message.content)
    return KeyContributions(**result_json)


def generate_paper_summary(chunks: list) -> CompletePaperSummary:

    full_text = " ".join([chunk.text for chunk in chunks[:15]]) 
    
    prompt = f"""
    You are an expert research assistant. Analyze the following research paper text and generate a structured summary.
    
    Provide the response in JSON format matching this structure:
    {{
        "simple_english": {{
            "problem_solved": "What problem does this paper solve?",
            "why_it_matters": "Why does this matter to the world/field?",
            "what_they_built": "What did they build or propose?"
        }},
        "detailed_technical": {{
            "how_it_works": "How does the underlying methodology work?",
            "proposed_system": "What specific model or architecture is proposed?",
            "experiments_run": "What experiments did they run?",
            "results_obtained": "What precise results did they achieve?"
        }}
    }}

    Paper Text:
    {full_text}
    """
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192", 
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    result_json = json.loads(response.choices[0].message.content)
    return CompletePaperSummary(**result_json)