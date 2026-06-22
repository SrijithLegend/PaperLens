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

load_dotenv()
groq_client = Groq()


TARGET_SECTIONS = [
    "ABSTRACT", "INTRODUCTION", "RELATED WORK", 
    "METHODOLOGY", "PROPOSED METHOD", "EXPERIMENTS", 
    "RESULTS", "LIMITATIONS", "CONCLUSION"
]

def extract_metadata_from_pdf(file_path: str) -> PaperMetadata:

    doc = fitz.open(file_path)
    first_pages_text = ""
    for page in doc[:2]: 
        first_pages_text += page.get_text()
    doc.close()

    prompt = f"Extract the title, authors, publication year, abstract, and venue/journal from this academic text:\n\n{first_pages_text}"
    
    response = groq_client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"} 
    )
    
    data = json.loads(response.choices[0].message.content)
    return PaperMetadata(**data)


def process_pdf_into_chunks(file_path: str, chunk_size: int = 1000) -> list[DocumentChunk]:
    doc = fitz.open(file_path)
    chunks = []
    
    current_section = "Introduction" 
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        
        lines = text.split("\n")
        page_buffer = []
        
        for line in lines:
            clean_line = line.strip().upper()
            

            matched_section = next((sec for sec in TARGET_SECTIONS if sec in clean_line), None)
            
            if matched_section and len(clean_line) < 50:

                if page_buffer:
                    chunks.extend(split_text_into_chunks(" ".join(page_buffer), current_section, page_num, chunk_size))
                    page_buffer = []
                
                current_section = matched_section.title()
            
            page_buffer.append(line)
            

        if page_buffer:
            chunks.extend(split_text_into_chunks(" ".join(page_buffer), current_section, page_num, chunk_size))
            
    doc.close()
    return chunks


def split_text_into_chunks(text: str, section: str, page_num: int, chunk_size: int) -> list[DocumentChunk]:
    """Splits text into smaller pieces of roughly `chunk_size` characters."""
    words = text.split()
    generated_chunks = []
    current_chunk_words = []
    current_length = 0
    
    for word in words:
        current_chunk_words.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunk_text = " ".join(current_chunk_words)
            generated_chunks.append(DocumentChunk(text=chunk_text, section_name=section, page_number=page_num))
            current_chunk_words = []
            current_length = 0
            
    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        generated_chunks.append(DocumentChunk(text=chunk_text, section_name=section, page_number=page_num))
        
    return generated_chunks