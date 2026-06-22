from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
import shutil
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import fitz
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from groq import Groq
import json
from classes import ChatResponse
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import re
from classes import KeyContributions, SectionSummaries , PaperMetadata, DocumentChunk, SimpleSummary, TechnicalSummary, CompletePaperSummary, VerifiedSummary
from chunk import process_pdf_into_chunks, split_text_into_chunks , extract_metadata_from_pdf
from review import generate_verified_summary_with_coverage
from summary import generate_section_summaries, extract_key_contributions , generate_paper_summary
from extraction import extract_problem_statement, extract_methodology, extract_results, extract_limitations
from question import get_chat_response
from analysis import detect_paper_novelty, extract_baseline_comparisons, BaselineComparison, NoveltyAssessment
from multipaper import evaluate_and_compare_papers, generate_markdown_export, CrossPaperEvaluation


load_dotenv()
app = FastAPI()
groq_client = Groq()

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploaded_papers")
UPLOAD_DIR.mkdir(exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DATABASE_URL = "sqlite:///./paperlens.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 38. Database Model for Personal Library
class SavedPaper(Base):
    __tablename__ = "saved_papers"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    file_path = Column(String)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)
        
@app.get("/upload")
async def read_root():
    return {"message": "Welcome to the PaperLens API"}


@app.post("/upload")
async def upload_research_paper(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        processed_chunks = process_pdf_into_chunks(str(file_path))
        
        return {
            "message": f"Successfully chunked into {len(processed_chunks)} parts.",
            "sample_chunks": processed_chunks[:3] 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")





@app.post("/papers/summarize", response_model=CompletePaperSummary)
async def summarize_paper(file_path: str):

    try:
        chunks = process_pdf_into_chunks(file_path)
        summary = generate_paper_summary(chunks)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")
    

@app.post("/papers/section-summaries", response_model=SectionSummaries)
async def get_section_summaries(file_path: str):
    try:
        chunks = process_pdf_into_chunks(file_path)
        return generate_section_summaries(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/papers/contributions", response_model=KeyContributions)
async def get_contributions(file_path: str):
    try:
        chunks = process_pdf_into_chunks(file_path)
        return extract_key_contributions(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/papers/analyze-deep Dive")
async def deep_dive_analysis(file_path: str):
    try:
        # 1. Process document into structured chunks
        chunks = process_pdf_into_chunks(file_path)
        
        # 2. Run parallel or sequential feature extractions
        problem = extract_problem_statement(chunks)
        methodology = extract_methodology(chunks)
        results = extract_results(chunks)
        limitations = extract_limitations(chunks)
        
        # 3. Return everything grouped together
        return {
            "problem_statement": problem,
            "methodology": methodology,
            "results": results,
            "limitations": limitations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep analysis failed: {str(e)}")
    

@app.post("/papers/verified-summary", response_model=VerifiedSummary)
async def get_verified_summary(file_path: str):
    try:
        chunks = process_pdf_into_chunks(file_path)
        verified_summary =  generate_verified_summary_with_coverage(chunks)
        return verified_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/papers/chat", response_model=ChatResponse)
async def chat_with_paper(file_path: str, question: str):
    try:
        # Re-use our existing chunk pipeline
        chunks = process_pdf_into_chunks(file_path)
        return get_chat_response(question, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/suggested-questions", response_model=List[str])
async def get_suggested_questions():
    # Hardcoded list for the UI to display as quick-action buttons
    return [
        "What is the core idea?",
        "What datasets were used?",
        "What results matter most?",
        "What are the limitations?",
        "Is this paper actually novel?",
        "Explain the methodology simply"
    ]


@app.post("/papers/analyze-novelty", response_model=NoveltyAssessment)
async def get_novelty_analysis(file_path: str):
    try:
        chunks = process_pdf_into_chunks(file_path)
        return detect_paper_novelty(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/papers/extract-baselines", response_model=BaselineComparison)
async def get_baseline_table(file_path: str):
    try:
        chunks = process_pdf_into_chunks(file_path)
        return extract_baseline_comparisons(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/papers/compare", response_model=CrossPaperEvaluation)
async def compare_two_papers(file_path_1: str, file_path_2: str):
    p1_chunks = process_pdf_into_chunks(file_path_1) # From Step 6
    p2_chunks = process_pdf_into_chunks(file_path_2)
    return evaluate_and_compare_papers(p1_chunks, p2_chunks)

# --- Endpoint: Save Paper to Dashboard Library ---
@app.post("/library/save")
async def save_to_library(title: str, file_path: str, db: Session = Depends(get_db)):
    paper = SavedPaper(title=title, file_path=file_path)
    db.add(paper)
    db.commit()
    return {"message": "Paper safely pinned to your personal workspace dashboard library"}

# --- Endpoint: Fetch Dashboard Library ---
@app.get("/library/")
async def get_library(db: Session = Depends(get_db)):
    return db.query(SavedPaper).all()

# --- Endpoint: Export Summary File (Markdown/Doc Note) ---
@app.get("/papers/export/markdown")
async def export_paper_markdown(title: str, problem: str, method: str):
    dummy_data = {"title": title, "problem": problem, "method": method}
    file_stream = generate_markdown_export(dummy_data)
    
    return StreamingResponse(
        file_stream, 
        media_type="text/markdown", 
        headers={"Content-Disposition": f"attachment; filename={title.replace(' ', '_')}_summary.md"}
    )