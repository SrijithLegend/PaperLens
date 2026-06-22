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
from typing import List, Dict

class PaperMetadata(BaseModel):
    title: str
    authors: List[str]
    publication_year: Optional[int] = None
    abstract: str
    venue: Optional[str] = None


class DocumentChunk(BaseModel):
    text: str
    section_name: str
    page_number: int


class SimpleSummary(BaseModel):
    problem_solved: str
    why_it_matters: str
    what_they_built: str


class TechnicalSummary(BaseModel):
    how_it_works: str
    proposed_system: str
    experiments_run: str
    results_obtained: str


class CompletePaperSummary(BaseModel):
    simple_english: SimpleSummary
    detailed_technical: TechnicalSummary


class KeyContributions(BaseModel):
    contributions: List[str]  


class SectionSummaries(BaseModel):
    summaries: Dict[str, str]


class ProblemStatement(BaseModel):
    research_gap: str
    previous_method_failures: str
    exact_target_problem: str

# 11. Methodology Breakdown Schema
class MethodologyBreakdown(BaseModel):
    main_pipeline: str
    components_modules: List[str]
    architecture_details: Optional[str] = None
    step_by_step_workflow: List[str]
    simplified_algorithms: str

# 12. Results Summary Schema
class ResultsSummary(BaseModel):
    key_performance_numbers: List[str]
    baseline_comparison: str
    datasets_used: List[str]
    improvements_and_tradeoffs: str

# 13. Limitations Schema
class LimitationsExtractor(BaseModel):
    explicit_limitations: List[str]
    method_constraints: List[str]
    failure_cases: List[str]
    dataset_and_cost_limitations: Optional[str] = None


class GroundedBulletPoint(BaseModel):
    statement: str        # The summary bullet point (e.g., "Improves F1 by 3.2%")
    section_name: str     # e.g., "Results"
    page_number: int      # e.g., 7
    source_quote: str     # The exact sentence/paragraph pulled from the paper

class VerifiedSummary(BaseModel):
    bullet_points: List[GroundedBulletPoint]


class SourceCitation(BaseModel):
    page_number: int
    section_name: str
    excerpt: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[SourceCitation]


class NoveltyAssessment(BaseModel):
    real_novelty: str                     # The breakthrough or unique angle
    standard_components_reused: List[str] # Stuff they borrowed (e.g., standard Adam optimizer, ResNet backbone)
    incremental_improvements: str          # Small tweaks rather than fundamental shifts

# 25. Baseline Table Row Schema
class BaselineRow(BaseModel):
    baseline_model: str
    metric_name: str                      # e.g., "Accuracy", "F1 Score", "Inference Time"
    baseline_score: str
    proposed_score: str
    improvement: str                      # e.g., "+1.9", "-0.5"

class BaselineComparison(BaseModel):
    benchmarks: List[BaselineRow]
    significance_summary: str


class PaperComparisonRow(BaseModel):
    title: str
    problem: str
    method: str
    dataset: str
    metric: str
    main_result: str
    limitation: str

class CrossPaperEvaluation(BaseModel):
    matrix_table: List[PaperComparisonRow]
    problem_comparison: str
    method_comparison: str
    result_comparison: str
    novelty_verdict: str  # Critical analysis of which is genuinely more novel
    rigor_verdict: str