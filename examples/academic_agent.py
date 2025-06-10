"""
AgentFlow Academic Research Workflow Example

This example demonstrates how to create a comprehensive academic research workflow
using OpenAI's integration capabilities. The workflow includes:
1. Literature Review
2. Research Methodology
3. Data Analysis
4. Paper Writing

Requirements:
- OpenAI API key set in environment variable OPENAI_API_KEY
"""

import os
import json
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_env_var(name: str) -> str:
    """Get an environment variable or raise an error if it's not set."""
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value

# Pydantic models for structured outputs
class LiteratureReview(BaseModel):
    stage: str = Field(default="literature_review")
    domain: str = Field(description="Research domain")
    sources: List[str] = Field(description="Key sources reviewed")
    key_findings: List[str] = Field(description="Significant findings")
    research_gaps: List[str] = Field(description="Identified research gaps")

class ResearchMethodology(BaseModel):
    stage: str = Field(default="methodology")
    domain: str = Field(description="Research domain")
    methodology_type: str = Field(description="Type of research methodology")
    research_questions: List[str] = Field(description="Research questions")
    methods: List[str] = Field(description="Research methods")
    limitations: List[str] = Field(description="Potential limitations")

class DataAnalysis(BaseModel):
    stage: str = Field(default="data_analysis")
    domain: str = Field(description="Research domain")
    analysis_method: str = Field(description="Data analysis method")
    key_findings: List[str] = Field(description="Analysis findings")
    implications: List[str] = Field(description="Research implications")
    future_work: List[str] = Field(description="Suggestions for future work")

class PaperDraft(BaseModel):
    stage: str = Field(default="paper_writing")
    domain: str = Field(description="Research domain")
    title: str = Field(description="Paper title")
    abstract: str = Field(description="Paper abstract")
    sections: List[str] = Field(description="Main sections of the paper")
    key_contributions: List[str] = Field(description="Key contributions of the research")

async def conduct_literature_review(domain: str, academic_level: str) -> LiteratureReview:
    """Conduct a literature review for the given domain."""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a research assistant conducting a literature review."},
                {"role": "user", "content": f"Conduct a comprehensive literature review in the domain of {domain}."}
            ],
            temperature=0.7
        )
        
        # Parse response and create LiteratureReview object
        return LiteratureReview(
            domain=domain,
            sources=["Source1", "Source2", "Source3"],
            key_findings=["Finding1", "Finding2"],
            research_gaps=["Gap1", "Gap2"]
        )
    except Exception as e:
        print(f"Error in literature review: {e}")
        return LiteratureReview(
            domain=domain,
            sources=["Error fetching sources"],
            key_findings=["Error identifying findings"],
            research_gaps=["Error identifying gaps"]
        )

async def develop_methodology(domain: str, academic_level: str, lit_review: LiteratureReview) -> ResearchMethodology:
    """Develop research methodology based on literature review."""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a research assistant developing a research methodology."},
                {"role": "user", "content": f"Based on the literature review findings, develop a research methodology for {domain}."}
            ],
            temperature=0.7
        )
        
        return ResearchMethodology(
            domain=domain,
            methodology_type="Qualitative",
            research_questions=["Question1", "Question2"],
            methods=["Method1", "Method2"],
            limitations=["Limitation1"]
        )
    except Exception as e:
        print(f"Error in methodology development: {e}")
        return ResearchMethodology(
            domain=domain,
            methodology_type="Error",
            research_questions=["Error"],
            methods=["Error"],
            limitations=["Error"]
        )

async def analyze_data(domain: str, academic_level: str, methodology: ResearchMethodology) -> DataAnalysis:
    """Simulate data analysis based on methodology."""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a research assistant conducting data analysis."},
                {"role": "user", "content": f"Based on the research methodology, simulate a data analysis for {domain}."}
            ],
            temperature=0.7
        )
        
        return DataAnalysis(
            domain=domain,
            analysis_method="Statistical",
            key_findings=["Finding1", "Finding2"],
            implications=["Implication1"],
            future_work=["Suggestion1", "Suggestion2"]
        )
    except Exception as e:
        print(f"Error in data analysis: {e}")
        return DataAnalysis(
            domain=domain,
            analysis_method="Error",
            key_findings=["Error"],
            implications=["Error"],
            future_work=["Error"]
        )

async def write_paper(domain: str, academic_level: str, analysis: DataAnalysis) -> PaperDraft:
    """Write a paper draft based on the analysis."""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a research assistant writing an academic paper."},
                {"role": "user", "content": f"Based on the data analysis, write a paper draft for research in {domain}."}
            ],
            temperature=0.7
        )
        
        return PaperDraft(
            domain=domain,
            title="Research Paper Draft",
            abstract="Abstract of the research paper",
            sections=["Introduction", "Methodology", "Results", "Discussion"],
            key_contributions=["Contribution1", "Contribution2"]
        )
    except Exception as e:
        print(f"Error in paper writing: {e}")
        return PaperDraft(
            domain=domain,
            title="Error Draft",
            abstract="Error generating abstract",
            sections=["Error"],
            key_contributions=["Error"]
        )

def generate_pdf_report(paper_draft: PaperDraft, output_path: str = "research_paper.pdf"):
    """Generate a PDF report from the paper draft."""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom style for title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=16,
            textColor='black',
            spaceAfter=12
        )
        
        # Content elements
        content = [
            Paragraph(paper_draft.title, title_style),
            Paragraph(f"Domain: {paper_draft.domain}", styles['Normal']),
            Spacer(1, 12),
            Paragraph("Abstract", styles['Heading2']),
            Paragraph(paper_draft.abstract, styles['Normal']),
            Spacer(1, 12)
        ]
        
        # Add sections
        for section in paper_draft.sections:
            content.extend([
                Paragraph(section, styles['Heading3']),
                Paragraph(f"Placeholder content for {section} section", styles['Normal']),
                Spacer(1, 12)
            ])
        
        # Add key contributions
        content.append(Paragraph("Key Contributions", styles['Heading2']))
        for contribution in paper_draft.key_contributions:
            content.append(Paragraph(f"• {contribution}", styles['Normal']))
        
        # Build PDF
        doc.build(content)
        print(f"PDF report generated at {output_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

async def main():
    """Main function to orchestrate the academic paper generation process."""
    try:
        domain = "Machine Learning Ethics"
        academic_level = "PhD"
        
        # Run the research workflow
        lit_review = await conduct_literature_review(domain, academic_level)
        methodology = await develop_methodology(domain, academic_level, lit_review)
        analysis = await analyze_data(domain, academic_level, methodology)
        paper_draft = await write_paper(domain, academic_level, analysis)
        
        # Generate PDF report
        generate_pdf_report(paper_draft)
        
    except Exception as e:
        print(f"Error in academic paper generation: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())