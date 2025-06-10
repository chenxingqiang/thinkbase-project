import requests
import json
from agentflow.core.workflow import WorkflowEngine
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig
from agentflow.agents.agent import Agent
from agentflow.agents.agent_types import AgentType
from agentflow.core.config import AgentConfig, ModelConfig
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole
import asyncio
import os
import logging

from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
import io
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize ELL with configuration
ell = ELLIntegration()
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True,
    "model": "deepseek-chat",
    "default_model": "deepseek-chat",
    "temperature": 0.7,
    "mode": "simple",
    "api_key": os.getenv("DEEPSEEK_API_KEY", "sk-3671f0e962bf4e4980750a47c549ccd4"),
    "base_url": "https://api.deepseek.com/v1",
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})
logger.debug("ELL configured")

def format_time(seconds: float) -> str:
    """Format time duration in a readable way."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes = int(seconds / 60)
    seconds = seconds % 60
    return f"{minutes} minutes {seconds:.1f} seconds"

def print_step_result(step: dict, step_number: int, total_steps: int):
    """Print formatted step result."""
    print(f"\nStep {step_number}/{total_steps}: {step.get('name', 'Unknown')}")
    print("-" * 40)
    print(f"ID: {step.get('id', 'Unknown')}")
    print(f"Type: {step.get('type', 'Unknown')}")
    print(f"Status: {step.get('status', 'Unknown')}")
    print(f"Duration: {format_time(step.get('duration', 0))}")
    
    if step.get('error'):
        print("\nError:")
        print(f"  {step['error']}")
    
    if step.get('result'):
        print("\nResult:")
        try:
            # Try to format specific result types
            result = step['result']
            if isinstance(result, dict):
                if 'research_params' in result:
                    print("  Research Parameters:")
                    print(f"    Topic: {result['research_params']['topic']}")
                    print(f"    Academic Level: {result['research_params']['academic_level']}")
                    print(f"    Deadline: {result['research_params']['deadline']}")
                    print("\n    Required Sections:")
                    for section in result['research_params']['required_sections']:
                        print(f"      - {section}")
                
                if 'document_params' in result:
                    print("\n  Document Parameters:")
                    print(f"    Format: {result['document_params']['format']}")
                    print(f"    Include Figures: {result['document_params']['include_figures']}")
                    print(f"    Length: {result['document_params']['min_length']} - {result['document_params']['max_length']} words")
            else:
                print(json.dumps(result, indent=2))
        except Exception as e:
            print(json.dumps(step['result'], indent=2))

class ResearchPaperGenerator(Agent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.generation_progress = {}

    @ell.with_ell(mode="simple")
    async def generate_section(self, section_name: str, prompt: str) -> str:
        """Generate a section of the research paper."""
        try:
            logger.debug(f"Starting generation of {section_name}")
            self.generation_progress[section_name] = "in_progress"
            print(f"\nGenerating {section_name}...")
            
            # Create message for ELL
            system_msg = Message(
                role=MessageRole.SYSTEM,
                content=f"""You are an expert in distributed machine learning systems. Write the {section_name} section of a research paper. 
                Follow these specific guidelines:
                1. Provide detailed, technical content with specific examples and references.
                2. Focus on implementation details, optimization techniques, and real-world applications.
                3. Include relevant citations from recent (2023-2024) academic papers.
                4. Maintain a formal academic writing style.
                5. Ensure content is thorough and well-structured with clear subsections.
                6. Include specific metrics, methodologies, and results where appropriate.
                7. Discuss practical implications and future research directions.
                8. Minimum length should be 500 words with comprehensive coverage.
                
                Format the content with appropriate academic structure and terminology."""
            )
            user_msg = Message(
                role=MessageRole.USER,
                content=prompt,
                metadata={
                    "section": section_name,
                    "type": "research_paper"
                }
            )
            logger.debug(f"Created messages for {section_name}")
            
            # Process messages through ELL
            logger.debug(f"Processing message for {section_name}")
            try:
                # Process messages one at a time
                response = await ell.process_message(system_msg)
                response = await ell.process_message(user_msg)
                logger.debug(f"Received response for {section_name}: {response}")
                
                if not response:
                    raise ValueError(f"No response received from ELL for {section_name}")
                
                if not hasattr(response, 'content') or not response.content:
                    raise ValueError(f"Response has no content for {section_name}")
                
                content = str(response.content)
                if not content.strip():
                    raise ValueError(f"Response content is empty for {section_name}")
                
                # Add error handling for response validation
                if len(content.strip()) < 100:  # Basic validation for minimum content length
                    raise ValueError(f"Response content is too short for {section_name} (less than 100 characters)")
                
                self.generation_progress[section_name] = "completed"
                logger.debug(f"Successfully generated {section_name}")
                print(f"✓ {section_name} completed")
                return content.strip()
                
            except Exception as e:
                logger.error(f"Error processing message for {section_name}: {str(e)}")
                raise
            
        except Exception as e:
            self.generation_progress[section_name] = "failed"
            error_msg = f"Error generating {section_name}: {str(e)}"
            logger.error(f"Error in generate_section: {error_msg}")
            print(f"✗ {error_msg}")
            return error_msg

    async def generate_paper_content(self) -> Dict[str, str]:
        """Generate all sections of the research paper asynchronously."""
        print("\nStarting paper content generation...")
        print("=" * 50)
        
        sections = {
            "abstract": "Write an abstract for a research paper titled 'Advanced Machine Learning Techniques in Distributed Systems'. Focus on implementation, optimization, and real-world applications.",
            "introduction": "Write the introduction section for a research paper on advanced machine learning techniques in distributed systems.",
            "literature_review": "Write the literature review section focusing on recent developments in distributed machine learning systems.",
            "methodology": "Write the methodology section for research on distributed machine learning systems, including approaches to model parallelism and data distribution.",
            "results": "Write the results and analysis section for research on distributed machine learning systems, including performance metrics and comparative analysis.",
            "discussion": "Write the discussion section analyzing the implications of the findings for distributed machine learning systems.",
            "conclusion": "Write the conclusion section for the research on distributed machine learning systems.",
            "references": "Generate 5-7 recent academic references (2023-2024) related to distributed machine learning systems research."
        }

        content = {}
        tasks = []
        
        # Create tasks for all sections
        for section_name, prompt in sections.items():
            task = asyncio.create_task(self.generate_section(section_name, prompt))
            tasks.append((section_name, task))
        
        # Wait for all tasks to complete
        for section_name, task in tasks:
            try:
                content[section_name] = await task
            except Exception as e:
                print(f"Failed to generate {section_name}: {str(e)}")
                content[section_name] = f"Error generating {section_name}: {str(e)}"
        
        # Print generation summary
        print("\nContent Generation Summary:")
        print("-" * 30)
        for section, status in self.generation_progress.items():
            status_symbol = "✓" if status == "completed" else "✗"
            print(f"{status_symbol} {section}: {status}")
        
        return content

    def create_methodology_diagram(self) -> bytes:
        """Create methodology diagram for the paper."""
        print("\nCreating methodology diagram...")
        plt.figure(figsize=(8, 6))
        
        # Create a simple flow diagram
        stages = ['Data\nCollection', 'Model\nDesign', 'Distributed\nTraining', 'Performance\nEvaluation']
        x = np.arange(len(stages))
        y = np.zeros_like(x)
        
        plt.plot(x, y, 'bo-', linewidth=2, markersize=20)
        
        for i, stage in enumerate(stages):
            plt.annotate(stage, (x[i], y[i]), xytext=(0, 20),
                        textcoords='offset points', ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5))
            
        plt.axis('off')
        plt.title('Research Methodology Flow')
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print("✓ Methodology diagram created")
        return img_buffer.getvalue()

    def create_research_progress_plot(self) -> bytes:
        """Create research progress visualization."""
        print("\nCreating research progress plot...")
        metrics = {
            'Model Accuracy': 0.92,
            'Training Speed': 0.85,
            'Resource Efficiency': 0.78,
            'Scalability': 0.88
        }
        
        plt.figure(figsize=(10, 4))
        bars = plt.bar(list(metrics.keys()), list(metrics.values()), color='#2E86C1')
        plt.ylim(0, 1.0)
        plt.title('Research Performance Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print("✓ Research progress plot created")
        return img_buffer.getvalue()

    async def generate_pdf_report(self, results: dict, output_path: str) -> str:
        """Generate a complete research paper PDF."""
        print("\nGenerating PDF report...")
        print("=" * 50)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print("\nGenerating content...")
        content = await self.generate_paper_content()
        
        print("\nCreating PDF document...")
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title Page
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1B4F72'),
            alignment=1
        )
        story.append(Paragraph("Advanced Machine Learning Techniques in Distributed Systems", title_style))
        story.append(Spacer(1, 40))
        
        # Author and Affiliation
        author_style = ParagraphStyle(
            'Author',
            parent=styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#2C3E50'),
            alignment=1
        )
        story.append(Paragraph("Research Team", author_style))
        story.append(Paragraph("Department of Computer Science", author_style))
        story.append(Paragraph("University Research Institute", author_style))
        story.append(Spacer(1, 40))
        
        # Date
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#666666'),
            alignment=1
        )
        story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), date_style))
        story.append(Spacer(1, 12))
        
        # Page break after title page
        story.append(Paragraph("<br clear=all style='page-break-before:always'/>", styles['Normal']))
        
        # Style definitions
        h1_style = ParagraphStyle(
            'Heading1',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=16,
            textColor=colors.HexColor('#2E86C1'),
            borderWidth=0,
            borderPadding=10,
            keepWithNext=True
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            leading=16,
            alignment=4  # Justified alignment
        )
        
        print("\nAdding content sections...")
        sections = [
            ("Abstract", content["abstract"]),
            ("1. Introduction", content["introduction"]),
            ("2. Literature Review", content["literature_review"]),
            ("3. Methodology", content["methodology"]),
            ("4. Results and Analysis", content["results"]),
            ("5. Discussion", content["discussion"]),
            ("6. Conclusion", content["conclusion"]),
            ("References", content["references"])
        ]
        
        for title, text in sections:
            print(f"Adding section: {title}")
            story.append(Paragraph(title, h1_style))
            story.append(Spacer(1, 12))
            
            # Add methodology diagram
            if title == "3. Methodology":
                methodology_text = """Our research methodology follows a systematic approach that integrates theoretical analysis 
                with practical implementation. The methodology consists of several interconnected components, as illustrated in 
                the following diagram:"""
                story.append(Paragraph(methodology_text, normal_style))
                story.append(Spacer(1, 12))
                
                img_data = self.create_methodology_diagram()
                img = Image(io.BytesIO(img_data), width=6*inch, height=4.5*inch)
                story.append(img)
                story.append(Spacer(1, 20))
            
            # Add results visualization
            if title == "4. Results and Analysis":
                results_text = """Our experimental results demonstrate the effectiveness of the proposed approaches. 
                The following visualizations illustrate key findings from our research:"""
                story.append(Paragraph(results_text, normal_style))
                story.append(Spacer(1, 12))
                
                img_data = self.create_research_progress_plot()
                img = Image(io.BytesIO(img_data), width=7.5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 20))
            
            # Add section text
            paragraphs = text.split('\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    story.append(Paragraph(paragraph.strip(), normal_style))
                    story.append(Spacer(1, 12))
        
        print("\nBuilding PDF...")
        doc.build(story)
        print(f"✓ PDF report generated: {output_path}")
        return output_path

async def main():
    """Run the example."""
    try:
        # Create workflow engine
        engine = WorkflowEngine()
        
        # Create workflow steps
        steps = [
            WorkflowStep(
                id="research_planning",
                name="Research Planning",
                type=WorkflowStepType.TRANSFORM,
                description="Plan the research approach",
                config=StepConfig(
                    strategy="standard",
                    params={
                        "topic": "Advanced Machine Learning Techniques in Distributed Systems",
                        "deadline": "2024-06-30",
                        "academic_level": "PhD",
                        "required_sections": [
                            "Introduction",
                            "Literature Review",
                            "Methodology",
                            "Results",
                            "Discussion",
                            "Conclusion"
                        ]
                    }
                )
            ),
            WorkflowStep(
                id="document_generation",
                name="Document Generation",
                type=WorkflowStepType.TRANSFORM,
                description="Generate research document",
                dependencies=["research_planning"],
                config=StepConfig(
                    strategy="standard",
                    params={
                        "format": "pdf",
                        "include_figures": True,
                        "min_length": 2000,
                        "max_length": 5000
                    }
                )
            )
        ]
        
        # Create workflow configuration
        workflow_config = WorkflowConfig(
            name="research_workflow",
            description="A workflow for academic research",
            max_iterations=3,
            timeout=3600,
            steps=steps
        )
        
        # Print workflow configuration
        print("\nInitializing Research Workflow...")
        print("=" * 50)
        print("\nWorkflow Configuration:")
        print(f"Name: {workflow_config.name}")
        print(f"Description: {workflow_config.model_dump().get('description', 'No description')}")
        print(f"Max Iterations: {workflow_config.max_iterations}")
        timeout = workflow_config.timeout
        print(f"Timeout: {format_time(float(timeout)) if timeout is not None else 'No timeout'}")
        print(f"Total Steps: {len(workflow_config.steps)}")
        
        # Initialize workflow engine
        print("\nInitializing workflow engine...")
        
        # Execute workflow
        print("\nStarting workflow execution...")
        print("=" * 50)
        
        # Create agent config with Deepseek model
        agent_config = AgentConfig(
            name="research_agent",
            type=AgentType.RESEARCH,
            model=ModelConfig(name="deepseek-chat", provider="deepseek"),
            workflow=workflow_config.model_dump()
        )
        
        # Create ResearchPaperGenerator instance
        agent = ResearchPaperGenerator(agent_config)
        
        context = {
            "research_params": {
                "topic": "Advanced Machine Learning Techniques in Distributed Systems",
                "deadline": "2024-06-30",
                "academic_level": "PhD",
                "required_sections": [
                    "Introduction",
                    "Literature Review",
                    "Methodology",
                    "Results",
                    "Discussion",
                    "Conclusion"
                ]
            },
            "document_params": {
                "format": "pdf",
                "include_figures": True,
                "min_length": 2000,
                "max_length": 5000
            }
        }
        
        # Register workflow and get workflow_id
        workflow_id = await engine.register_workflow(agent)
        result = await engine.execute_workflow(workflow_id, context)
        
        # Generate PDF report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.abspath(f"output/reports/research_workflow_report_{timestamp}.pdf")
        pdf_path = await agent.generate_pdf_report(result, report_path)
        print(f"\nGenerated PDF report: {pdf_path}")
        
        # Print results
        print("\nWorkflow Execution Results:")
        print("=" * 50)
        print(f"\nTotal Duration: {format_time(result.get('duration', 0))}")
        print(f"Steps Completed: {len(result.get('steps', []))}")
        print(f"Status: {result.get('status', 'unknown')}")
        
        # Cleanup
        print("\nWorkflow cleanup completed.")
        
    except Exception as e:
        print(f"Error executing workflow: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
