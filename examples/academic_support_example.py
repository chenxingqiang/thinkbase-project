"""
Example script demonstrating how to use the Academic Research Support Agent
This example shows how to:
1. Initialize the agent with the YAML config
2. Run a research workflow
3. Generate output documents
"""

import os
import yaml
from typing import Dict, Any, List, Optional
import asyncio
from openai import OpenAI
import json
from jinja2 import Environment, FileSystemLoader
import subprocess
import shutil
from pathlib import Path
import logging
import re
import chardet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AcademicSupportAgent:
    def __init__(self, config_path: str):
        """Initialize the academic support agent with YAML configuration."""
        self.config_path = config_path
        self.config = self.load_configuration(config_path)
        self.setup_client()
        self.output_dir = Path(self.config['output']['base_directory'])
        self.template_dir = Path('template_sources')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging based on config
        if self.config['logging']['output']['file']:
            log_file = Path(self.config['logging']['output']['file'])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(self.config['logging']['format']))
            logging.getLogger().addHandler(file_handler)
        
    def load_configuration(self, config_path: str):
        """Load the YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded configuration for {config['name']}")
            return config
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    def setup_client(self):
        """Configure OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"
        )

    async def process_all_templates(self) -> List[Dict[str, Any]]:
        """Process all templates in the template_sources directory."""
        results = []
        templates = list(self.template_dir.glob('*.tex'))
        total_templates = len(templates)
        
        logging.info(f"Found {total_templates} templates to process")
        
        for idx, template_path in enumerate(templates, 1):
            logging.info(f"Processing template {idx}/{total_templates}: {template_path.name}")
            try:
                result = await self.process_template(template_path)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing template {template_path.name}: {e}")
                results.append({
                    'template': template_path.name,
                    'status': 'error',
                    'error': str(e)
                })
                
        return results

    async def process_template(self, template_path: Path) -> Dict[str, Any]:
        """Process a single template file."""
        try:
            # Create output directory for this template
            template_output_dir = self.output_dir / template_path.stem
            template_output_dir.mkdir(exist_ok=True)
            
            # Create necessary subdirectories
            for subdir in ['frontmatter', 'mainmatter', 'figures', 'tables']:
                (template_output_dir / subdir).mkdir(exist_ok=True)
            
            # Copy template to output directory
            output_tex_path = template_output_dir / template_path.name
            
            # Process the template content
            processed_content = self._process_template_content(template_path)
            
            # Write processed content
            with open(output_tex_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            # Generate PDF
            pdf_path = await self._generate_pdf(output_tex_path)
            
            # Clean up auxiliary files if configured
            if self.config['output']['cleanup']:
                self._cleanup_auxiliary_files(output_tex_path)
            
            return {
                'template': template_path.name,
                'status': 'success',
                'output_dir': str(template_output_dir),
                'pdf_path': str(pdf_path) if pdf_path else None
            }
            
        except Exception as e:
            logging.error(f"Error processing {template_path.name}: {e}")
            raise

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect the encoding of a file."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'

    def _process_template_content(self, template_path: Path) -> str:
        """Process template content to make it compilable."""
        try:
            # Detect file encoding
            encoding = self._detect_encoding(template_path)
            with open(template_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Create a minimal document class if using custom class
            document_class = re.search(r'\\documentclass(?:\[.*?\])?\{([^}]+)\}', content)
            if document_class:
                class_name = document_class.group(1)
                if not any(class_name in std_class for std_class in ['article', 'report', 'book', 'beamer']):
                    content = content.replace(f'\\documentclass{{{class_name}}}', '\\documentclass{article}')
            
            # Add common packages if not present
            common_packages = [
                '\\usepackage[utf8]{inputenc}',
                '\\usepackage[T1]{fontenc}',
                '\\usepackage{amsmath}',
                '\\usepackage{amssymb}',
                '\\usepackage{graphicx}',
                '\\usepackage{hyperref}',
                '\\usepackage{geometry}',
                '\\usepackage{xeCJK}',
                '\\usepackage{fontspec}'
            ]
            
            # Find document begin
            doc_begin = content.find('\\begin{document}')
            if doc_begin == -1:
                # If no document environment found, create one
                content = '\\documentclass{article}\n' + '\n'.join(common_packages) + '\n\\begin{document}\n' + content + '\n\\end{document}'
            else:
                # Insert packages before \begin{document}
                preamble = content[:doc_begin]
                for package in common_packages:
                    if package not in preamble:
                        content = content[:doc_begin] + package + '\n' + content[doc_begin:]
            
            # Create empty files for \input commands
            input_files = re.findall(r'\\input\{([^}]+)\}', content)
            for input_file in input_files:
                content = content.replace(f'\\input{{{input_file}}}', '')
            
            # Add minimal content for bibliography
            if '\\bibliography{' in content:
                content = content.replace('\\bibliography{references}', '')
            
            # Remove problematic commands
            content = re.sub(r'\\printindex.*', '', content)
            
            return content
            
        except Exception as e:
            logging.error(f"Error processing template content: {e}")
            raise

    async def _generate_pdf(self, tex_path: Path) -> Optional[Path]:
        """Generate PDF from LaTeX file using appropriate engine."""
        try:
            # Determine which LaTeX engine to use based on content
            engine = self._determine_latex_engine(tex_path)
            
            # Get compile options from config
            compile_options = self.config['latex']['compile_options']
            
            # Prepare the command
            cmd = [
                engine,
                '-interaction=nonstopmode',
                '-shell-escape',
                str(tex_path)
            ]
            
            # Run LaTeX compilation with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config['latex']['timeout_seconds']
                )
                
                if process.returncode != 0:
                    logging.warning(f"LaTeX compilation warning for {tex_path.name}. Return code: {process.returncode}")
                    logging.debug(f"Compilation output: {stdout.decode(errors='replace')}")
                    logging.debug(f"Compilation errors: {stderr.decode(errors='replace')}")
                
                # Check if PDF was generated
                pdf_path = tex_path.with_suffix('.pdf')
                if pdf_path.exists():
                    logging.info(f"Successfully generated PDF: {pdf_path}")
                    return pdf_path
                else:
                    logging.error(f"PDF generation failed for {tex_path}")
                    return None
                    
            except asyncio.TimeoutError:
                logging.error(f"LaTeX compilation timed out for {tex_path}")
                process.kill()
                return None
                
        except Exception as e:
            logging.error(f"Error generating PDF for {tex_path}: {e}")
            return None

    def _determine_latex_engine(self, tex_path: Path) -> str:
        """Determine which LaTeX engine to use based on file content."""
        try:
            encoding = self._detect_encoding(tex_path)
            with open(tex_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
                
            # Check for indicators of which engine to use
            if '\\usepackage{xeCJK}' in content or '\\usepackage{fontspec}' in content:
                return 'xelatex'
            elif '\\usepackage{luatexja}' in content:
                return 'lualatex'
            else:
                return 'pdflatex'
                
        except Exception as e:
            logging.warning(f"Error reading template content, defaulting to pdflatex: {e}")
            return 'pdflatex'

    def _cleanup_auxiliary_files(self, tex_path: Path):
        """Clean up auxiliary files after PDF generation."""
        extensions = ['.aux', '.log', '.out', '.toc', '.lof', '.lot', '.bbl', '.blg']
        for ext in extensions:
            aux_file = tex_path.with_suffix(ext)
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except Exception as e:
                    logging.warning(f"Could not delete {aux_file}: {e}")

async def main():
    """Main function to process all templates."""
    # Path to the YAML config
    config_path = "examples/academic_config.yaml"
    
    # Initialize agent
    agent = AcademicSupportAgent(config_path)
    
    # Process all templates
    logging.info("Starting template processing...")
    results = await agent.process_all_templates()
    
    # Generate summary report
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    logging.info("\nProcessing Summary:")
    logging.info(f"Total templates processed: {len(results)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    
    # Save detailed results
    results_path = Path('output/processing_results.json')
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    asyncio.run(main()) 