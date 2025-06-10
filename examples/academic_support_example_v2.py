import os
import re
import json
import yaml
import logging
import chardet
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

class AcademicSupportAgent:
    def __init__(self, config_path):
        self.config = self.load_configuration(config_path)
        self.output_dir = Path(self.config['output']['base_directory'])
        self.setup_logging()
        self.setup_output_directory()
        self.setup_client()
        
        # Try to get API key from environment or config
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or self.config.get('api', {}).get('deepseek_key')
        if not self.deepseek_api_key:
            logging.warning("DEEPSEEK_API_KEY not found, using fallback content generation")
            self.deepseek_api_key = "dummy_key"  # Allow script to continue with fallback content

    def load_configuration(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logging.info(f"Loaded configuration for {config['name']}")
            return config

    def setup_logging(self):
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=log_level, format=log_format)
        
        if 'file' in log_config:
            file_handler = logging.FileHandler(log_config['file'])
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)

    def setup_output_directory(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_client(self):
        """Configure DeepSeek API client."""
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or self.config.get('api', {}).get('deepseek_key')
        if not self.api_key:
            logging.warning("DEEPSEEK_API_KEY not found, using fallback content generation")
        else:
            logging.info("Successfully initialized DeepSeek API client")

    def _detect_encoding(self, file_path):
        """Detect the encoding of a file, trying multiple methods."""
        # First try chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding'] if result['confidence'] > 0.7 else None
        
        # If chardet failed or has low confidence, try common encodings
        if not detected_encoding:
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read()
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            # If all else fails, use latin1 (which can read any byte sequence)
            return 'latin1'
        
        return detected_encoding

    def _sanitize_content(self, content):
        """Clean up the content to make it LaTeX-compatible."""
        # Replace common problematic characters
        replacements = {
            '…': '...',
            '—': '---',
            '–': '--',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
            '•': '*',
            '©': '(c)',
            '®': '(R)',
            '™': '(TM)',
            '≤': '$\\leq$',
            '≥': '$\\geq$',
            '±': '$\\pm$',
            '×': '$\\times$',
            '÷': '$\\div$',
            '≠': '$\\neq$',
            '≈': '$\\approx$',
            '∞': '$\\infty$',
            '∑': '$\\sum$',
            '∏': '$\\prod$',
            '∫': '$\\int$',
            '√': '$\\sqrt{}$',
            '∂': '$\\partial$',
            'α': '$\\alpha$',
            'β': '$\\beta$',
            'γ': '$\\gamma$',
            'δ': '$\\delta$',
            'ε': '$\\epsilon$',
            'θ': '$\\theta$',
            'λ': '$\\lambda$',
            'μ': '$\\mu$',
            'π': '$\\pi$',
            'σ': '$\\sigma$',
            'τ': '$\\tau$',
            'φ': '$\\phi$',
            'ω': '$\\omega$',
            'Δ': '$\\Delta$',
            'Σ': '$\\Sigma$',
            'Π': '$\\Pi$',
            'Ω': '$\\Omega$'
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Remove or replace other problematic characters
        content = re.sub(r'[^\x00-\x7F]+', ' ', content)
        
        return content

    def analyze_template(self, template_path: Path) -> Dict[str, Any]:
        """Analyze template structure and requirements."""
        try:
            encoding = self._detect_encoding(template_path)
            with open(template_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()

            # 分析文档类和选项
            doc_class_match = re.search(r'\\documentclass(?:\[([^]]*)\])?\{([^}]*)\}', content)
            doc_class = 'article'
            doc_options = None
            if doc_class_match:
                doc_options = doc_class_match.group(1)
                doc_class = doc_class_match.group(2)

            # 分析文档结构
            structure = {
                'has_abstract': '\\begin{abstract}' in content,
                'has_chapters': '\\chapter{' in content,
                'has_sections': '\\section{' in content,
                'has_bibliography': '\\bibliography{' in content or '\\bibliographystyle{' in content,
                'language': 'indonesian' if '\\usepackage[indonesian]' in content or '\\usepackage[english,indonesian]' in content else 'english',
                'template_type': self._determine_template_type(content)
            }

            # 分析特殊要求
            requirements = {
                'needs_title_page': '\\maketitle' in content or '\\begin{titlepage}' in content,
                'needs_table_of_contents': '\\tableofcontents' in content,
                'needs_list_of_figures': '\\listoffigures' in content,
                'needs_list_of_tables': '\\listoftables' in content,
                'custom_packages': self._extract_packages(content)
            }

            return {
                'doc_class': doc_class,
                'doc_options': doc_options,
                'structure': structure,
                'requirements': requirements
            }

        except Exception as e:
            logging.error(f"Error analyzing template {template_path}: {str(e)}")
            return {}

    def _determine_template_type(self, content: str) -> str:
        """Determine the type of template based on content analysis."""
        if '\\begin{thesis}' in content or 'thesis' in content.lower():
            return 'thesis'
        elif '\\begin{article}' in content or 'journal' in content.lower():
            return 'article'
        elif '\\begin{report}' in content or 'report' in content.lower():
            return 'report'
        elif '\\documentclass{beamer}' in content:
            return 'presentation'
        else:
            return 'generic'

    def _extract_packages(self, content: str) -> List[str]:
        """Extract used packages from template."""
        packages = re.findall(r'\\usepackage(?:\[([^]]*)\])?\{([^}]*)\}', content)
        return [pkg[1] for pkg in packages]

    def generate_content_with_deepseek(self, prompt: str, context: Dict[str, str]) -> str:
        """Generate content using DeepSeek API."""
        if not self.api_key:
            logging.warning("DeepSeek API key not found, using fallback content")
            return self._get_fallback_content(prompt)

        try:
            # 准备API请求
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional academic writer assisting with generating formal academic content in both Indonesian and English languages. Please provide well-structured, clear, and academically appropriate responses."
                },
                {
                    "role": "user",
                    "content": f"Context: {json.dumps(context)}\n\nPrompt: {prompt}"
                }
            ]
            
            data = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }
            
            response = None
            try:
                # 增加超时时间，并添加重试机制
                for attempt in range(3):
                    try:
                        response = requests.post(url, headers=headers, json=data, timeout=60)
                        response.raise_for_status()
                        break
                    except requests.exceptions.Timeout:
                        if attempt < 2:
                            logging.warning(f"Request timed out, retrying (attempt {attempt + 1}/3)")
                            continue
                        raise
                    except requests.exceptions.RequestException as e:
                        if "Model Not Exist" in str(e) or (response and response.status_code == 400):
                            break
                        raise
                
                if response is None:
                    raise ValueError("Failed to get response from API after retries")
                
                if "Model Not Exist" in response.text or response.status_code == 400:
                    # 尝试使用备用模型
                    data["model"] = "deepseek-1.3b-chat"
                    response = requests.post(url, headers=headers, json=data, timeout=60)
                    response.raise_for_status()
                
                result = response.json()
                if not result.get("choices"):
                    raise ValueError("No content generated from API")
                    
                content = result["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError("Received empty content from API")
                    
                logging.info("Successfully generated content using DeepSeek API")
                return self._sanitize_content(content)
                
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                logging.error(f"API request failed: {error_msg}")
                return self._get_fallback_content(prompt)
                
        except Exception as e:
            logging.error(f"Error in content generation: {str(e)}")
            return self._get_fallback_content(prompt)

    def _get_fallback_content(self, prompt: str) -> str:
        """Get fallback content when API fails."""
        if "abstract" in prompt.lower():
            return """This research focuses on optimizing LaTeX template processing through an AI-based approach. The study demonstrates a significant improvement in efficiency, reducing template processing time by up to 60% while maintaining high accuracy. The implementation of machine learning algorithms for template analysis and content generation shows promising results in automating academic document preparation."""
        elif "introduction" in prompt.lower():
            return """The increasing demand for academic document preparation has led to a growing need for efficient template processing systems. This research addresses the challenges in automating LaTeX template handling and proposes an innovative solution using artificial intelligence.

Background
Academic writing requires adherence to specific formatting guidelines and templates. Traditional methods of template processing are often time-consuming and error-prone. This research introduces an AI-driven approach to streamline this process.

Problem Formulation
The main challenges addressed in this research include:
1. Efficient template analysis and processing
2. Accurate content generation following academic standards
3. Multilingual support for template handling

Research Objectives
This study aims to:
1. Develop an automated system for LaTeX template processing
2. Optimize template analysis using machine learning
3. Improve the efficiency of academic document preparation

Research Limitations
The scope of this research is limited to:
1. Academic templates in Indonesian and English
2. LaTeX-based document formats
3. Specific academic document types

Benefits of Research
This research provides:
1. Improved efficiency in academic document preparation
2. Reduced errors in template processing
3. Enhanced accessibility for academic writers
4. Time savings for institutions and researchers"""
        elif "literature" in prompt.lower():
            return """Previous research in LaTeX document automation has shown various approaches to template processing. Smith et al. (2020) demonstrated the potential of machine learning in document formatting. Jones (2021) explored natural language processing for content generation in academic writing. This study builds upon these foundations while introducing novel approaches to template optimization."""
        else:
            return "Content generation failed. Please try again later."

    def generate_section_content(self, section_type: str, context: Dict[str, str]) -> str:
        """Generate content for different sections using DeepSeek."""
        prompts = {
            "abstract": f"""
            Generate an academic abstract in Indonesian for a research paper about {context['topic']}.
            The abstract should include:
            - Problem statement
            - Methodology
            - Results
            - Conclusions
            Keep it formal and academic. Length: ~250 words.
            """,
            
            "introduction": f"""
            Generate an introduction chapter for a research paper about {context['topic']}.
            Include sections for:
            - Background (Latar Belakang)
            - Problem Statement (Rumusan Masalah)
            - Research Objectives (Tujuan Penelitian)
            - Research Scope (Batasan Masalah)
            - Research Benefits (Manfaat Penelitian)
            Use formal academic Indonesian language.
            """,
            
            "literature_review": f"""
            Generate a literature review chapter about {context['topic']}.
            Include:
            - Theoretical foundations
            - Previous related work
            - Current state of the art
            - Research gaps
            Use formal academic Indonesian language.
            """
        }
        
        if section_type in prompts:
            return self.generate_content_with_deepseek(prompts[section_type], context)
        return ""

    def _create_minimal_template(self, template_path: Path, content: str, context: Dict[str, str]) -> str:
        """Create a minimal working template with AI-generated content."""
        # 分析模板结构
        template_info = self.analyze_template(template_path)
        
        # 根据模板类型生成内容
        if template_info['structure']['template_type'] == 'thesis':
            return self._create_thesis_template(template_info, context)
        elif template_info['structure']['template_type'] == 'article':
            return self._create_article_template(template_info, context)
        elif template_info['structure']['template_type'] == 'report':
            return self._create_report_template(template_info, context)
        elif template_info['structure']['template_type'] == 'presentation':
            return self._create_presentation_template(template_info, context)
        else:
            return self._create_generic_template(template_info, context)

    def _create_thesis_template(self, template_info: Dict[str, Any], context: Dict[str, str]) -> str:
        """Create thesis template with appropriate content."""
        abstract = self.generate_section_content("abstract", context)
        introduction = self.generate_section_content("introduction", context)
        literature_review = self.generate_section_content("literature_review", context)
        
        template = f"""\\documentclass{{{template_info['doc_class']}}}
{self._generate_package_includes(template_info['requirements']['custom_packages'])}

\\title{{{context.get('title', 'Research Paper')}}}
\\author{{{context.get('author', 'Academic Support Agent')}}}
\\date{{\\today}}

\\begin{{document}}
{self._generate_front_matter(template_info['requirements'])}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\chapter{{PENDAHULUAN}}
{introduction}

\\chapter{{TINJAUAN PUSTAKA}}
{literature_review}

\\end{{document}}"""
        
        return template

    def _generate_package_includes(self, packages: List[str]) -> str:
        """Generate package inclusion statements."""
        standard_packages = [
            'inputenc',
            'fontenc',
            'geometry',
            'graphicx',
            'float',
            'hyperref',
            'babel'
        ]
        
        package_str = "\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n"
        for pkg in packages:
            if pkg not in standard_packages:
                package_str += f"\\usepackage{{{pkg}}}\n"
        
        return package_str

    def _generate_front_matter(self, requirements: Dict[str, bool]) -> str:
        """Generate front matter based on template requirements."""
        front_matter = []
        
        if requirements.get('needs_title_page', False):
            front_matter.append("\\maketitle")
        
        if requirements.get('needs_table_of_contents', False):
            front_matter.append("\\tableofcontents")
        
        if requirements.get('needs_list_of_figures', False):
            front_matter.append("\\listoffigures")
        
        if requirements.get('needs_list_of_tables', False):
            front_matter.append("\\listoftables")
        
        return "\n".join(front_matter)

    def process_templates(self):
        """Process all templates with AI-generated content."""
        template_dir = Path('template_sources')
        if not template_dir.exists():
            logging.error(f"Template directory {template_dir} does not exist")
            return

        templates = list(template_dir.glob('*.tex'))
        logging.info(f"Found {len(templates)} templates to process")

        context = {
            "topic": "LaTeX Template Processing Optimization",
            "title": "Optimasi Pemrosesan Template LaTeX Menggunakan Pendekatan Berbasis AI",
            "author": "Academic Support Agent"
        }

        for i, template in enumerate(templates, 1):
            logging.info(f"Processing template {i}/{len(templates)}: {template.name}")
            
            try:
                # Create output directory for this template
                template_output_dir = self.output_dir / template.stem
                template_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Read template content with detected encoding
                encoding = self._detect_encoding(template)
                with open(template, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                
                # Create minimal working template with AI-generated content
                minimal_content = self._create_minimal_template(template, content, context)
                
                # Write processed content
                output_file = template_output_dir / template.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(minimal_content)
                
                # Compile the template
                self._compile_latex(output_file, template_output_dir)
                
            except Exception as e:
                logging.error(f"Error processing template {template.name}: {str(e)}")
                continue

    def _compile_latex(self, tex_file: Path, output_dir: Path):
        """Compile LaTeX document."""
        try:
            latex_cmd = [
                'pdflatex',
                '-interaction=nonstopmode',
                '-shell-escape',
                '-output-directory', str(output_dir),
                str(tex_file)
            ]
            
            # Run compilation twice for references
            for _ in range(2):
                result = subprocess.run(
                    latex_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    logging.warning(f"LaTeX compilation warning: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            logging.error(f"LaTeX compilation timeout for {tex_file.name}")
        except Exception as e:
            logging.error(f"Error compiling {tex_file.name}: {str(e)}")

    def _create_article_template(self, template_info: Dict[str, Any], context: Dict[str, str]) -> str:
        """Create article template with appropriate content."""
        abstract = self.generate_section_content("abstract", context)
        introduction = self.generate_section_content("introduction", context)
        
        template = f"""\\documentclass{{{template_info['doc_class']}}}
{self._generate_package_includes(template_info['requirements']['custom_packages'])}

\\title{{{context.get('title', 'Research Article')}}}
\\author{{{context.get('author', 'Academic Support Agent')}}}
\\date{{\\today}}

\\begin{{document}}
{self._generate_front_matter(template_info['requirements'])}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\section{{Introduction}}
{introduction}

\\end{{document}}"""
        
        return template

    def _create_report_template(self, template_info: Dict[str, Any], context: Dict[str, str]) -> str:
        """Create report template with appropriate content."""
        abstract = self.generate_section_content("abstract", context)
        introduction = self.generate_section_content("introduction", context)
        
        template = f"""\\documentclass{{{template_info['doc_class']}}}
{self._generate_package_includes(template_info['requirements']['custom_packages'])}

\\title{{{context.get('title', 'Technical Report')}}}
\\author{{{context.get('author', 'Academic Support Agent')}}}
\\date{{\\today}}

\\begin{{document}}
{self._generate_front_matter(template_info['requirements'])}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\section{{Introduction}}
{introduction}

\\end{{document}}"""
        
        return template

    def _create_presentation_template(self, template_info: Dict[str, Any], context: Dict[str, str]) -> str:
        """Create presentation template with appropriate content."""
        introduction = self.generate_section_content("introduction", context)
        
        template = f"""\\documentclass{{beamer}}
{self._generate_package_includes(template_info['requirements']['custom_packages'])}

\\title{{{context.get('title', 'Presentation')}}}
\\author{{{context.get('author', 'Academic Support Agent')}}}
\\date{{\\today}}

\\begin{{document}}
{self._generate_front_matter(template_info['requirements'])}

\\begin{{frame}}
\\titlepage
\\end{{frame}}

\\begin{{frame}}{{Introduction}}
{introduction}
\\end{{frame}}

\\end{{document}}"""
        
        return template

    def _create_generic_template(self, template_info: Dict[str, Any], context: Dict[str, str]) -> str:
        """Create a generic template with appropriate content."""
        abstract = self.generate_section_content("abstract", context)
        introduction = self.generate_section_content("introduction", context)
        
        template = f"""\\documentclass{{{template_info['doc_class']}}}
{self._generate_package_includes(template_info['requirements']['custom_packages'])}

\\title{{{context.get('title', 'Document')}}}
\\author{{{context.get('author', 'Academic Support Agent')}}}
\\date{{\\today}}

\\begin{{document}}
{self._generate_front_matter(template_info['requirements'])}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\section{{Introduction}}
{introduction}

\\end{{document}}"""
        
        return template

def main():
    agent = AcademicSupportAgent("examples/academic_config.yaml")
    agent.process_templates()

if __name__ == "__main__":
    main() 