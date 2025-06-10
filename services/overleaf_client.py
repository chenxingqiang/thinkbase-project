"""
Overleaf client for fetching templates.
"""

from typing import Dict, List, Any, Optional, Union, cast
import aiohttp
import asyncio
import logging
import os
import json
from pathlib import Path
from bs4 import BeautifulSoup, Tag, NavigableString
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OverleafClient:
    """Client for interacting with Overleaf."""
    
    def __init__(self):
        """Initialize the Overleaf client."""
        self.data_dir = Path('/Users/xingqiangchen/.supabase/supabase-overleaf/data')
        self.general_dir = self.data_dir / 'general'
        self.detail_dir = self.data_dir / 'detail'
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = "https://cn.overleaf.com"
        self.api_url = f"{self.base_url}/api/v2"
        self.email = os.getenv("OVERLEAF_EMAIL")
        self.password = os.getenv("OVERLEAF_PASSWORD")
        
        if not self.email or not self.password:
            raise ValueError("OVERLEAF_EMAIL and OVERLEAF_PASSWORD must be set in environment variables")
            
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
    
    async def login(self) -> bool:
        """Login to Overleaf."""
        if not self.session:
            logging.error("Session not initialized")
            return False
            
        try:
            # First get the login page to get the CSRF token
            logging.info("Fetching login page to get CSRF token...")
            async with self.session.get(
                f"{self.base_url}/login",
                allow_redirects=True,
                headers={
                    **self.headers,
                    'Referer': f"{self.base_url}/login"
                }
            ) as response:
                if response.status != 200:
                    logging.error(f"Failed to get login page: {response.status}")
                    return False
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Get CSRF token from meta tag
                csrf_meta = soup.find('meta', {'name': 'ol-csrfToken'})
                csrf_value = None
                
                if csrf_meta and isinstance(csrf_meta, Tag):
                    csrf_value = csrf_meta.get('content', '')
                    logging.debug("Found CSRF token in meta tag")
                
                # If not found in meta, try form input
                if not csrf_value:
                    csrf_input = soup.find('input', {'name': '_csrf'})
                    if csrf_input and isinstance(csrf_input, Tag):
                        csrf_value = csrf_input.get('value', '')
                        logging.debug("Found CSRF token in form input")
                
                if not csrf_value:
                    logging.error("Could not find CSRF token")
                    return False
                
                # Store cookies from the response
                cookies = response.cookies
                
            # Now perform the login
            login_data = {
                '_csrf': csrf_value,
                'email': self.email,
                'password': self.password,
                'redir': '/',
                'login': 'true'
            }
            
            headers = {
                **self.headers,
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-Csrf-Token': csrf_value,
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': f"{self.base_url}/login",
                'Origin': self.base_url
            }
            
            logging.info("Attempting to log in...")
            async with self.session.post(
                f"{self.base_url}/login", 
                data=login_data,
                headers=headers,
                allow_redirects=True,
                cookies=cookies
            ) as response:
                if response.status not in [200, 302]:
                    logging.error(f"Login failed with status: {response.status}")
                    return False
                    
                # Store the cookies from the login response
                self.session.cookie_jar.update_cookies(response.cookies)
                
                # Check if login was successful
                try:
                    response_data = await response.json()
                    if response_data.get('redir'):
                        logging.info("Login successful")
                        return True
                except:
                    # If we can't parse JSON, check the response text
                    text = await response.text()
                    if "Invalid email or password" in text:
                        logging.error("Login failed: Invalid credentials")
                        return False
                    elif response.status == 302 or "Redirecting" in text:
                        logging.info("Login successful (redirect)")
                        return True
                        
                logging.error("Login failed: Unexpected response")
                return False
                
        except Exception as e:
            logging.error(f"Error during login: {str(e)}")
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        await self.login()  # Attempt to login when entering context
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_templates(self) -> list:
        """
        Fetch templates from local JSON files.
        
        Returns:
            list: List of templates with their details.
        """
        try:
            templates = []
            # Read all general template files
            for file_path in self.general_dir.glob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template = json.load(f)
                        # Try to find and merge detail data
                        detail_file = self.detail_dir / f"{file_path.stem.replace('general_', '')}.json"
                        if detail_file.exists():
                            with open(detail_file, 'r', encoding='utf-8') as df:
                                detail_data = json.load(df)
                                template.update(detail_data)
                        templates.append(template)
                except Exception as e:
                    logging.error(f"Error reading template file {file_path}: {str(e)}")
                    continue
            
            logging.info(f"Successfully loaded {len(templates)} templates from local files")
            return templates
            
        except Exception as e:
            logging.error(f"Error loading templates from files: {str(e)}")
            return []
            
    async def get_template_details(self, template_id: str) -> Optional[dict]:
        """
        Fetch details for a specific template from local JSON files.
        
        Args:
            template_id (str): The ID of the template.
            
        Returns:
            Optional[dict]: Template details or None if not found.
        """
        try:
            # Look for the specific template file
            general_file = self.general_dir / f"general_{template_id}.json"
            detail_file = self.detail_dir / f"{template_id}.json"
            
            if not general_file.exists():
                return None
                
            with open(general_file, 'r', encoding='utf-8') as f:
                template = json.load(f)
                
            if detail_file.exists():
                with open(detail_file, 'r', encoding='utf-8') as f:
                    detail_data = json.load(f)
                    template.update(detail_data)
                    
            return template
            
        except Exception as e:
            logging.error(f"Error fetching template details: {str(e)}")
            return None

    def _get_element_text(self, element: Optional[Union[Tag, NavigableString]], default: str = "") -> str:
        """Safely get text from a BeautifulSoup element."""
        if element and isinstance(element, Tag):
            return element.get_text(strip=True)
        return default

    def _get_element_attr(self, element: Optional[Tag], attr: str, default: str = "") -> str:
        """Safely get attribute from a BeautifulSoup element."""
        if element and isinstance(element, Tag):
            value = element.get(attr, default)
            return value if isinstance(value, str) else default
        return default

    def _extract_template_id(self, view_pdf_link: str) -> str:
        """Extract template ID from the view PDF link."""
        match = re.search(r'/([^/]+)\.pdf$', view_pdf_link)
        return match.group(1) if match else ""

    def _extract_template_name(self, view_pdf_link: str) -> str:
        """Extract template name from the view PDF link."""
        match = re.search(r'/templates/([^/]+)/', view_pdf_link)
        return match.group(1) if match else ""

    async def download_template_details(self, open_template_link: str, view_pdf_link: str) -> Optional[Dict[str, Any]]:
        """Download template details from Overleaf."""
        if not self.session:
            logging.error("Session not initialized")
            return None

        try:
            # Extract template ID and name
            template_id = self._extract_template_id(view_pdf_link)
            template_name = self._extract_template_name(view_pdf_link)
            
            logging.info(f"Attempting to download template details for ID: {template_id}, Name: {template_name}")

            # Construct gallery URL
            gallery_url = f"{self.base_url}/latex/templates/{template_name}/{template_id}"
            logging.info(f"Accessing gallery URL: {gallery_url}")

            # Fetch template details
            async with self.session.get(
                gallery_url,
                allow_redirects=True,
                headers={
                    **self.headers,
                    'Referer': self.base_url
                }
            ) as response:
                logging.info(f"Gallery response status: {response.status}")
                if response.status != 200:
                    logging.error(f"Failed to fetch template details: {response.status}")
                    return None

                try:
                    html = await response.text(encoding='utf-8')
                    logging.debug("Successfully retrieved HTML content")
                except UnicodeDecodeError:
                    try:
                        html = await response.text(encoding='latin1')
                        logging.debug("Retrieved HTML content using latin1 encoding")
                    except UnicodeDecodeError:
                        html = await response.text(encoding='iso-8859-1')
                        logging.debug("Retrieved HTML content using iso-8859-1 encoding")

                # Parse HTML
                soup = BeautifulSoup(html, 'html.parser')
                logging.debug("Successfully parsed HTML content")

                # Extract title from h1 tag with class 'h2'
                title = self._get_element_text(soup.find('h1', class_='h2'))
                logging.info(f"Found template title: {title}")

                # Find all template detail divs
                template_details = soup.find_all('div', class_='template-detail')
                logging.debug(f"Found {len(template_details)} template detail sections")
                
                # Initialize variables with default values
                author = "Unknown"
                description = ""
                last_updated = ""
                license = "Unknown"
                abstract = ""
                preview_image_url = ""
                source_code_link = ""

                # Extract information from template details
                for detail in template_details:
                    if isinstance(detail, Tag):
                        text = detail.get_text(strip=True)
                        if text.startswith('作者:'):
                            author = text.replace('作者:', '').strip()
                            logging.debug(f"Found author: {author}")
                        elif text.startswith('摘要:'):
                            abstract = text.replace('摘要:', '').strip()
                            logging.debug(f"Found abstract: {abstract[:50]}...")
                            if not description:
                                description = abstract
                        elif text.startswith('许可:'):
                            license = text.replace('许可:', '').strip()
                            logging.debug(f"Found license: {license}")
                        elif text.startswith('最近上传:'):
                            last_updated = text.replace('最近上传:', '').strip()
                            logging.debug(f"Found last updated: {last_updated}")

                # Find preview image
                img = soup.find('img', src=re.compile(r'writelatex\.s3\.amazonaws\.com'))
                if isinstance(img, Tag):
                    preview_image_url = img.get('src', '')
                    logging.debug(f"Found preview image URL: {preview_image_url}")

                # Find source code link - try different approaches
                logging.info("Looking for source code link...")
                
                # Try to find the download button
                download_btn = soup.find('a', {'class': 'btn-download'})
                if download_btn and isinstance(download_btn, Tag):
                    href = download_btn.get('href')
                    if href and isinstance(href, str):
                        source_code_link = href
                        if not source_code_link.startswith('http'):
                            source_code_link = f"{self.base_url}{source_code_link}"
                        logging.info(f"Found source code link in download button: {source_code_link}")
                
                # If not found, try data attributes
                if not source_code_link:
                    for element in soup.find_all(attrs={"data-url": True}):
                        if isinstance(element, Tag):
                            data_url = element.get('data-url')
                            if isinstance(data_url, str) and ('/download/' in data_url or '/template/' in data_url):
                                source_code_link = data_url
                                if not source_code_link.startswith('http'):
                                    source_code_link = f"{self.base_url}{source_code_link}"
                                logging.info(f"Found source code link in data-url: {source_code_link}")
                                break
                
                # If still not found, try to find it in onclick attributes
                if not source_code_link:
                    for element in soup.find_all(attrs={"onclick": True}):
                        if isinstance(element, Tag):
                            onclick = element.get('onclick')
                            if isinstance(onclick, str) and 'download' in onclick.lower():
                                match = re.search(r"window\.location='([^']+)'", onclick)
                                if match:
                                    url = match.group(1)
                                    if isinstance(url, str):
                                        source_code_link = url
                                        if not source_code_link.startswith('http'):
                                            source_code_link = f"{self.base_url}{source_code_link}"
                                        logging.info(f"Found source code link in onclick: {source_code_link}")
                                        break

                # Create template details dictionary
                template_details = {
                    'id': template_id,
                    'link': open_template_link,
                    'view_pdf_link': view_pdf_link,
                    'open_template_link': open_template_link,
                    'title': title,
                    'author': author,
                    'description': description,
                    'last_updated': last_updated,
                    'license': license,
                    'abstract': abstract,
                    'preview_image_url': preview_image_url,
                    'source_code_link': source_code_link
                }

                logging.info(f"Successfully downloaded template details for ID: {template_id}")
                return template_details

        except Exception as e:
            logging.error(f"Error downloading template details: {str(e)}")
            return None