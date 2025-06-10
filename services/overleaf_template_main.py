"""Main module for fetching Overleaf templates and storing them in Supabase."""
import asyncio
import logging
import sys
from dotenv import load_dotenv
import os

# Configure logging to show detailed information
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler('overleaf_template_sync.log')  # Log to file
    ]
)

# Load environment variables
load_dotenv()

# Check if required environment variables are set
if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
    logging.error("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    sys.exit(1)

from services.overleaf_client import OverleafClient
from services.supabase_service import SupabaseService

async def fetch_and_store_templates():
    """Fetch templates from Overleaf and store them in Supabase."""
    try:
        # Log environment variables for debugging
        logging.info(f"Supabase URL: {os.getenv('SUPABASE_URL')}")
        logging.info(f"Supabase Service Key Present: {bool(os.getenv('SUPABASE_SERVICE_KEY'))}")

        # Initialize clients
        overleaf_client = OverleafClient()
        supabase_service = SupabaseService()

        # Fetch templates from Overleaf
        templates = await overleaf_client.get_templates()
        if not templates:
            logging.warning("No templates found")
            return

        # Transform templates for storage
        transformed_templates = []
        for template in templates:
            try:
                transformed_template = {
                    "title": template.get("title", ""),
                    "description": template.get("description", ""),
                    "author": template.get("author", "Unknown"),
                    "link": template.get("link", ""),
                    "preview_image_url": template.get("preview_image_url", ""),
                    "last_updated": template.get("last_updated", ""),
                    "license": template.get("license", "Unknown"),
                    "abstract": template.get("abstract", ""),
                    "open_template_link": template.get("open_template_link", ""),
                    "view_pdf_link": template.get("view_pdf_link", ""),
                    "tags": template.get("tags", [])
                }
                transformed_templates.append(transformed_template)
            except Exception as transform_error:
                logging.error(f"Error transforming template: {transform_error}")

        # Log number of templates before sync
        logging.info(f"Prepared {len(transformed_templates)} templates for sync")

        # Sync templates with Supabase
        await supabase_service.sync_templates(transformed_templates)
        logging.info(f"Successfully synced {len(transformed_templates)} templates")

    except Exception as e:
        logging.error(f"Error in fetch_and_store_templates: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        
        # Additional error context
        import traceback
        logging.error(traceback.format_exc())
        
        raise

async def main():
    """Main function."""
    try:
        await fetch_and_store_templates()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        
        # Additional error context
        import traceback
        logging.error(traceback.format_exc())
        
        raise

if __name__ == "__main__":
    asyncio.run(main())