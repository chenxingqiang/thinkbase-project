from typing import Dict, List, Any, Optional
from datetime import datetime
from config.supabase import SupabaseConfig
from config.auth import AuthService
from supabase_overleaf.db_handler import SupabaseHandler
import logging
import os
from supabase import create_client

class SupabaseService:
    """Service for interacting with Supabase."""
    
    def __init__(self):
        """Initialize the Supabase service."""
        self.config = SupabaseConfig()
        self.handler = SupabaseHandler()
        self.auth = AuthService()
        
    async def ensure_authenticated(self) -> None:
        """Ensure we have an authenticated user."""
        if not self.auth.get_current_user():
            # Create a test user if not exists
            try:
                await self.auth.sign_in("test@example.com", "test123")
            except Exception:
                await self.auth.sign_up("test@example.com", "test123")
                await self.auth.sign_in("test@example.com", "test123")
                
    async def save_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save Overleaf template to Supabase.
        
        Args:
            template_data: Template data to save
            
        Returns:
            Saved template data
        """
        await self.ensure_authenticated()
        template_data["created_at"] = datetime.utcnow().isoformat()
        
        result = self.config.client.table("templates").insert(template_data).execute()
        return result.data[0] if result.data else {}
    
    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template by ID.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template data if found, None otherwise
        """
        await self.ensure_authenticated()
        result = self.config.client.table("templates").select("*").eq("id", template_id).execute()
        return result.data[0] if result.data else None
    
    async def list_templates(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List templates with optional filters.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            List of templates
        """
        await self.ensure_authenticated()
        query = self.config.client.table("templates").select("*")
        
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
                
        result = query.execute()
        return result.data if result.data else []
    
    async def update_template(self, template_id: str, template_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update template by ID.
        
        Args:
            template_id: Template ID
            template_data: Template data to update
            
        Returns:
            Updated template data if found, None otherwise
        """
        await self.ensure_authenticated()
        template_data["updated_at"] = datetime.utcnow().isoformat()
        
        result = self.config.client.table("templates").update(template_data).eq("id", template_id).execute()
        return result.data[0] if result.data else None
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete template by ID.
        
        Args:
            template_id: Template ID
            
        Returns:
            True if deleted successfully, False otherwise
        """
        await self.ensure_authenticated()
        result = self.config.client.table("templates").delete().eq("id", template_id).execute()
        return bool(result.data)
    
    async def sync_templates(self, templates: List[Dict[str, Any]]) -> None:
        """Synchronize templates with Supabase, using service role if available."""
        # Explicitly load service key from environment
        service_key = os.getenv('SUPABASE_SERVICE_KEY', '').strip()
        logging.info(f"Supabase Service Key Present: {bool(service_key)}")
        
        # If service key is available, create a new client with service role
        if service_key:
            try:
                service_client = create_client(
                    supabase_url=self.config.client.supabase_url, 
                    supabase_key=service_key
                )
                logging.info("Created service role client successfully")
            except Exception as e:
                logging.error(f"Error creating service client: {e}")
                service_client = self.config.client
        else:
            service_client = self.config.client
        
        # Ensure we have templates to sync
        if not templates:
            logging.warning("No templates to synchronize")
            return
        
        try:
            # Sync general templates
            await self.handler.insert_templates_general(
                templates, 
                client=service_client
            )
            
            # Sync detail templates
            await self.handler.insert_templates_detail(
                templates, 
                client=service_client
            )
            
            logging.info(f"Successfully synchronized {len(templates)} templates")
        
        except Exception as e:
            logging.error(f"Template synchronization error: {e}")
            raise