"""Test suite for CRUD Instruction.

This module provides comprehensive test cases for the CRUDInstruction 
to ensure robust and reliable entity management functionality.
"""

import pytest
import asyncio
from typing import Dict, Any

from agentflow.core.instructions.crud import (
    CRUDInstruction, 
    CRUDContext, 
    CRUDInstructionType
)
from agentflow.core.memory import WorkingMemory
from agentflow.core.llm import MockLLM

class TestCRUDInstruction:
    """Comprehensive test suite for CRUD Instruction."""
    
    @pytest.fixture
    def crud_instruction(self):
        """Create a CRUD instruction instance for testing."""
        memory = WorkingMemory()
        llm = MockLLM()
        return CRUDInstruction(llm=llm, memory=memory)
    
    @pytest.fixture
    def sample_entity(self):
        """Generate a sample entity for testing."""
        return {
            "name": "Test Entity",
            "description": "A sample entity for testing",
            "type": "test",
            "active": True
        }
    
    @pytest.fixture
    def crud_context(self):
        """Create a standard CRUD context for testing."""
        return CRUDContext(
            entity_type="test_entity",
            operation=CRUDInstructionType.CREATE,
            user_permissions=["admin", "write"]
        )
    
    @pytest.mark.asyncio
    async def test_create_entity(self, crud_instruction, crud_context, sample_entity):
        """Test creating a new entity."""
        # Execute create operation
        result = await crud_instruction.create(crud_context, sample_entity)
        
        # Assertions
        assert result.success is True
        assert "id" in result.output
        assert result.output["name"] == sample_entity["name"]
        assert result.metrics["operation"] == "create"
    
    @pytest.mark.asyncio
    async def test_read_entity(self, crud_instruction, crud_context, sample_entity):
        """Test reading an existing entity."""
        # First create an entity
        create_result = await crud_instruction.create(crud_context, sample_entity)
        entity_id = create_result.output["id"]
        
        # Now read the entity
        crud_context.operation = CRUDInstructionType.READ
        read_result = await crud_instruction.read(crud_context, entity_id)
        
        # Assertions
        assert read_result.success is True
        assert read_result.output["name"] == sample_entity["name"]
        assert read_result.metrics["operation"] == "read"
    
    @pytest.mark.asyncio
    async def test_update_entity(self, crud_instruction, crud_context, sample_entity):
        """Test updating an existing entity."""
        # First create an entity
        create_result = await crud_instruction.create(crud_context, sample_entity)
        entity_id = create_result.output["id"]
        
        # Prepare update data
        update_data = {"description": "Updated description"}
        crud_context.operation = CRUDInstructionType.UPDATE
        
        # Execute update
        update_result = await crud_instruction.update(crud_context, entity_id, update_data)
        
        # Assertions
        assert update_result.success is True
        assert update_result.output["description"] == "Updated description"
        assert update_result.metrics["operation"] == "update"
    
    @pytest.mark.asyncio
    async def test_delete_entity(self, crud_instruction, crud_context, sample_entity):
        """Test deleting an existing entity."""
        # First create an entity
        create_result = await crud_instruction.create(crud_context, sample_entity)
        entity_id = create_result.output["id"]
        
        # Prepare delete operation
        crud_context.operation = CRUDInstructionType.DELETE
        
        # Execute delete
        delete_result = await crud_instruction.delete(crud_context, entity_id)
        
        # Assertions
        assert delete_result.success is True
        assert delete_result.output["deleted"] is True
        assert delete_result.metrics["operation"] == "delete"
    
    @pytest.mark.asyncio
    async def test_search_entities(self, crud_instruction, crud_context, sample_entity):
        """Test searching for entities."""
        # Create multiple entities
        entities = [
            sample_entity,
            {**sample_entity, "name": "Another Test Entity"},
            {**sample_entity, "active": False}
        ]
        
        # Create entities
        for entity in entities:
            crud_context.operation = CRUDInstructionType.CREATE
            await crud_instruction.create(crud_context, entity)
        
        # Prepare search
        crud_context.operation = CRUDInstructionType.SEARCH
        search_query = {"active": True}
        
        # Execute search
        search_result = await crud_instruction.search(crud_context, search_query)
        
        # Assertions
        assert search_result.success is True
        assert len(search_result.output) > 0
        assert all(entity["active"] is True for entity in search_result.output)
        assert search_result.metrics["operation"] == "search"
    
    @pytest.mark.asyncio
    async def test_create_invalid_entity(self, crud_instruction, crud_context):
        """Test creating an invalid entity."""
        # Attempt to create an invalid entity
        invalid_entity = {}
        crud_context.operation = CRUDInstructionType.CREATE
        
        # Execute create with invalid data
        result = await crud_instruction.create(crud_context, invalid_entity)
        
        # Assertions
        assert result.success is False
        assert "error" in result.output
        assert "validation" in result.output["error"].lower()
    
    @pytest.mark.asyncio
    async def test_read_nonexistent_entity(self, crud_instruction, crud_context):
        """Test reading a non-existent entity."""
        crud_context.operation = CRUDInstructionType.READ
        
        # Attempt to read non-existent entity
        result = await crud_instruction.read(crud_context, "non_existent_id")
        
        # Assertions
        assert result.success is False
        assert "error" in result.output
        assert "not found" in result.output["error"].lower()
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_entity(self, crud_instruction, crud_context):
        """Test updating a non-existent entity."""
        crud_context.operation = CRUDInstructionType.UPDATE
        
        # Attempt to update non-existent entity
        result = await crud_instruction.update(
            crud_context, 
            "non_existent_id", 
            {"name": "Updated Name"}
        )
        
        # Assertions
        assert result.success is False
        assert "error" in result.output
        assert "not found" in result.output["error"].lower()
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_entity(self, crud_instruction, crud_context):
        """Test deleting a non-existent entity."""
        crud_context.operation = CRUDInstructionType.DELETE
        
        # Attempt to delete non-existent entity
        result = await crud_instruction.delete(crud_context, "non_existent_id")
        
        # Assertions
        assert result.success is False
        assert "error" in result.output
        assert "not found" in result.output["error"].lower()
    
    @pytest.mark.asyncio
    async def test_audit_trail(self, crud_instruction, crud_context, sample_entity):
        """Test audit trail functionality."""
        # Create an entity
        crud_context.operation = CRUDInstructionType.CREATE
        create_result = await crud_instruction.create(crud_context, sample_entity)
        entity_id = create_result.output["id"]
        
        # Perform various operations
        crud_context.operation = CRUDInstructionType.READ
        await crud_instruction.read(crud_context, entity_id)
        
        crud_context.operation = CRUDInstructionType.UPDATE
        await crud_instruction.update(
            crud_context, 
            entity_id, 
            {"description": "Updated"}
        )
        
        crud_context.operation = CRUDInstructionType.DELETE
        await crud_instruction.delete(crud_context, entity_id)
        
        # Check audit trail
        assert len(crud_context.audit_trail) > 0
        assert all(
            entry.get("operation") in ["CREATE", "READ", "UPDATE", "DELETE"] 
            for entry in crud_context.audit_trail
        )

# Utility function to run tests
def run_tests():
    """Run the test suite."""
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    run_tests()
