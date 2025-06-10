"""CRUD Instruction Set for Agent Improvement.

This module provides a comprehensive CRUD (Create, Read, Update, Delete) 
instruction set to enhance agent capabilities across various domains.
"""

import asyncio
import logging
import time
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Set, Union, Type, cast
from enum import Enum, auto
from dataclasses import dataclass, field
from pathlib import Path

from agentflow.core.instructions import (
    BaseInstruction,
    InstructionType,
    InstructionMetadata,
    InstructionResult,
    InstructionStatus
)
from agentflow.core.memory import WorkingMemory
from agentflow.core.llm import BaseLLM, DefaultLLM
from agentflow.core.errors import (
    ValidationError,
    ResourceError,
    EntityNotFoundError,
    OperationError
)
from .base import (
    ExecutionContext,
    ExecutionResult,
    InstructionMetrics
)

logger = logging.getLogger(__name__)

@dataclass
class CRUDContext:
    """Context for CRUD operations."""
    entity_type: str
    operation: 'CRUDInstructionType'
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_permissions: Set[str] = field(default_factory=set)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    resource_limits: Dict[str, float] = field(default_factory=dict)

@dataclass
class CRUDMetrics:
    """Metrics for CRUD operations."""
    operation_count: int = 0
    validation_errors: int = 0
    operation_time: float = 0.0
    memory_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    entities_processed: int = 0

class CRUDInstructionType(Enum):
    """Types of CRUD operations."""
    CREATE = auto()
    READ = auto()
    UPDATE = auto()
    DELETE = auto()
    SEARCH = auto()
    VALIDATE = auto()

class CRUDInstruction(BaseInstruction):
    """Base class for CRUD instructions."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        instruction_type: Optional[InstructionType] = None,
        metadata: Optional[InstructionMetadata] = None,
        llm: Optional[BaseLLM] = None,
        memory: Optional[WorkingMemory] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize CRUD instruction."""
        super().__init__(
            name=name,
            description=description,
            instruction_type=instruction_type or InstructionType.DATA_MANAGEMENT,
            metadata=metadata or InstructionMetadata(
                capabilities={"crud"},
                tags={name.lower(), "crud"},
                cleanup_required=True
            )
        )
        self.llm: BaseLLM = llm or DefaultLLM()
        self.memory: WorkingMemory = memory or WorkingMemory()
        self.config: Dict[str, Any] = config or {}
        self.metrics = CRUDMetrics()
        
        # Initialize state
        self._entity_cache: Dict[str, Any] = {}
        self._active_transactions: Set[str] = set()
        self._temp_files: Set[Path] = set()
        
        # Register required parameters
        self.add_required_param("entity_type", str)
        self.add_required_param("operation", str)
        self.add_required_param("data", Dict[str, Any])
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clean up temp files
            for file_path in self._temp_files:
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {file_path}: {e}")
            
            # Clear cache
            self._entity_cache.clear()
            
            # Roll back active transactions
            for transaction_id in self._active_transactions:
                try:
                    await self._rollback_transaction(transaction_id)
                except Exception as e:
                    logger.error(f"Failed to rollback transaction {transaction_id}: {e}")
            
            # Call parent cleanup
            await super().cleanup()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise ResourceError(f"Failed to clean up resources: {str(e)}")
    
    async def _validate_operation(
        self, 
        context: CRUDContext, 
        data: Dict[str, Any]
    ) -> bool:
        """Validate CRUD operation permissions and data integrity."""
        start_time = time.time()
        
        try:
            # Check permissions
            if not context.user_permissions:
                logger.warning("No user permissions defined")
                self.metrics.validation_errors += 1
                return False
            
            # Validate data structure
            if not self._validate_data_structure(data):
                logger.error("Invalid data structure")
                self.metrics.validation_errors += 1
                return False
            
            # Check resource limits
            if not self._check_resource_limits(context.resource_limits):
                logger.error("Resource limits exceeded")
                self.metrics.validation_errors += 1
                return False
            
            # Use LLM for advanced validation if available
            if self.llm:
                validation_result = await self._llm_validate(context, data)
                if not validation_result:
                    self.metrics.validation_errors += 1
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.metrics.validation_errors += 1
            return False
            
        finally:
            self.metrics.operation_time += time.time() - start_time
    
    def _validate_data_structure(self, data: Dict[str, Any]) -> bool:
        """Validate data structure against schema."""
        try:
            # Basic validation
            if not isinstance(data, dict):
                return False
                
            # Check required fields
            required_fields = self.config.get("required_fields", [])
            if not all(field in data for field in required_fields):
                return False
                
            # Validate field types
            field_types = self.config.get("field_types", {})
            for field, expected_type in field_types.items():
                if field in data and not isinstance(data[field], expected_type):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _check_resource_limits(self, limits: Dict[str, float]) -> bool:
        """Check if operation would exceed resource limits."""
        try:
            current_memory = self.metrics.memory_usage
            max_memory = limits.get("memory", float("inf"))
            
            if current_memory >= max_memory:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _llm_validate(
        self,
        context: CRUDContext,
        data: Dict[str, Any]
    ) -> bool:
        """Use LLM for advanced validation."""
        try:
            validation_prompt = f"""
            Validate the following {context.entity_type} data for {context.operation.name} operation:
            
            Data: {data}
            
            Validation Rules:
            1. Check data completeness
            2. Verify data types
            3. Assess data consistency
            4. Check business rules
            5. Validate relationships
            
            Return JSON with:
            - is_valid: boolean
            - validation_errors: list of strings
            - suggestions: list of strings
            """
            
            validation_result = await self.llm.generate(validation_prompt)
            validation_data = self.llm.parse_json(validation_result)
            
            if not validation_data.get("is_valid", False):
                logger.warning(
                    f"LLM Validation failed: {validation_data.get('validation_errors', [])}"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return False
    
    async def _start_transaction(self) -> str:
        """Start a new transaction."""
        transaction_id = str(uuid.uuid4())
        self._active_transactions.add(transaction_id)
        return transaction_id
    
    async def _commit_transaction(self, transaction_id: str) -> None:
        """Commit a transaction."""
        if transaction_id in self._active_transactions:
            self._active_transactions.remove(transaction_id)
    
    async def _rollback_transaction(self, transaction_id: str) -> None:
        """Rollback a transaction."""
        if transaction_id in self._active_transactions:
            # Implement rollback logic
            self._active_transactions.remove(transaction_id)
    
    async def create(
        self, 
        context: CRUDContext, 
        data: Dict[str, Any]
    ) -> ExecutionResult:
        """Create a new entity."""
        start_time = time.time()
        transaction_id = await self._start_transaction()
        
        try:
            # Validate operation
            if not await self._validate_operation(context, data):
                raise ValidationError("Failed to validate entity creation")
            
            # Generate unique identifier
            entity_id = await self._generate_unique_id(context, data)
            
            # Store entity
            created_entity = {**data, "id": entity_id}
            await self.memory.store(
                f"{context.entity_type}_{entity_id}", 
                created_entity
            )
            
            # Update cache
            self._entity_cache[entity_id] = created_entity
            
            # Update metrics
            self.metrics.operation_count += 1
            self.metrics.entities_processed += 1
            
            # Log audit trail
            await self._log_audit_trail(context, data, created_entity)
            
            # Commit transaction
            await self._commit_transaction(transaction_id)
            
            return ExecutionResult(
                success=True,
                output=created_entity,
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            # Rollback transaction
            await self._rollback_transaction(transaction_id)
            
            logger.error(f"Create operation failed: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
    
    async def read(
        self, 
        context: CRUDContext, 
        entity_id: str
    ) -> ExecutionResult:
        """Read an existing entity."""
        start_time = time.time()
        
        try:
            # Check cache first
            if entity_id in self._entity_cache:
                self.metrics.cache_hits += 1
                entity = self._entity_cache[entity_id]
            else:
                self.metrics.cache_misses += 1
                entity = await self.memory.retrieve(
                    f"{context.entity_type}_{entity_id}"
                )
                if entity:
                    self._entity_cache[entity_id] = entity
            
            if not entity:
                raise EntityNotFoundError(f"Entity {entity_id} not found")
            
            # Update metrics
            self.metrics.operation_count += 1
            self.metrics.entities_processed += 1
            
            # Log audit trail
            await self._log_audit_trail(context, {"id": entity_id}, entity)
            
            return ExecutionResult(
                success=True,
                output=entity,
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
        except EntityNotFoundError as e:
            logger.error(f"Entity not found: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            logger.error(f"Read operation failed: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
    
    async def update(
        self, 
        context: CRUDContext, 
        entity_id: str, 
        data: Dict[str, Any]
    ) -> ExecutionResult:
        """Update an existing entity."""
        start_time = time.time()
        transaction_id = await self._start_transaction()
        
        try:
            # Validate operation
            if not await self._validate_operation(context, data):
                raise ValidationError("Failed to validate entity update")
            
            # Retrieve existing entity
            existing_entity = await self.memory.retrieve(
                f"{context.entity_type}_{entity_id}"
            )
            
            if not existing_entity:
                raise EntityNotFoundError(f"Entity {entity_id} not found")
            
            # Update entity
            updated_entity = {**existing_entity, **data}
            await self.memory.store(
                f"{context.entity_type}_{entity_id}", 
                updated_entity
            )
            
            # Update cache
            self._entity_cache[entity_id] = updated_entity
            
            # Update metrics
            self.metrics.operation_count += 1
            self.metrics.entities_processed += 1
            
            # Log audit trail
            await self._log_audit_trail(context, data, updated_entity)
            
            # Commit transaction
            await self._commit_transaction(transaction_id)
            
            return ExecutionResult(
                success=True,
                output=updated_entity,
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
        except (ValidationError, EntityNotFoundError) as e:
            # Rollback transaction
            await self._rollback_transaction(transaction_id)
            
            logger.error(f"Update operation failed: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            # Rollback transaction
            await self._rollback_transaction(transaction_id)
            
            logger.error(f"Update operation failed: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
    
    async def delete(
        self, 
        context: CRUDContext, 
        entity_id: str
    ) -> ExecutionResult:
        """Delete an existing entity."""
        start_time = time.time()
        transaction_id = await self._start_transaction()
        
        try:
            # Retrieve existing entity
            existing_entity = await self.memory.retrieve(
                f"{context.entity_type}_{entity_id}"
            )
            
            if not existing_entity:
                raise EntityNotFoundError(f"Entity {entity_id} not found")
            
            # Delete entity
            await self.memory.delete(
                f"{context.entity_type}_{entity_id}"
            )
            
            # Remove from cache
            self._entity_cache.pop(entity_id, None)
            
            # Update metrics
            self.metrics.operation_count += 1
            self.metrics.entities_processed += 1
            
            # Log audit trail
            await self._log_audit_trail(context, {"id": entity_id}, {"deleted": True})
            
            # Commit transaction
            await self._commit_transaction(transaction_id)
            
            return ExecutionResult(
                success=True,
                output={"deleted": True, "entity_id": entity_id},
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
        except EntityNotFoundError as e:
            # Rollback transaction
            await self._rollback_transaction(transaction_id)
            
            logger.error(f"Delete operation failed: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            # Rollback transaction
            await self._rollback_transaction(transaction_id)
            
            logger.error(f"Delete operation failed: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
    
    async def search(
        self, 
        context: CRUDContext, 
        query: Dict[str, Any]
    ) -> ExecutionResult:
        """Search for entities matching query."""
        start_time = time.time()
        
        try:
            # Validate query
            if not self._validate_search_query(query):
                raise ValidationError("Invalid search query")
            
            # Retrieve matching entities
            matching_entities = await self._search_entities(context, query)
            
            # Update metrics
            self.metrics.operation_count += 1
            self.metrics.entities_processed += len(matching_entities)
            
            # Log audit trail
            await self._log_audit_trail(context, query, {
                "matches": len(matching_entities)
            })
            
            return ExecutionResult(
                success=True,
                output={
                    "entities": matching_entities,
                    "total": len(matching_entities)
                },
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
        except ValidationError as e:
            logger.error(f"Search operation failed: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
    
    def _validate_search_query(self, query: Dict[str, Any]) -> bool:
        """Validate search query structure."""
        try:
            if not isinstance(query, dict):
                return False
            
            # Validate operators
            valid_operators = {"eq", "ne", "gt", "lt", "gte", "lte", "in", "nin"}
            for key, value in query.items():
                if isinstance(value, dict):
                    if not all(op in valid_operators for op in value.keys()):
                        return False
            
            return True
            
        except Exception:
            return False
    
    async def _search_entities(
        self,
        context: CRUDContext,
        query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search for entities matching query."""
        try:
            # Get all entities
            all_entities = await self.memory.retrieve_all(
                f"{context.entity_type}_*"
            )
            
            # Apply filters
            matching_entities = []
            for entity in all_entities:
                if self._matches_query(entity, query):
                    matching_entities.append(entity)
            
            return matching_entities
            
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            raise OperationError(f"Failed to search entities: {str(e)}")
    
    def _matches_query(self, entity: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if entity matches query."""
        try:
            for key, value in query.items():
                if isinstance(value, dict):
                    # Handle operators
                    for op, op_value in value.items():
                        if not self._apply_operator(
                            entity.get(key),
                            op,
                            op_value
                        ):
                            return False
                else:
                    # Direct comparison
                    if entity.get(key) != value:
                        return False
            return True
            
        except Exception:
            return False
    
    def _apply_operator(
        self,
        field_value: Any,
        operator: str,
        query_value: Any
    ) -> bool:
        """Apply comparison operator."""
        try:
            if operator == "eq":
                return field_value == query_value
            elif operator == "ne":
                return field_value != query_value
            elif operator == "gt":
                return field_value > query_value
            elif operator == "lt":
                return field_value < query_value
            elif operator == "gte":
                return field_value >= query_value
            elif operator == "lte":
                return field_value <= query_value
            elif operator == "in":
                return field_value in query_value
            elif operator == "nin":
                return field_value not in query_value
            else:
                return False
                
        except Exception:
            return False
    
    async def _generate_unique_id(
        self, 
        context: CRUDContext, 
        data: Dict[str, Any]
    ) -> str:
        """Generate a unique identifier for an entity."""
        timestamp = int(time.time() * 1000)
        data_hash = hashlib.md5(
            str(data).encode('utf-8')
        ).hexdigest()[:8]
        
        return f"{context.entity_type}_{timestamp}_{data_hash}"
    
    async def _log_audit_trail(
        self, 
        context: CRUDContext, 
        data: Dict[str, Any], 
        result: Dict[str, Any]
    ) -> None:
        """Log audit trail for CRUD operation."""
        audit_entry = {
            "timestamp": time.time(),
            "operation": context.operation.name,
            "entity_type": context.entity_type,
            "input_data": data,
            "result": result,
            "user_permissions": list(context.user_permissions),
            "metrics": {
                "operation_time": self.metrics.operation_time,
                "memory_usage": self.metrics.memory_usage
            }
        }
        
        context.audit_trail.append(audit_entry)
        
        # Store in persistent memory
        await self.memory.store(
            f"audit_trail_{context.entity_type}",
            context.audit_trail
        )
    
    async def execute(self, context: Dict[str, Any]) -> ExecutionResult:
        """Execute CRUD operation based on context."""
        # Convert context to CRUDContext
        crud_context = CRUDContext(
            entity_type=context.get('entity_type', 'default'),
            operation=context.get('operation', CRUDInstructionType.READ),
            metadata=context.get('metadata', {}),
            user_permissions=set(context.get('user_permissions', [])),
            resource_limits=context.get('resource_limits', {})
        )
        
        # Dispatch to appropriate method
        try:
            if crud_context.operation == CRUDInstructionType.CREATE:
                return await self.create(crud_context, context.get('data', {}))
            elif crud_context.operation == CRUDInstructionType.READ:
                return await self.read(crud_context, context.get('entity_id', ''))
            elif crud_context.operation == CRUDInstructionType.UPDATE:
                return await self.update(
                    crud_context, 
                    context.get('entity_id', ''), 
                    context.get('data', {})
                )
            elif crud_context.operation == CRUDInstructionType.DELETE:
                return await self.delete(crud_context, context.get('entity_id', ''))
            elif crud_context.operation == CRUDInstructionType.SEARCH:
                return await self.search(crud_context, context.get('query', {}))
            else:
                return ExecutionResult(
                    success=False,
                    output={},
                    errors=[{"error": "Unsupported CRUD operation"}],
                    metrics={
                        "crud_metrics": self.metrics.__dict__,
                        "operation": crud_context.operation.name
                    }
                )
                
        except Exception as e:
            logger.error(f"CRUD operation failed: {e}")
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={
                    "crud_metrics": self.metrics.__dict__,
                    "operation": crud_context.operation.name
                }
            )
