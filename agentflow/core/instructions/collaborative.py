"""Collaborative instruction implementations."""

from typing import Dict, Any, List, Optional, Set, cast, Union
from dataclasses import dataclass, field
from abc import abstractmethod
import asyncio
import logging
import time
import uuid
import json
from pathlib import Path

from .base import (
    BaseInstruction,
    InstructionType,
    InstructionMetadata,
    ExecutionContext,
    ExecutionResult
)

logger = logging.getLogger(__name__)

@dataclass
class CollaborationConfig:
    """Configuration for collaborative instructions."""
    required_capabilities: Set[str] = field(default_factory=set)
    min_collaborators: int = 1
    max_collaborators: int = 10
    timeout_seconds: float = 300.0
    retry_attempts: int = 3
    synchronization_mode: str = "barrier"  # barrier, async, or sequential

@dataclass
class CollaborationMetrics:
    """Metrics for collaborative operations."""
    messages_sent: int = 0
    messages_received: int = 0
    round_trips: int = 0
    execution_time: float = 0.0
    consensus_reached: bool = False
    participants: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = set()

class CollaborativeInstruction(BaseInstruction):
    """Base class for collaborative instructions."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        instruction_type: Optional[InstructionType] = None,
        metadata: Optional[InstructionMetadata] = None,
        config: Optional[CollaborationConfig] = None
    ):
        """Initialize collaborative instruction."""
        super().__init__(
            name=name,
            description=description,
            instruction_type=instruction_type or InstructionType.COLLABORATION,
            metadata=metadata or InstructionMetadata(
                capabilities={"collaboration"},
                tags={name.lower(), "collaborative"},
                cleanup_required=True,
                resource_requirements={"network": 1.0}
            )
        )
        self.collaboration_metrics = CollaborationMetrics()
        self.config = config or CollaborationConfig()
        
        # Register required parameters with proper type hints
        self.add_required_param("participants", List[str])
        self.add_required_param("message", Dict[str, Any])
        self.add_required_param("timeout", float)
        
        # Initialize state
        self._active_sessions: Set[str] = set()
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._state: Dict[str, Any] = {}
        self._temp_files: Set[Path] = set()
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute collaborative instruction."""
        start_time = time.time()
        
        try:
            # Validate context and parameters
            valid, error_msg = context.validate_required_params(self.get_required_params())
            if not valid:
                return ExecutionResult(
                    success=False,
                    output={},
                    errors=[{"error": error_msg}]
                )
            
            # Validate collaboration info
            if not context.collaboration_info:
                return ExecutionResult(
                    success=False,
                    output={},
                    errors=[{"error": "Missing collaboration information"}]
                )
            
            # Validate resource requirements
            if self.metadata.resource_requirements:
                valid, error_msg = context.validate_resources(self.metadata.resource_requirements)
                if not valid:
                    return ExecutionResult(
                        success=False,
                        output={},
                        errors=[{"error": error_msg}]
                    )
            
            # Validate participant count
            participants = cast(List[str], context.params["participants"])
            if not self.config.min_collaborators <= len(participants) <= self.config.max_collaborators:
                return ExecutionResult(
                    success=False,
                    output={},
                    errors=[{"error": f"Invalid number of participants: {len(participants)}"}]
                )
            
            # Start collaboration session
            session_id = await self._start_session(context)
            self._active_sessions.add(session_id)
            
            try:
                # Execute collaboration
                result = await self._collaborate(context, session_id)
                
                # Update metrics
                execution_time = time.time() - start_time
                self.collaboration_metrics.execution_time = execution_time
                self.update_metrics(True, execution_time)
                
                return ExecutionResult(
                    success=True,
                    output=result,
                    metrics={
                        "collaboration_metrics": self.collaboration_metrics.__dict__,
                        "execution_time": execution_time
                    }
                )
                
            finally:
                # End session
                await self._end_session(session_id)
                self._active_sessions.remove(session_id)
            
        except asyncio.TimeoutError:
            logger.error("Collaboration timed out")
            execution_time = time.time() - start_time
            self.update_metrics(False, execution_time)
            
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": "Collaboration timed out"}],
                metrics={"execution_time": execution_time}
            )
            
        except Exception as e:
            logger.error(f"Collaboration failed: {str(e)}")
            execution_time = time.time() - start_time
            self.update_metrics(False, execution_time)
            
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={"execution_time": execution_time}
            )
            
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Clean up collaboration resources."""
        try:
            # End any active sessions
            for session_id in list(self._active_sessions):
                try:
                    await self._end_session(session_id)
                    self._active_sessions.remove(session_id)
                except Exception as e:
                    logger.warning(f"Failed to end session {session_id}: {e}")
            
            # Clear message queue
            while not self._message_queue.empty():
                try:
                    self._message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Clear state
            self._state.clear()
            
            # Remove temporary files
            for file_path in self._temp_files:
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {file_path}: {e}")
            
            # Call parent cleanup
            await super().cleanup()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    @abstractmethod
    async def _start_session(self, context: ExecutionContext) -> str:
        """Start collaboration session."""
        raise NotImplementedError("Subclasses must implement _start_session")
    
    @abstractmethod
    async def _end_session(self, session_id: str) -> None:
        """End collaboration session."""
        raise NotImplementedError("Subclasses must implement _end_session")
    
    @abstractmethod
    async def _collaborate(self, context: ExecutionContext, session_id: str) -> Dict[str, Any]:
        """Execute collaboration logic."""
        raise NotImplementedError("Subclasses must implement _collaborate")
    
    def get_required_params(self) -> Set[str]:
        """Get required parameters."""
        return {"participants", "message", "timeout"}

class DataSharingInstruction(CollaborativeInstruction):
    """Instruction for sharing data between collaborators."""
    
    def __init__(
        self,
        name: str,
        collab_config: CollaborationConfig,
        metadata: Optional[InstructionMetadata] = None
    ):
        """Initialize data sharing instruction."""
        super().__init__(
            name=name,
            description="Share data between collaborators",
            instruction_type=InstructionType.COLLABORATION,
            metadata=metadata or InstructionMetadata(
                capabilities={"collaboration", "data_sharing"},
                tags={"data_sharing", "collaborative"},
                cleanup_required=True,
                resource_requirements={"network": 1.0, "memory": 256}
            ),
            config=collab_config
        )
        
        # Add specific parameters
        self.add_required_param("data_format", str)
        self.add_required_param("validation_rules", Dict[str, Any])
    
    async def _start_session(self, context: ExecutionContext) -> str:
        """Start data sharing session."""
        session_id = f"data_sharing_{uuid.uuid4()}"
        self._state[session_id] = {
            "shared_data": {},
            "participants": cast(List[str], context.params["participants"]),
            "start_time": time.time(),
            "validation_rules": context.params["validation_rules"]
        }
        return session_id
    
    async def _end_session(self, session_id: str) -> None:
        """End data sharing session."""
        if session_id in self._state:
            # Clean up any temporary files created during sharing
            session_data = self._state[session_id]
            if "temp_files" in session_data:
                for file_path in session_data["temp_files"]:
                    try:
                        path = Path(file_path)
                        if path.exists():
                            path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {file_path}: {e}")
            
            del self._state[session_id]
    
    async def _collaborate(self, context: ExecutionContext, session_id: str) -> Dict[str, Any]:
        """Share data with collaborators."""
        collab = self._state[session_id]
        
        try:
            # Validate data format
            data_format = context.params["data_format"]
            if not self._validate_data_format(data_format):
                raise ValueError(f"Invalid data format: {data_format}")
            
            # Share data
            shared_data = context.params["message"].get("data", {})
            if not self._validate_data(shared_data, collab["validation_rules"]):
                raise ValueError("Data validation failed")
            
            participant_id = cast(str, context.metadata["participant_id"])
            collab["shared_data"][participant_id] = shared_data
            self.collaboration_metrics.participants.add(participant_id)
            
            # Wait for all data to be shared
            timeout = cast(float, context.params["timeout"])
            try:
                async with asyncio.timeout(timeout):
                    while len(collab["shared_data"]) < len(collab["participants"]):
                        await asyncio.sleep(0.1)
                        if time.time() - collab["start_time"] > timeout:
                            raise TimeoutError("Data sharing timeout")
            except asyncio.TimeoutError:
                raise TimeoutError("Data sharing timeout")
            
            # Validate combined data
            combined_data = self._combine_shared_data(collab["shared_data"])
            
            return {
                "shared_data": combined_data,
                "participants": list(collab["shared_data"].keys()),
                "metrics": {
                    "total_participants": len(collab["shared_data"]),
                    "data_size": len(json.dumps(combined_data))
                }
            }
            
        except Exception as e:
            logger.error(f"Data sharing failed: {str(e)}")
            raise
    
    def _validate_data_format(self, data_format: str) -> bool:
        """Validate data format."""
        return data_format in {"json", "text", "binary"}
    
    def _validate_data(self, data: Dict[str, Any], rules: Dict[str, Any]) -> bool:
        """Validate shared data against rules."""
        try:
            for key, rule in rules.items():
                if key in data:
                    value = data[key]
                    if "type" in rule and not isinstance(value, eval(rule["type"])):
                        return False
                    if "min" in rule and value < rule["min"]:
                        return False
                    if "max" in rule and value > rule["max"]:
                        return False
            return True
        except Exception:
            return False
    
    def _combine_shared_data(self, shared_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine shared data from all participants."""
        combined: Dict[str, Any] = {}
        for participant_data in shared_data.values():
            for key, value in participant_data.items():
                if key not in combined:
                    combined[key] = []
                if value not in combined[key]:
                    combined[key].append(value)
        return combined

class ConsensusInstruction(CollaborativeInstruction):
    """Instruction for reaching consensus between collaborators."""
    
    def __init__(
        self,
        name: str = "consensus",
        description: str = "Reach consensus among participants",
        metadata: Optional[InstructionMetadata] = None,
        config: Optional[CollaborationConfig] = None
    ):
        """Initialize consensus instruction."""
        super().__init__(
            name=name,
            description=description,
            instruction_type=InstructionType.COLLABORATION,
            metadata=metadata or InstructionMetadata(
                capabilities={"collaboration", "consensus"},
                tags={"consensus", "collaborative", "agreement"},
                cleanup_required=True,
                resource_requirements={"network": 1.0, "memory": 512}
            ),
            config=config or CollaborationConfig(
                required_capabilities={"consensus"},
                min_collaborators=2,
                max_collaborators=10,
                timeout_seconds=300.0
            )
        )
        
        # Add specific parameters
        self.add_required_param("consensus_threshold", float)
        self.add_required_param("max_rounds", int)
        self.add_required_param("vote_options", List[str])
    
    async def _start_session(self, context: ExecutionContext) -> str:
        """Start consensus session."""
        session_id = f"consensus_{uuid.uuid4()}"
        self._state[session_id] = {
            "round": 0,
            "votes": {},
            "participants": cast(List[str], context.params["participants"]),
            "start_time": time.time(),
            "vote_options": context.params["vote_options"]
        }
        return session_id
    
    async def _end_session(self, session_id: str) -> None:
        """End consensus session."""
        if session_id in self._state:
            # Clean up any temporary vote records
            session_data = self._state[session_id]
            if "temp_files" in session_data:
                for file_path in session_data["temp_files"]:
                    try:
                        path = Path(file_path)
                        if path.exists():
                            path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {file_path}: {e}")
            
            del self._state[session_id]
    
    async def _collaborate(self, context: ExecutionContext, session_id: str) -> Dict[str, Any]:
        """Execute consensus logic."""
        try:
            threshold = cast(float, context.params["consensus_threshold"])
            max_rounds = cast(int, context.params["max_rounds"])
            timeout = cast(float, context.params["timeout"])
            
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Invalid consensus threshold: {threshold}")
            
            if max_rounds < 1:
                raise ValueError(f"Invalid max rounds: {max_rounds}")
            
            while self._state[session_id]["round"] < max_rounds:
                # Collect votes
                votes = await self._collect_votes(context, session_id, timeout)
                self._state[session_id]["votes"] = votes
                
                # Check consensus
                consensus, decision = self._check_consensus(votes, threshold)
                if consensus:
                    self.collaboration_metrics.consensus_reached = True
                    return {
                        "consensus": True,
                        "decision": decision,
                        "rounds": self._state[session_id]["round"] + 1,
                        "vote_distribution": self._get_vote_distribution(votes),
                        "participants": list(votes.keys())
                    }
                
                self._state[session_id]["round"] += 1
                self.collaboration_metrics.round_trips += 1
            
            # No consensus reached
            final_votes = self._state[session_id]["votes"]
            return {
                "consensus": False,
                "rounds": max_rounds,
                "vote_distribution": self._get_vote_distribution(final_votes),
                "participants": list(final_votes.keys())
            }
            
        except Exception as e:
            logger.error(f"Consensus failed: {str(e)}")
            raise
    
    async def _collect_votes(
        self,
        context: ExecutionContext,
        session_id: str,
        timeout: float
    ) -> Dict[str, Dict[str, Any]]:
        """Collect votes from participants."""
        votes: Dict[str, Dict[str, Any]] = {}
        try:
            async with asyncio.timeout(timeout):
                for participant in self._state[session_id]["participants"]:
                    vote = await self._get_vote(
                        participant,
                        context.params["message"],
                        self._state[session_id]["vote_options"]
                    )
                    votes[participant] = vote
                    self.collaboration_metrics.messages_received += 1
                    self.collaboration_metrics.participants.add(participant)
        except asyncio.TimeoutError:
            logger.warning("Vote collection timed out")
        return votes
    
    async def _get_vote(
        self,
        participant: str,
        message: Dict[str, Any],
        vote_options: List[str]
    ) -> Dict[str, Any]:
        """Get vote from participant."""
        self.collaboration_metrics.messages_sent += 1
        
        # Validate vote against options
        vote = message.get("vote")
        if vote and vote in vote_options:
            return {"vote": vote}
        return {"vote": "abstain"}
    
    def _check_consensus(
        self,
        votes: Dict[str, Dict[str, Any]],
        threshold: float
    ) -> tuple[bool, Optional[str]]:
        """Check if consensus is reached."""
        if not votes:
            return False, None
            
        # Count votes
        vote_counts: Dict[str, int] = {}
        for vote_data in votes.values():
            vote = vote_data.get("vote", "abstain")
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Check threshold
        total_votes = len(votes)
        for vote, count in vote_counts.items():
            if vote != "abstain" and count / total_votes >= threshold:
                return True, vote
        
        return False, None
    
    def _get_vote_distribution(self, votes: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Get distribution of votes."""
        if not votes:
            return {}
            
        vote_counts: Dict[str, int] = {}
        total_votes = len(votes)
        
        for vote_data in votes.values():
            vote = vote_data.get("vote", "abstain")
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        return {
            vote: count / total_votes
            for vote, count in vote_counts.items()
        }

class NegotiationInstruction(CollaborativeInstruction):
    """Negotiate agreement among participants."""
    
    def __init__(
        self,
        name: str = "negotiation",
        description: str = "Negotiate agreement among participants",
        metadata: Optional[InstructionMetadata] = None,
        config: Optional[CollaborationConfig] = None
    ):
        """Initialize negotiation instruction."""
        super().__init__(
            name=name,
            description=description,
            instruction_type=InstructionType.COLLABORATION,
            metadata=metadata or InstructionMetadata(
                capabilities={"collaboration", "negotiation"},
                tags={"negotiation", "collaborative", "agreement"},
                cleanup_required=True,
                resource_requirements={"network": 1.0, "memory": 768}
            ),
            config=config or CollaborationConfig(
                required_capabilities={"negotiation"},
                min_collaborators=2,
                max_collaborators=10,
                timeout_seconds=600.0
            )
        )
        
        # Add specific parameters
        self.add_required_param("max_offers", int)
        self.add_required_param("acceptance_threshold", float)
        self.add_required_param("negotiation_topics", List[str])
        self.add_required_param("value_ranges", Dict[str, Dict[str, Any]])
    
    async def _start_session(self, context: ExecutionContext) -> str:
        """Start negotiation session."""
        session_id = f"negotiation_{uuid.uuid4()}"
        self._state[session_id] = {
            "round": 0,
            "offers": [],
            "participants": cast(List[str], context.params["participants"]),
            "start_time": time.time(),
            "topics": context.params["negotiation_topics"],
            "value_ranges": context.params["value_ranges"]
        }
        return session_id
    
    async def _end_session(self, session_id: str) -> None:
        """End negotiation session."""
        if session_id in self._state:
            # Clean up any temporary negotiation records
            session_data = self._state[session_id]
            if "temp_files" in session_data:
                for file_path in session_data["temp_files"]:
                    try:
                        path = Path(file_path)
                        if path.exists():
                            path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {file_path}: {e}")
            
            del self._state[session_id]
    
    async def _collaborate(self, context: ExecutionContext, session_id: str) -> Dict[str, Any]:
        """Execute negotiation logic."""
        try:
            max_offers = cast(int, context.params["max_offers"])
            threshold = cast(float, context.params["acceptance_threshold"])
            timeout = cast(float, context.params["timeout"])
            
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Invalid acceptance threshold: {threshold}")
            
            if max_offers < 1:
                raise ValueError(f"Invalid max offers: {max_offers}")
            
            while len(self._state[session_id]["offers"]) < max_offers:
                # Get offers
                offers = await self._collect_offers(context, session_id, timeout)
                self._state[session_id]["offers"].extend(offers)
                
                # Check agreement
                agreement = self._check_agreement(offers, threshold)
                if agreement:
                    return {
                        "agreement": True,
                        "final_offer": agreement,
                        "rounds": len(self._state[session_id]["offers"]),
                        "offer_history": self._get_offer_history(self._state[session_id]["offers"]),
                        "participants": list(set(o["participant"] for o in offers))
                    }
                
                self.collaboration_metrics.round_trips += 1
            
            # No agreement reached
            return {
                "agreement": False,
                "offer_history": self._get_offer_history(self._state[session_id]["offers"]),
                "rounds": max_offers,
                "final_state": self._get_negotiation_state(self._state[session_id]["offers"])
            }
            
        except Exception as e:
            logger.error(f"Negotiation failed: {str(e)}")
            raise
    
    async def _collect_offers(
        self,
        context: ExecutionContext,
        session_id: str,
        timeout: float
    ) -> List[Dict[str, Any]]:
        """Collect offers from participants."""
        offers: List[Dict[str, Any]] = []
        try:
            async with asyncio.timeout(timeout):
                for participant in self._state[session_id]["participants"]:
                    offer = await self._get_offer(
                        participant,
                        context.params["message"],
                        self._state[session_id]["topics"],
                        self._state[session_id]["value_ranges"]
                    )
                    if self._validate_offer(offer, self._state[session_id]["value_ranges"]):
                        offers.append({
                            "participant": participant,
                            "offer": offer,
                            "timestamp": time.time()
                        })
                        self.collaboration_metrics.messages_received += 1
                        self.collaboration_metrics.participants.add(participant)
        except asyncio.TimeoutError:
            logger.warning("Offer collection timed out")
        return offers
    
    async def _get_offer(
        self,
        participant: str,
        message: Dict[str, Any],
        topics: List[str],
        value_ranges: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get offer from participant."""
        self.collaboration_metrics.messages_sent += 1
        
        # Extract and validate offer
        offer = message.get("offer", {})
        return {
            topic: offer.get(topic)
            for topic in topics
            if topic in offer and self._validate_value(
                offer[topic],
                value_ranges.get(topic, {})
            )
        }
    
    def _validate_offer(
        self,
        offer: Dict[str, Any],
        value_ranges: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Validate offer against value ranges."""
        try:
            return all(
                self._validate_value(value, value_ranges.get(topic, {}))
                for topic, value in offer.items()
            )
        except Exception:
            return False
    
    def _validate_value(self, value: Any, range_spec: Dict[str, Any]) -> bool:
        """Validate a single value against its range specification."""
        try:
            if "type" in range_spec:
                if not isinstance(value, eval(range_spec["type"])):
                    return False
            if "min" in range_spec and value < range_spec["min"]:
                return False
            if "max" in range_spec and value > range_spec["max"]:
                return False
            if "allowed_values" in range_spec and value not in range_spec["allowed_values"]:
                return False
            return True
        except Exception:
            return False
    
    def _check_agreement(
        self,
        offers: List[Dict[str, Any]],
        threshold: float
    ) -> Optional[Dict[str, Any]]:
        """Check if agreement is reached."""
        if not offers:
            return None
        
        # Extract actual offers
        offer_values = [o["offer"] for o in offers]
        
        # Find common elements in offers
        common: Dict[str, Dict[str, int]] = {}
        for offer in offer_values:
            for topic, value in offer.items():
                if topic not in common:
                    common[topic] = {}
                value_str = json.dumps(value, sort_keys=True)
                common[topic][value_str] = common[topic].get(value_str, 0) + 1
        
        # Check threshold
        agreement: Dict[str, Any] = {}
        total_offers = len(offers)
        for topic, values in common.items():
            for value_str, count in values.items():
                if count / total_offers >= threshold:
                    agreement[topic] = json.loads(value_str)
        
        return agreement if agreement else None
    
    def _get_offer_history(self, offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get formatted offer history."""
        return [
            {
                "round": i + 1,
                "participant": offer["participant"],
                "offer": offer["offer"],
                "timestamp": offer["timestamp"]
            }
            for i, offer in enumerate(offers)
        ]
    
    def _get_negotiation_state(self, offers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get current negotiation state."""
        if not offers:
            return {}
        
        # Get latest offers per participant
        latest_offers: Dict[str, Dict[str, Any]] = {}
        for offer in reversed(offers):
            participant = offer["participant"]
            if participant not in latest_offers:
                latest_offers[participant] = offer["offer"]
        
        # Calculate average values for numeric fields
        averages: Dict[str, float] = {}
        for topic in self._state[list(self._state.keys())[0]]["topics"]:
            values = []
            for offer in latest_offers.values():
                if topic in offer and isinstance(offer[topic], (int, float)):
                    values.append(offer[topic])
            if values:
                averages[topic] = sum(values) / len(values)
        
        return {
            "latest_offers": latest_offers,
            "averages": averages,
            "participants": list(latest_offers.keys())
        } 