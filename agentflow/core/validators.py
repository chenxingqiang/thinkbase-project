"""Validators for objective success criteria."""
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from typing import List, Tuple
import json

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class BaseValidator(ABC):
    """Base class for all validators."""
    
    @abstractmethod
    def validate(self, data: Any, criteria: Dict[str, Any]) -> ValidationResult:
        """Validate data against criteria.
        
        Args:
            data: Data to validate
            criteria: Validation criteria
            
        Returns:
            ValidationResult
        """
        pass

class SchemaValidator(BaseValidator):
    """Validates data against a schema."""
    
    def validate(self, data: Any, criteria: Dict[str, Any]) -> ValidationResult:
        try:
            schema = criteria.get("schema", {})
            if not schema:
                return ValidationResult(
                    is_valid=False,
                    message="No schema provided for validation"
                )
                
            validation_errors = []
            
            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    validation_errors.append(f"Missing required field: {field}")
                    
            # Check field types
            properties = schema.get("properties", {})
            for field, field_schema in properties.items():
                if field in data:
                    if not self._validate_type(data[field], field_schema.get("type")):
                        validation_errors.append(
                            f"Invalid type for field {field}: "
                            f"expected {field_schema.get('type')}"
                        )
                        
            # Check patterns
            for field, field_schema in properties.items():
                if field in data and "pattern" in field_schema:
                    if not re.match(field_schema["pattern"], str(data[field])):
                        validation_errors.append(
                            f"Field {field} does not match pattern: "
                            f"{field_schema['pattern']}"
                        )
                        
            is_valid = len(validation_errors) == 0
            return ValidationResult(
                is_valid=is_valid,
                score=1.0 if is_valid else 0.0,
                details={"errors": validation_errors} if validation_errors else None,
                message="Schema validation " + ("passed" if is_valid else "failed")
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Schema validation error: {str(e)}"
            )
            
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value against expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        return isinstance(value, type_map.get(expected_type, object))

class MetricValidator(BaseValidator):
    """Validates metrics against thresholds."""
    
    def validate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> ValidationResult:
        try:
            metric_name = criteria.get("metric_name")
            threshold = criteria.get("threshold", 0.0)
            
            if not metric_name or "value" not in data:
                return ValidationResult(
                    is_valid=False,
                    message="Missing metric name or value"
                )
                
            value = data["value"]
            is_valid = value >= threshold
            
            return ValidationResult(
                is_valid=is_valid,
                score=value,
                details={
                    "metric_name": metric_name,
                    "threshold": threshold,
                    "actual_value": value
                },
                message=f"Metric {metric_name} validation "
                       f"{'passed' if is_valid else 'failed'}"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Metric validation error: {str(e)}"
            )

class ModelPerformanceValidator(BaseValidator):
    """Validates machine learning model performance."""
    
    def validate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> ValidationResult:
        try:
            metric_name = criteria.get("metric_name", "accuracy")
            threshold = criteria.get("threshold", 0.0)
            
            y_true = data.get("y_true")
            y_pred = data.get("y_pred")
            
            if y_true is None or y_pred is None:
                return ValidationResult(
                    is_valid=False,
                    message="Missing true or predicted values"
                )
                
            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Calculate metric
            metric_functions = {
                "accuracy": accuracy_score,
                "f1": f1_score,
                "precision": precision_score,
                "recall": recall_score
            }
            
            if metric_name not in metric_functions:
                return ValidationResult(
                    is_valid=False,
                    message=f"Unsupported metric: {metric_name}"
                )
                
            score = metric_functions[metric_name](y_true, y_pred)
            is_valid = score >= threshold
            
            return ValidationResult(
                is_valid=is_valid,
                score=score,
                details={
                    "metric_name": metric_name,
                    "threshold": threshold,
                    "actual_score": score
                },
                message=f"Model performance validation "
                       f"{'passed' if is_valid else 'failed'}"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Model performance validation error: {str(e)}"
            )

class ContentValidator(BaseValidator):
    """Validates content against requirements."""
    
    def validate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> ValidationResult:
        try:
            content = data.get("content", "")
            required_elements = criteria.get("required_elements", [])
            min_length = criteria.get("min_length", 0)
            max_length = criteria.get("max_length", float("inf"))
            
            validation_errors = []
            
            # Check length
            content_length = len(content)
            if content_length < min_length:
                validation_errors.append(
                    f"Content length {content_length} is below minimum {min_length}"
                )
            if content_length > max_length:
                validation_errors.append(
                    f"Content length {content_length} exceeds maximum {max_length}"
                )
                
            # Check required elements
            for element in required_elements:
                if element not in content:
                    validation_errors.append(f"Missing required element: {element}")
                    
            is_valid = len(validation_errors) == 0
            return ValidationResult(
                is_valid=is_valid,
                score=1.0 if is_valid else 0.0,
                details={
                    "errors": validation_errors,
                    "content_length": content_length
                } if validation_errors else {"content_length": content_length},
                message="Content validation " + ("passed" if is_valid else "failed")
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Content validation error: {str(e)}"
            )

class PatternValidator(BaseValidator):
    """Validates data against patterns."""
    
    def validate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> ValidationResult:
        try:
            patterns = criteria.get("patterns", [])
            if not patterns:
                return ValidationResult(
                    is_valid=False,
                    message="No patterns provided for validation"
                )
                
            validation_results = []
            for pattern in patterns:
                pattern_type = pattern.get("type")
                pattern_value = pattern.get("value")
                
                if pattern_type == "regex":
                    matches = re.findall(pattern_value, str(data))
                    validation_results.append({
                        "pattern": pattern_value,
                        "matches": len(matches),
                        "matched_values": matches
                    })
                elif pattern_type == "keyword":
                    count = str(data).lower().count(pattern_value.lower())
                    validation_results.append({
                        "pattern": pattern_value,
                        "matches": count
                    })
                    
            # Consider valid if at least one pattern matched
            total_matches = sum(r["matches"] for r in validation_results)
            is_valid = total_matches > 0
            
            return ValidationResult(
                is_valid=is_valid,
                score=float(total_matches > 0),
                details={"pattern_matches": validation_results},
                message=f"Pattern validation {'passed' if is_valid else 'failed'} "
                       f"with {total_matches} total matches"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Pattern validation error: {str(e)}"
            )

class ResponseQualityValidator(BaseValidator):
    """Validates the quality of agent responses."""
    
    def validate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> ValidationResult:
        try:
            response_text = data.get("response", "")
            if not response_text:
                return ValidationResult(
                    is_valid=False,
                    message="No response text provided"
                )
                
            # Initialize NLTK components
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('punkt')
                nltk.download('vader_lexicon')
                
            results = {}
            
            # 1. Length checks
            min_length = criteria.get("min_length", 0)
            max_length = criteria.get("max_length", float("inf"))
            word_count = len(word_tokenize(response_text))
            results["length"] = {
                "word_count": word_count,
                "within_limits": min_length <= word_count <= max_length
            }
            
            # 2. Sentiment analysis
            if criteria.get("check_sentiment", False):
                sia = SentimentIntensityAnalyzer()
                sentiment_scores = sia.polarity_scores(response_text)
                target_sentiment = criteria.get("target_sentiment", "neutral")
                
                sentiment_map = {
                    "positive": sentiment_scores["pos"],
                    "negative": sentiment_scores["neg"],
                    "neutral": sentiment_scores["neu"]
                }
                
                results["sentiment"] = {
                    "scores": sentiment_scores,
                    "matches_target": sentiment_map.get(target_sentiment, 0) > 0.5
                }
                
            # 3. Coherence check
            sentences = sent_tokenize(response_text)
            if len(sentences) > 1:
                vectorizer = TfidfVectorizer()
                try:
                    tfidf_matrix = vectorizer.fit_transform(sentences)
                    tfidf_array = np.asarray(tfidf_matrix.todense())  # type: ignore
                    coherence_scores = cosine_similarity(tfidf_array[:-1], tfidf_array[1:])
                    avg_coherence = coherence_scores.mean()
                    
                    min_coherence = criteria.get("min_coherence", 0.3)
                    results["coherence"] = {
                        "score": float(avg_coherence),
                        "sufficient": avg_coherence >= min_coherence
                    }
                except ValueError:
                    results["coherence"] = {
                        "error": "Could not compute coherence"
                    }
                    
            # 4. Grammar check
            if criteria.get("check_grammar", False):
                blob = TextBlob(response_text)
                results["grammar"] = {
                    "spelling_errors": len(blob.correct().split()) - len(response_text.split())
                }
                
            # Determine overall validity
            required_checks = criteria.get("required_checks", ["length"])
            is_valid = True
            
            for check in required_checks:
                if check == "length" and not results["length"]["within_limits"]:
                    is_valid = False
                elif check == "sentiment" and not results["sentiment"]["matches_target"]:
                    is_valid = False
                elif check == "coherence" and not results["coherence"].get("sufficient", False):
                    is_valid = False
                elif check == "grammar" and results["grammar"]["spelling_errors"] > criteria.get("max_spelling_errors", 0):
                    is_valid = False
                    
            return ValidationResult(
                is_valid=is_valid,
                score=1.0 if is_valid else 0.0,
                details=results,
                message="Response quality validation " + ("passed" if is_valid else "failed")
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Response quality validation error: {str(e)}"
            )

class ConversationFlowValidator(BaseValidator):
    """Validates the flow and consistency of a conversation."""
    
    def validate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> ValidationResult:
        try:
            conversation = data.get("messages", [])
            if not conversation:
                return ValidationResult(
                    is_valid=False,
                    message="No conversation messages provided"
                )
                
            results = {}
            
            # 1. Turn-taking pattern
            if criteria.get("check_turn_taking", True):
                roles = [msg.get("role") for msg in conversation]
                valid_pattern = all(
                    roles[i] != roles[i+1]
                    for i in range(len(roles)-1)
                )
                results["turn_taking"] = {
                    "valid": valid_pattern,
                    "pattern": roles
                }
                
            # 2. Context maintenance
            if criteria.get("check_context", True):
                vectorizer = TfidfVectorizer()
                messages = [msg.get("content", "") for msg in conversation]
                try:
                    tfidf_matrix = vectorizer.fit_transform(messages)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    
                    # Check similarity between consecutive messages
                    context_scores = []
                    for i in range(len(messages)-1):
                        score = similarity_matrix[i][i+1]
                        context_scores.append(float(score))
                        
                    min_context_score = criteria.get("min_context_score", 0.1)
                    results["context_maintenance"] = {
                        "scores": context_scores,
                        "sufficient": all(score >= min_context_score for score in context_scores)
                    }
                except ValueError:
                    results["context_maintenance"] = {
                        "error": "Could not compute context maintenance"
                    }
                    
            # 3. Response times
            if criteria.get("check_response_times", False):
                times = [
                    datetime.fromisoformat(msg.get("timestamp", ""))
                    for msg in conversation
                    if "timestamp" in msg
                ]
                if len(times) > 1:
                    delays = [
                        (times[i+1] - times[i]).total_seconds()
                        for i in range(len(times)-1)
                    ]
                    max_delay = criteria.get("max_response_time", 300)  # 5 minutes default
                    results["response_times"] = {
                        "delays": delays,
                        "within_limits": all(delay <= max_delay for delay in delays)
                    }
                    
            # 4. Topic consistency
            if criteria.get("check_topic", False):
                expected_topics = set(criteria.get("expected_topics", []))
                if expected_topics:
                    all_text = " ".join(msg.get("content", "") for msg in conversation)
                    found_topics = set()
                    for topic in expected_topics:
                        if topic.lower() in all_text.lower():
                            found_topics.add(topic)
                            
                    results["topic_consistency"] = {
                        "expected_topics": list(expected_topics),
                        "found_topics": list(found_topics),
                        "coverage": len(found_topics) / len(expected_topics)
                    }
                    
            # Determine overall validity
            required_checks = criteria.get("required_checks", ["turn_taking"])
            is_valid = True
            
            for check in required_checks:
                if check == "turn_taking" and not results["turn_taking"]["valid"]:
                    is_valid = False
                elif check == "context_maintenance" and not results["context_maintenance"].get("sufficient", False):
                    is_valid = False
                elif check == "response_times" and not results["response_times"]["within_limits"]:
                    is_valid = False
                elif check == "topic_consistency" and results["topic_consistency"]["coverage"] < criteria.get("min_topic_coverage", 0.8):
                    is_valid = False
                    
            return ValidationResult(
                is_valid=is_valid,
                score=1.0 if is_valid else 0.0,
                details=results,
                message="Conversation flow validation " + ("passed" if is_valid else "failed")
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Conversation flow validation error: {str(e)}"
            )

class APIResponseValidator(BaseValidator):
    """Validates responses from external APIs."""
    
    def validate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> ValidationResult:
        try:
            response_data = data.get("response", {})
            if not response_data:
                return ValidationResult(
                    is_valid=False,
                    message="No API response data provided"
                )
                
            results = {}
            
            # 1. Status code check
            if "status_code" in response_data:
                expected_codes = criteria.get("expected_status_codes", [200])
                results["status_code"] = {
                    "actual": response_data["status_code"],
                    "valid": response_data["status_code"] in expected_codes
                }
                
            # 2. Response time check
            if "response_time" in response_data:
                max_time = criteria.get("max_response_time", 5000)  # 5 seconds default
                results["response_time"] = {
                    "actual": response_data["response_time"],
                    "within_limit": response_data["response_time"] <= max_time
                }
                
            # 3. Schema validation
            if criteria.get("response_schema"):
                schema_validator = SchemaValidator()
                schema_result = schema_validator.validate(
                    response_data.get("body", {}),
                    {"schema": criteria["response_schema"]}
                )
                results["schema"] = {
                    "valid": schema_result.is_valid,
                    "details": schema_result.details
                }
                
            # 4. Rate limit check
            if "rate_limit" in response_data:
                min_remaining = criteria.get("min_rate_limit_remaining", 10)
                results["rate_limit"] = {
                    "remaining": response_data["rate_limit"],
                    "sufficient": response_data["rate_limit"] >= min_remaining
                }
                
            # 5. Error handling
            if "error" in response_data:
                allowed_errors = criteria.get("allowed_errors", [])
                results["error"] = {
                    "message": response_data["error"],
                    "acceptable": response_data["error"] in allowed_errors
                }
                
            # Determine overall validity
            required_checks = criteria.get("required_checks", ["status_code"])
            is_valid = True
            
            for check in required_checks:
                if check == "status_code" and not results["status_code"]["valid"]:
                    is_valid = False
                elif check == "response_time" and not results["response_time"]["within_limit"]:
                    is_valid = False
                elif check == "schema" and not results["schema"]["valid"]:
                    is_valid = False
                elif check == "rate_limit" and not results["rate_limit"]["sufficient"]:
                    is_valid = False
                elif check == "error" and not results["error"].get("acceptable", True):
                    is_valid = False
                    
            return ValidationResult(
                is_valid=is_valid,
                score=1.0 if is_valid else 0.0,
                details=results,
                message="API response validation " + ("passed" if is_valid else "failed")
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"API response validation error: {str(e)}"
            )

class TaskCompletionValidator(BaseValidator):
    """Validates task completion status and quality."""
    
    def validate(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> ValidationResult:
        try:
            task_data = data.get("task", {})
            if not task_data:
                return ValidationResult(
                    is_valid=False,
                    message="No task data provided"
                )
                
            results = {}
            
            # 1. Completion status
            required_outputs = set(criteria.get("required_outputs", []))
            actual_outputs = set(task_data.get("outputs", {}).keys())
            missing_outputs = required_outputs - actual_outputs
            
            results["completion"] = {
                "required_outputs": list(required_outputs),
                "actual_outputs": list(actual_outputs),
                "missing_outputs": list(missing_outputs),
                "complete": len(missing_outputs) == 0
            }
            
            # 2. Time constraints
            if "time_taken" in task_data:
                max_time = criteria.get("max_time", float("inf"))
                results["time"] = {
                    "actual": task_data["time_taken"],
                    "within_limit": task_data["time_taken"] <= max_time
                }
                
            # 3. Resource usage
            if "resources" in task_data:
                resources = task_data["resources"]
                resource_limits = criteria.get("resource_limits", {})
                resource_status = {}
                
                for resource, usage in resources.items():
                    if resource in resource_limits:
                        resource_status[resource] = {
                            "usage": usage,
                            "limit": resource_limits[resource],
                            "within_limit": usage <= resource_limits[resource]
                        }
                        
                results["resources"] = {
                    "status": resource_status,
                    "all_within_limits": all(
                        status["within_limit"]
                        for status in resource_status.values()
                    )
                }
                
            # 4. Quality metrics
            if "quality_metrics" in task_data:
                metrics = task_data["quality_metrics"]
                thresholds = criteria.get("quality_thresholds", {})
                metric_status = {}
                
                for metric, value in metrics.items():
                    if metric in thresholds:
                        metric_status[metric] = {
                            "value": value,
                            "threshold": thresholds[metric],
                            "meets_threshold": value >= thresholds[metric]
                        }
                        
                results["quality"] = {
                    "metrics": metric_status,
                    "all_meet_thresholds": all(
                        status["meets_threshold"]
                        for status in metric_status.values()
                    )
                }
                
            # 5. Dependencies
            if "dependencies" in task_data:
                required_deps = set(criteria.get("required_dependencies", []))
                completed_deps = set(
                    dep for dep, status in task_data["dependencies"].items()
                    if status.get("status") == "completed"
                )
                missing_deps = required_deps - completed_deps
                
                results["dependencies"] = {
                    "required": list(required_deps),
                    "completed": list(completed_deps),
                    "missing": list(missing_deps),
                    "all_satisfied": len(missing_deps) == 0
                }
                
            # Determine overall validity
            required_checks = criteria.get("required_checks", ["completion"])
            is_valid = True
            
            for check in required_checks:
                if check == "completion" and not results["completion"]["complete"]:
                    is_valid = False
                elif check == "time" and not results["time"]["within_limit"]:
                    is_valid = False
                elif check == "resources" and not results["resources"]["all_within_limits"]:
                    is_valid = False
                elif check == "quality" and not results["quality"]["all_meet_thresholds"]:
                    is_valid = False
                elif check == "dependencies" and not results["dependencies"]["all_satisfied"]:
                    is_valid = False
                    
            return ValidationResult(
                is_valid=is_valid,
                score=1.0 if is_valid else 0.0,
                details=results,
                message="Task completion validation " + ("passed" if is_valid else "failed")
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                message=f"Task completion validation error: {str(e)}"
            )

class ValidatorRegistry:
    """Registry for validators."""
    
    _validators: Dict[str, Type[BaseValidator]] = {
        "schema": SchemaValidator,
        "metric": MetricValidator,
        "model_performance": ModelPerformanceValidator,
        "content": ContentValidator,
        "pattern": PatternValidator,
        "response_quality": ResponseQualityValidator,
        "conversation_flow": ConversationFlowValidator,
        "api_response": APIResponseValidator,
        "task_completion": TaskCompletionValidator
    }
    
    @classmethod
    def get_validator(cls, validator_type: str) -> Optional[BaseValidator]:
        """Get validator instance by type.
        
        Args:
            validator_type: Type of validator to get
            
        Returns:
            Validator instance if type exists, None otherwise
        """
        validator_class = cls._validators.get(validator_type)
        return validator_class() if validator_class else None
        
    @classmethod
    def register_validator(
        cls,
        validator_type: str,
        validator_class: Type[BaseValidator]
    ):
        """Register a new validator type.
        
        Args:
            validator_type: Type name for the validator
            validator_class: Validator class to register
        """
        cls._validators[validator_type] = validator_class
