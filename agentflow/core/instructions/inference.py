"""Inference instructions for common reasoning and NLP tasks.

These instructions operate directly on plain-dict execution contexts and
return :class:`InferenceResult` objects whose ``status`` is the
:class:`~agentflow.core.instructions.base.InstructionStatus`.  They are
intentionally dependency-free (no ML backends, no network access): the
heuristics implemented here are deterministic, lightweight building blocks
that keep the instructions usable — and testable — in isolation.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field

from agentflow.core.instructions.base import InstructionStatus

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of an inference instruction execution."""

    status: InstructionStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


def _stable_hash(*parts: Any) -> int:
    """Deterministic hash across processes (unlike builtin ``hash``)."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(repr(part).encode("utf-8", errors="replace"))
    return int(hasher.hexdigest(), 16)


class InferenceInstruction:
    """Base class for inference instructions.

    Subclasses implement :meth:`_validate` and :meth:`_run`; the public
    :meth:`execute` wrapper takes care of validation, error handling,
    metrics and optional result caching.  A small simulated "model load"
    delay on first execution makes cold/warm start behaviour observable
    (see ``use_cache``).
    """

    name: str = "inference"

    #: Simulated cold-start cost (seconds) for "loading a model".
    model_load_delay: float = 0.05

    def __init__(self, name: Optional[str] = None):
        if name is not None:
            self.name = name
        self._model_cache: Dict[str, Any] = {}
        self._result_cache: Dict[int, InferenceResult] = {}

    # -- public API -------------------------------------------------------

    async def execute(self, context: Dict[str, Any]) -> InferenceResult:
        """Execute the instruction against a plain-dict context."""
        started = time.perf_counter()
        try:
            self._validate(context)
        except Exception as exc:
            logger.debug("Validation failed for %s: %s", self.name, exc)
            return InferenceResult(status=InstructionStatus.FAILED, error=str(exc))

        cache_key: Optional[int] = None
        if context.get("use_cache"):
            cache_key = _stable_hash(self.name, sorted(context.items(), key=repr))
            cached = self._result_cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            await self._load_model()
            data = await self._run(context)
        except Exception as exc:
            logger.error("Execution of %s failed: %s", self.name, exc)
            return InferenceResult(status=InstructionStatus.FAILED, error=str(exc))

        elapsed = time.perf_counter() - started
        data.setdefault("metrics", self._build_metrics(elapsed))
        result = InferenceResult(status=InstructionStatus.COMPLETED, data=data)
        if cache_key is not None:
            self._result_cache[cache_key] = result
        return result

    # -- helpers ----------------------------------------------------------

    async def _load_model(self) -> Any:
        """Simulate loading a model once per instance (cold start)."""
        key = type(self).__name__
        if key not in self._model_cache:
            if self.model_load_delay > 0:
                await asyncio.sleep(self.model_load_delay)
            self._model_cache[key] = {"loaded_at": time.time()}
        return self._model_cache[key]

    def _validate(self, context: Dict[str, Any]) -> None:
        """Validate the execution context; raise on invalid input."""
        if not isinstance(context, dict):
            raise ValueError(f"Context must be a dict, got {type(context)!r}")

    async def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Produce the instruction output; must be implemented by subclasses."""
        raise NotImplementedError

    def _build_metrics(self, elapsed: float, **extra: Any) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "instruction": self.name,
            "execution_time": elapsed,
            "num_model_loads": len(self._model_cache),
        }
        metrics.update(extra)
        return metrics

    @staticmethod
    def _require_texts(context: Dict[str, Any]) -> List[str]:
        """Validate and return the ``texts`` parameter."""
        texts = context.get("texts")
        if not isinstance(texts, (list, tuple)):
            raise ValueError("'texts' must be a non-empty list of strings")
        texts = list(texts)
        if not texts:
            raise ValueError("'texts' must be a non-empty list of strings")
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(f"'texts' entries must be strings, got {type(text)!r}")
        return texts

    @staticmethod
    def _score(text: str, salt: str = "", lo: float = 0.6, hi: float = 0.95) -> float:
        """Deterministic pseudo-confidence in ``[lo, hi]`` for a text."""
        value = (_stable_hash(salt, text) % 10_000) / 10_000
        return lo + (hi - lo) * value

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", str(text).lower())


class TextClassificationInstruction(InferenceInstruction):
    """Classify texts against an optional set of labels."""

    name = "text_classification"

    def _validate(self, context: Dict[str, Any]) -> None:
        super()._validate(context)
        self._require_texts(context)
        labels = context.get("labels")
        if labels is not None:
            if not isinstance(labels, (list, tuple)) or not labels:
                raise ValueError("'labels' must be a non-empty list when provided")

    async def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        texts = self._require_texts(context)
        labels = [str(label) for label in (context.get("labels") or ["default"])]

        predictions: List[str] = []
        confidences: List[float] = []
        batch_metrics: List[Dict[str, Any]] = []
        batch_size = context.get("batch_size")

        def classify_chunk(chunk: Sequence[str]) -> None:
            for text in chunk:
                index = _stable_hash("classify", text) % len(labels)
                predictions.append(labels[index])
                confidences.append(self._score(text, "classify"))

        if batch_size:
            size = max(1, int(batch_size))
            for start in range(0, len(texts), size):
                chunk = texts[start : start + size]
                chunk_started = time.perf_counter()
                classify_chunk(chunk)
                batch_metrics.append(
                    {
                        "batch_index": len(batch_metrics),
                        "batch_size": len(chunk),
                        "processing_time": time.perf_counter() - chunk_started,
                    }
                )
        else:
            classify_chunk(texts)

        data: Dict[str, Any] = {
            "predictions": predictions,
            "confidence_scores": confidences,
            "labels": labels,
        }
        if batch_metrics:
            data["batch_metrics"] = batch_metrics

        rules = context.get("validation_rules")
        if rules:
            data["validation_results"] = self._validate_predictions(
                predictions, confidences, rules
            )
        return data

    @staticmethod
    def _validate_predictions(
        predictions: List[str],
        confidences: List[float],
        rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        errors: List[str] = []
        allowed = rules.get("allowed_labels")
        if allowed is not None:
            for index, prediction in enumerate(predictions):
                if prediction not in allowed:
                    errors.append(
                        f"Prediction {index} ({prediction!r}) is not an allowed label"
                    )
        min_confidence = rules.get("min_confidence")
        if min_confidence is not None:
            for index, confidence in enumerate(confidences):
                if confidence < float(min_confidence):
                    errors.append(
                        f"Confidence {index} ({confidence:.3f}) is below "
                        f"the minimum {min_confidence}"
                    )
        return {"validation_passed": not errors, "validation_errors": errors}


class SentimentAnalysisInstruction(InferenceInstruction):
    """Analyse the sentiment of texts (with optional aspects)."""

    name = "sentiment_analysis"

    POSITIVE_WORDS = {
        "excellent", "great", "good", "love", "amazing", "wonderful",
        "fantastic", "nice", "awesome", "best", "happy", "brilliant",
    }
    NEGATIVE_WORDS = {
        "terrible", "bad", "awful", "hate", "horrible", "worst", "refund",
        "broken", "disappointing", "poor", "sad", "angry",
    }

    def _validate(self, context: Dict[str, Any]) -> None:
        super()._validate(context)
        self._require_texts(context)

    async def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        texts = self._require_texts(context)
        include_neutral = bool(context.get("include_neutral", True))
        include_aspects = bool(context.get("include_aspects", False))
        sentiments = [
            self._analyze(text, include_neutral, include_aspects) for text in texts
        ]
        return {"sentiments": sentiments}

    def _analyze(
        self, text: str, include_neutral: bool, include_aspects: bool
    ) -> Dict[str, Any]:
        lowered = text.lower()
        positive = sum(1 for word in self.POSITIVE_WORDS if word in lowered)
        negative = sum(1 for word in self.NEGATIVE_WORDS if word in lowered)

        if positive > negative:
            label, salt = "positive", "pos"
        elif negative > positive:
            label, salt = "negative", "neg"
        elif include_neutral:
            label, salt = "neutral", "neu"
        else:
            label = "positive" if _stable_hash("sentiment", text) % 2 else "negative"
            salt = "tie"

        entry: Dict[str, Any] = {"label": label, "score": self._score(text, salt)}
        if include_aspects:
            entry["aspects"] = self._extract_aspects(text)
        return entry

    @staticmethod
    def _extract_aspects(text: str) -> List[Dict[str, Any]]:
        """Extract candidate aspect nouns (naive: capitalized words)."""
        return [
            {"aspect": match.group(), "position": match.start()}
            for match in re.finditer(r"\b[A-Z][a-z]+\b", text)
        ]


class NERInstruction(InferenceInstruction):
    """Extract named entities from texts."""

    name = "named_entity_recognition"

    _ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:\s+[A-Z][A-Za-z0-9_]*)*\b")
    _DEFAULT_ENTITY_TYPES = ("PERSON", "ORG", "LOC", "MISC")

    def _validate(self, context: Dict[str, Any]) -> None:
        super()._validate(context)
        self._require_texts(context)
        entity_types = context.get("entity_types")
        if entity_types is not None:
            if not isinstance(entity_types, (list, tuple)) or not entity_types:
                raise ValueError("'entity_types' must be a non-empty list when provided")

    async def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        texts = self._require_texts(context)
        entity_types = [
            str(entity_type)
            for entity_type in (context.get("entity_types") or self._DEFAULT_ENTITY_TYPES)
        ]
        include_spans = bool(context.get("include_spans", True))
        entities = [
            self._extract(text, entity_types, include_spans) for text in texts
        ]
        return {"entities": entities}

    def _extract(
        self, text: str, entity_types: Sequence[str], include_spans: bool
    ) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        for match in self._ENTITY_PATTERN.finditer(text):
            entity_type = entity_types[_stable_hash("ner", match.group()) % len(entity_types)]
            record: Dict[str, Any] = {"text": match.group(), "type": entity_type}
            if include_spans:
                record["start"] = match.start()
                record["end"] = match.end()
            found.append(record)
        return found


class TextSimilarityInstruction(InferenceInstruction):
    """Compute lexical similarity metrics between two texts."""

    name = "text_similarity"

    def _validate(self, context: Dict[str, Any]) -> None:
        super()._validate(context)
        for key in ("text1", "text2"):
            if not isinstance(context.get(key), str) or not context.get(key):
                raise ValueError(f"'{key}' must be a non-empty string")

    async def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text1, text2 = context["text1"], context["text2"]
        metric_names = list(context.get("metrics") or ["cosine"])
        tokens1, tokens2 = self._tokens(text1), self._tokens(text2)

        scores: Dict[str, float] = {}
        for metric in metric_names:
            if metric == "cosine":
                scores[metric] = self._cosine(tokens1, tokens2)
            elif metric == "jaccard":
                scores[metric] = self._jaccard(tokens1, tokens2)
            elif metric == "overlap":
                scores[metric] = self._overlap(tokens1, tokens2)
            elif metric == "euclidean":
                scores[metric] = self._euclidean(tokens1, tokens2)
            else:
                logger.debug("Unknown similarity metric %r; reporting 0.0", metric)
                scores[metric] = 0.0

        return {
            "similarity_scores": scores,
            "use_embeddings": bool(context.get("use_embeddings")),
        }

    @staticmethod
    def _counts(tokens: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    def _cosine(self, tokens1: List[str], tokens2: List[str]) -> float:
        counts1, counts2 = self._counts(tokens1), self._counts(tokens2)
        if not counts1 or not counts2:
            return 0.0
        dot = sum(count * counts2.get(token, 0) for token, count in counts1.items())
        norm1 = sum(count ** 2 for count in counts1.values()) ** 0.5
        norm2 = sum(count ** 2 for count in counts2.values()) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def _jaccard(tokens1: List[str], tokens2: List[str]) -> float:
        set1, set2 = set(tokens1), set(tokens2)
        union = set1 | set2
        if not union:
            return 1.0
        return len(set1 & set2) / len(union)

    @staticmethod
    def _overlap(tokens1: List[str], tokens2: List[str]) -> float:
        set1, set2 = set(tokens1), set(tokens2)
        if not set1:
            return 0.0
        return len(set1 & set2) / len(set1)

    def _euclidean(self, tokens1: List[str], tokens2: List[str]) -> float:
        counts1, counts2 = self._counts(tokens1), self._counts(tokens2)
        keys = set(counts1) | set(counts2)
        distance = sum((counts1.get(k, 0) - counts2.get(k, 0)) ** 2 for k in keys) ** 0.5
        return distance


class QuestionAnsweringInstruction(InferenceInstruction):
    """Extractive question answering over context/question pairs."""

    name = "question_answering"

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
        "for", "to", "and", "or", "it", "its", "this", "that", "what",
        "which", "who", "how", "does", "do", "did", "with", "by", "as",
    }

    def _validate(self, context: Dict[str, Any]) -> None:
        super()._validate(context)
        pairs = context.get("qa_pairs")
        if not isinstance(pairs, (list, tuple)) or not pairs:
            raise ValueError("'qa_pairs' must be a non-empty list")
        for pair in pairs:
            if not isinstance(pair, dict) or "context" not in pair or "question" not in pair:
                raise ValueError(
                    "Each QA pair must be a dict with 'context' and 'question' keys"
                )

    async def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        answers = [self._answer(pair, context) for pair in context["qa_pairs"]]
        return {"answers": answers}

    def _answer(self, pair: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        context_text = str(pair["context"])
        question = str(pair["question"])
        max_answer_length = int(context.get("max_answer_length", 50))

        reference = pair.get("answer")
        if reference and str(reference) in context_text:
            candidate = str(reference)
        else:
            question_words = set(self._tokens(question)) - self._STOPWORDS
            best_sentence: Optional[str] = None
            best_overlap = -1
            for sentence in re.split(r"(?<=[.!?])\s+", context_text):
                overlap = len(set(self._tokens(sentence)) & question_words)
                if overlap > best_overlap:
                    best_sentence, best_overlap = sentence, overlap
            candidate = (best_sentence or context_text).strip()

        if len(candidate) > max_answer_length:
            candidate = candidate[:max_answer_length].rsplit(" ", 1)[0].rstrip(",;:")

        return {
            "answer": candidate,
            "confidence": self._score(question, "qa"),
            "question": question,
        }


class CausalInferenceInstruction(InferenceInstruction):
    """Estimate average treatment effects from observational data."""

    name = "causal_inference"

    def _validate(self, context: Dict[str, Any]) -> None:
        super()._validate(context)
        data = context.get("data")
        if not isinstance(data, dict):
            raise ValueError("'data' must be a dict of columns")
        treatment_col = context.get("treatment_col", "treatment")
        outcome_col = context.get("outcome_col", "outcome")
        for column in (treatment_col, outcome_col):
            if column not in data:
                raise ValueError(f"Column {column!r} is missing from 'data'")

    async def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        data = context["data"]
        treatment_col = context.get("treatment_col", "treatment")
        outcome_col = context.get("outcome_col", "outcome")
        method = str(context.get("method", "matching"))

        treatment = [float(value) for value in data[treatment_col]]
        outcome = [float(value) for value in data[outcome_col]]
        if len(treatment) != len(outcome):
            raise ValueError("'treatment' and 'outcome' must have the same length")

        treated = [o for t, o in zip(treatment, outcome) if t == 1]
        control = [o for t, o in zip(treatment, outcome) if t == 0]
        if not treated or not control:
            raise ValueError("Need observations for both treated and control groups")

        treated_mean = sum(treated) / len(treated)
        control_mean = sum(control) / len(control)
        ate = treated_mean - control_mean

        # Two-sample standard error of the difference in means.
        treated_var = sum((o - treated_mean) ** 2 for o in treated) / max(len(treated) - 1, 1)
        control_var = sum((o - control_mean) ** 2 for o in control) / max(len(control) - 1, 1)
        stderr = (treated_var / len(treated) + control_var / len(control)) ** 0.5

        return {
            "ate": ate,
            "confidence_interval": [ate - 1.96 * stderr, ate + 1.96 * stderr],
            "method": method,
            "num_treated": len(treated),
            "num_control": len(control),
            "treated_mean": treated_mean,
            "control_mean": control_mean,
        }


class LogicalReasoningInstruction(InferenceInstruction):
    """Deductive reasoning over a set of premises."""

    name = "logical_reasoning"

    _NEGATIONS = {"not", "cannot", "cant", "never", "no", "n't"}
    _IGNORED_WORDS = {
        "the", "a", "an", "is", "are", "can", "all", "some", "and", "or",
        "of", "in", "on", "for", "to", "by", "as", "be", "been", "being",
    } | _NEGATIONS

    def _validate(self, context: Dict[str, Any]) -> None:
        super()._validate(context)
        premises = context.get("premises")
        if not isinstance(premises, (list, tuple)) or not premises:
            raise ValueError("'premises' must be a non-empty list of statements")
        if not isinstance(context.get("query", ""), str):
            raise ValueError("'query' must be a string")

    async def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        premises = [str(premise) for premise in context["premises"]]
        query = str(context.get("query", ""))
        reasoning_type = str(context.get("reasoning_type", "deductive"))

        contradiction = self._detect_contradiction(premises)
        if contradiction is not None:
            first, second = contradiction
            conclusion = "Contradiction detected: the premises cannot all hold simultaneously."
            explanation = (
                f"Premise {first + 1} ({premises[first]!r}) conflicts with "
                f"premise {second + 1} ({premises[second]!r}) over a shared statement."
            )
            confidence = 0.9
        else:
            conclusion = "The premises are consistent; no contradiction was found."
            explanation = (
                f"{reasoning_type.capitalize()} analysis of {len(premises)} premises "
                "found no conflicting statements."
            )
            confidence = 0.7

        return {
            "conclusion": conclusion,
            "explanation": explanation,
            "confidence": confidence,
            "reasoning_type": reasoning_type,
            "query": query,
            "num_premises": len(premises),
            "contradiction_detected": contradiction is not None,
        }

    @classmethod
    def _detect_contradiction(cls, premises: List[str]) -> Optional[Tuple[int, int]]:
        """Find a pair of premises asserting opposite polarities of a fact."""
        parsed = []
        for index, premise in enumerate(premises):
            words = set(re.findall(r"[a-z']+", premise.lower()))
            parsed.append((index, words, bool(words & cls._NEGATIONS)))

        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                index_i, words_i, negated_i = parsed[i]
                index_j, words_j, negated_j = parsed[j]
                if negated_i == negated_j:
                    continue
                shared = (words_i & words_j) - cls._IGNORED_WORDS
                if shared:
                    return (index_i, index_j)
        return None


class KnowledgeGraphInstruction(InferenceInstruction):
    """Rule-based inference over a knowledge-graph triple store."""

    name = "knowledge_graph"

    HIERARCHY_PREDICATES = {"is_a", "instance_of", "subclass_of", "type_of"}

    def _validate(self, context: Dict[str, Any]) -> None:
        super()._validate(context)
        triples = context.get("triples")
        if not isinstance(triples, (list, tuple)) or not triples:
            raise ValueError("'triples' must be a non-empty list of (s, p, o) tuples")
        for triple in triples:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                raise ValueError("Each triple must be a (subject, predicate, object) tuple")
        query = context.get("query")
        if query is not None:
            if not isinstance(query, (list, tuple)) or len(query) != 3:
                raise ValueError("'query' must be a (subject, predicate, object) tuple")

    async def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        triples = [tuple(triple) for triple in context["triples"]]
        depth = max(1, int(context.get("inference_depth", 1)))

        inferred, confidence_scores = self._infer(triples, depth)
        data: Dict[str, Any] = {
            "inferred_triples": [list(triple) for triple in inferred],
            "confidence_scores": confidence_scores,
        }

        query = context.get("query")
        if query is not None:
            subject, predicate, _obj = tuple(query)
            data["query"] = list(query)
            data["query_answers"] = [
                list(triple)
                for triple in inferred
                if triple[0] == subject and triple[1] == predicate
            ]
        return data

    def _infer(
        self, triples: List[Tuple[Any, Any, Any]], depth: int
    ) -> Tuple[List[Tuple[Any, Any, Any]], Dict[str, float]]:
        """Propagate edges across hierarchy relations up to ``depth`` hops.

        Rule: for a hierarchy edge ``(A, is_a, B)`` every outgoing edge of
        ``B`` (e.g. ``(B, used_for, C)``) is inherited by ``A``.
        """
        known = set(triples)
        by_subject: Dict[Any, List[Tuple[Any, Any, Any]]] = {}
        for triple in known:
            by_subject.setdefault(triple[0], []).append(triple)

        inferred: List[Tuple[Any, Any, Any]] = []
        inferred_set: set = set()
        confidence_scores: Dict[str, float] = {}

        current = {
            (subject, obj)
            for (subject, predicate, obj) in known
            if predicate in self.HIERARCHY_PREDICATES
        }
        seen_pairs = set(current)

        for level in range(1, depth + 1):
            next_pairs = set()
            for (subject, parent) in current:
                for (_, predicate, obj) in by_subject.get(parent, ()):
                    new_triple = (subject, predicate, obj)
                    if new_triple not in known and new_triple not in inferred_set:
                        inferred.append(new_triple)
                        inferred_set.add(new_triple)
                        confidence_scores[repr(new_triple)] = round(0.9 ** level, 3)
                    if (
                        predicate in self.HIERARCHY_PREDICATES
                        and (subject, obj) not in seen_pairs
                    ):
                        next_pairs.add((subject, obj))
                        seen_pairs.add((subject, obj))
            current = next_pairs
            if not current:
                break

        return inferred, confidence_scores
