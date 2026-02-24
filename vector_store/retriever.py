"""Retrieve strategy knowledge from the local vector store."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from schemas.llm_strategist import LLMInput
from trading_core.rule_registry import allowed_identifiers

from .embeddings import cosine_similarity, get_embedding

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    doc_id: str
    title: str
    content: str
    doc_type: str
    regimes: List[str]
    tags: List[str]
    identifiers: List[str]
    source_path: Path
    embedding: List[float]
    template_file: str | None = None


@dataclass
class RetrievalResult:
    """Result from retrieve_context — carries the knowledge block and an optional template id."""

    context: str | None
    template_id: str | None


_FRONTMATTER_BOUNDARY = "---"


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_BOUNDARY:
        return {}, text

    meta_lines: list[str] = []
    body_start = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == _FRONTMATTER_BOUNDARY:
            body_start = idx + 1
            break
        meta_lines.append(lines[idx])

    if body_start is None:
        return {}, text

    meta: dict[str, Any] = {}
    for raw in meta_lines:
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
            meta[key] = items
        else:
            meta[key] = value
    body = "\n".join(lines[body_start:]).lstrip()
    return meta, body


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def _title_from_filename(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in name.split())


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    trimmed = text[: max_chars - 1].rsplit(" ", 1)[0]
    return trimmed.rstrip() + "..."


def _doc_to_text(doc: VectorDocument) -> str:
    meta_parts = [doc.title, doc.doc_type]
    if doc.regimes:
        meta_parts.append("regime:" + ",".join(doc.regimes))
    if doc.tags:
        meta_parts.append("tags:" + ",".join(doc.tags))
    if doc.identifiers:
        meta_parts.append("identifiers:" + ",".join(doc.identifiers))
    return "\n".join([" ".join(meta_parts), doc.content])


class StrategyVectorStore:
    """Loads and retrieves strategy documents for the strategist prompt."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.documents: List[VectorDocument] = []
        self._load_documents()

    def _load_documents(self) -> None:
        docs: List[VectorDocument] = []
        strategies_dir = self.base_dir / "strategies"
        playbooks_dir = self.base_dir / "playbooks"
        for path in sorted(list(strategies_dir.glob("*.md")) + list(playbooks_dir.glob("*.md"))):
            try:
                raw = path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to read vector store doc %s: %s", path, exc)
                continue
            meta, body = _parse_frontmatter(raw)
            doc_type = str(meta.get("type") or path.parent.name).lower()
            if doc_type == "strategies":
                doc_type = "strategy"
            elif doc_type == "playbooks":
                doc_type = "playbook"
            elif doc_type.endswith("s"):
                doc_type = doc_type[:-1]
            title = str(meta.get("title") or _title_from_filename(path))
            regimes = _normalize_list(meta.get("regimes") or meta.get("regime"))
            tags = _normalize_list(meta.get("tags"))
            identifiers = _normalize_list(meta.get("identifiers") or meta.get("rules"))
            doc_id = str(meta.get("id") or path.stem)
            template_file = str(meta.get("template_file") or "").strip() or None
            placeholder = VectorDocument(
                doc_id=doc_id,
                title=title,
                content=body.strip(),
                doc_type=doc_type,
                regimes=regimes,
                tags=tags,
                identifiers=identifiers,
                source_path=path,
                embedding=[],
                template_file=template_file,
            )
            embedding = get_embedding(_doc_to_text(placeholder))
            docs.append(
                VectorDocument(
                    doc_id=doc_id,
                    title=title,
                    content=body.strip(),
                    doc_type=doc_type,
                    regimes=regimes,
                    tags=tags,
                    identifiers=identifiers,
                    source_path=path,
                    embedding=embedding,
                    template_file=template_file,
                )
            )
        self.documents = docs

    def _build_query(self, llm_input: LLMInput) -> str:
        context = llm_input.global_context or {}
        parts: List[str] = []
        regime = self._infer_regime(llm_input)
        if regime:
            parts.append(f"regime:{regime}")
        strategy_profile = context.get("strategy_profile") or context.get("strategy_guidance")
        if strategy_profile:
            parts.append(f"profile:{strategy_profile}")
        if llm_input.assets:
            asset = llm_input.assets[0]
            if asset.trend_state:
                parts.append(f"trend:{asset.trend_state}")
            if asset.vol_state:
                parts.append(f"vol:{asset.vol_state}")
            if asset.indicators:
                rsi = asset.indicators[0].rsi_14
                if rsi is not None:
                    if rsi < 30:
                        parts.append("rsi:oversold")
                    elif rsi > 70:
                        parts.append("rsi:overbought")
                    else:
                        parts.append("rsi:neutral")
        return " ".join(parts)

    def search(self, query: str, top_k: int = 5) -> List[tuple[VectorDocument, float]]:
        if not self.documents:
            return []
        query_embedding = get_embedding(query)
        results: List[tuple[VectorDocument, float]] = []
        for doc in self.documents:
            score = cosine_similarity(query_embedding, doc.embedding)
            results.append((doc, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def retrieve_context(
        self,
        llm_input: LLMInput,
        max_strategies: int = 2,
        max_playbooks: int = 2,
        max_chars_per_doc: int = 700,
    ) -> RetrievalResult:
        if not self.documents:
            return RetrievalResult(context=None, template_id=None)

        query = self._build_query(llm_input)
        results = self.search(query, top_k=8)
        if not results:
            return RetrievalResult(context=None, template_id=None)

        available_timeframes = (llm_input.global_context or {}).get("available_timeframes") or []
        allowed = allowed_identifiers(available_timeframes)

        regime = self._infer_regime(llm_input)

        strategies: List[VectorDocument] = []
        playbooks: List[VectorDocument] = []
        for doc, _score in results:
            if doc.doc_type == "strategy":
                if regime and doc.regimes and regime not in doc.regimes:
                    continue
                strategies.append(doc)
            elif doc.doc_type == "playbook":
                playbooks.append(doc)

        if not strategies:
            strategies = [doc for doc, _ in results if doc.doc_type == "strategy"]
        if not playbooks:
            playbooks = [doc for doc, _ in results if doc.doc_type == "playbook"]

        strategies = strategies[:max_strategies]
        playbooks = playbooks[:max_playbooks]

        sections: List[str] = []
        if strategies:
            sections.append("STRATEGY_KNOWLEDGE:\n" + self._format_docs(strategies, allowed, max_chars_per_doc))
        if playbooks:
            sections.append("RULE_PLAYBOOKS:\n" + self._format_docs(playbooks, allowed, max_chars_per_doc))
        top_strategy = strategies[0] if strategies else None
        template_id = top_strategy.template_file if top_strategy else None
        context_str = "\n\n".join(section for section in sections if section) or None
        return RetrievalResult(context=context_str, template_id=template_id)

    def _format_docs(self, docs: Iterable[VectorDocument], allowed: set[str], max_chars: int) -> str:
        formatted: List[str] = []
        for idx, doc in enumerate(docs, 1):
            content = _trim_text(doc.content.strip(), max_chars)
            identifiers = [ident for ident in doc.identifiers if ident in allowed]
            if identifiers:
                content = f"{content}\nIdentifiers: {', '.join(sorted(identifiers))}"
            formatted.append(f"{idx}. {doc.title}\n{content}")
        return "\n\n".join(formatted)

    def _infer_regime(self, llm_input: LLMInput) -> str | None:
        context = llm_input.global_context or {}
        regime = context.get("regime")
        if regime:
            return str(regime)
        for asset in llm_input.assets:
            if asset.regime_assessment and asset.regime_assessment.regime:
                return asset.regime_assessment.regime
        for asset in llm_input.assets:
            if asset.indicators:
                try:
                    from trading_core.regime_classifier import classify_regime

                    return classify_regime(asset.indicators[0]).regime
                except Exception:
                    continue
        return None


_store_instance: StrategyVectorStore | None = None


def get_strategy_vector_store(base_dir: Path | None = None) -> StrategyVectorStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = StrategyVectorStore(base_dir=base_dir)
    return _store_instance


def vector_store_enabled() -> bool:
    flag = os.environ.get("STRATEGY_VECTOR_STORE_ENABLED", "true").strip().lower()
    return flag not in {"0", "false", "no"}


__all__ = ["RetrievalResult", "StrategyVectorStore", "get_strategy_vector_store", "vector_store_enabled"]
