from __future__ import annotations

import typing as t
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ragas.llms.json_load import json_loader
from ragas.testset.prompts import keyphrase_extraction_prompt

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM
    from ragas.llms.prompt import Prompt
    from ragas.testset.docstore import Node


logger = logging.getLogger(__name__)

@dataclass
class Extractor(ABC):
    llm: BaseRagasLLM

    @abstractmethod
    def extract(self, node: Node) -> t.Any:
        ...

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the extractor to a different language.
        """
        raise NotImplementedError("adapt() is not implemented for {} Extractor")

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the extractor prompts to a path.
        """
        raise NotImplementedError("adapt() is not implemented for {} Extractor")


@dataclass
class keyphraseExtractor(Extractor):
    keyphrase_extraction_prompt: Prompt = field(
        default_factory=lambda: keyphrase_extraction_prompt
    )

    async def extract(self, node: Node) -> t.List[str]:
        prompt = keyphrase_extraction_prompt.format(text=node.page_content)
        results = await self.llm.agenerate_text(prompt=prompt)
        keyphrases = json_loader.sync_safe_load(
            results.generations[0][0].text.strip(), llm=self.llm
        )
        logger.debug("keyphrases: %s", keyphrases)
        return keyphrases.get("keyphrases", [])

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the extractor to a different language.
        """
        self.keyphrase_extraction_prompt = keyphrase_extraction_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the extractor prompts to a path.
        """
        self.keyphrase_extraction_prompt.save(cache_dir)
