from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from llama_index.readers.schema import Document as LlamaindexDocument

from ragas.embeddings import BaseRagasEmbeddings
from ragas.executor import Executor
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.testset.docstore import Document, DocumentStore, InMemoryDocumentStore
from ragas.testset.evolutions import ComplexEvolution, CurrentNodes, DataRow
from ragas.testset.filters import EvolutionFilter, NodeFilter, QuestionFilter

logger = logging.getLogger(__name__)
Distributions = t.Dict[t.Any, float]


@dataclass
class TestDataset:
    """
    TestDataset class
    """

    test_data: t.List[DataRow]

    def to_pandas(self) -> pd.DataFrame:
        data_samples = []
        for data in self.test_data:
            data_dict = dict(data)
            data_dict["episode_done"] = True
            data_samples.append(data_dict)

        return pd.DataFrame.from_records(data_samples)


@dataclass
class TestsetGenerator:
    generator_llm: BaseRagasLLM
    critic_llm: BaseRagasLLM
    embeddings: BaseRagasEmbeddings
    docstore: DocumentStore

    @classmethod
    def with_openai(
        cls,
        generator_llm: str = "gpt-3.5-turbo",
        critic_llm: str = "gpt-4",
        embeddings: str = "text-embedding-ada-002",
        docstore: t.Optional[DocumentStore] = None,
        chunk_size: int = 512,
    ) -> "TestsetGenerator":
        generator_llm_model = LangchainLLMWrapper(ChatOpenAI(model=generator_llm))
        critic_llm_model = LangchainLLMWrapper(ChatOpenAI(model=critic_llm))
        embeddings_model = OpenAIEmbeddings(model=embeddings)
        if docstore is None:
            from langchain.text_splitter import TokenTextSplitter

            splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
            docstore = InMemoryDocumentStore(
                splitter=splitter, embeddings=embeddings_model
            )
            return cls(
                generator_llm=generator_llm_model,
                critic_llm=critic_llm_model,
                embeddings=embeddings_model,
                docstore=docstore,
            )
        else:
            return cls(
                generator_llm=generator_llm_model,
                critic_llm=critic_llm_model,
                embeddings=embeddings_model,
                docstore=docstore,
            )

    def generate_with_llamaindex_docs(
        self,
        documents: t.Sequence[LlamaindexDocument],
        test_size: int,
        distributions: Distributions = {},
        show_debug_logs=False,
    ):
        # chunk documents and add to docstore
        self.docstore.add_documents(
            [Document.from_llamaindex_document(doc) for doc in documents]
        )

        return self.generate(
            test_size=test_size,
            distributions=distributions,
            show_debug_logs=show_debug_logs,
        )

    def generate(
        self, test_size: int, distributions: Distributions = {}, show_debug_logs=False
    ):
        # init filters and evolutions
        for evolution in distributions:
            if evolution.generator_llm is None:
                evolution.generator_llm = self.generator_llm
            if evolution.docstore is None:
                evolution.docstore = self.docstore

            if evolution.question_filter is None:
                evolution.question_filter = QuestionFilter(llm=self.critic_llm)
            if evolution.node_filter is None:
                evolution.node_filter = NodeFilter(llm=self.critic_llm)

            if isinstance(evolution, ComplexEvolution):
                evolution.init_evolution()
                if evolution.evolution_filter is None:
                    evolution.evolution_filter = EvolutionFilter(llm=self.critic_llm)
        if show_debug_logs:
            from ragas.utils import patch_logger

            patch_logger("ragas.testset.evolutions", logging.DEBUG)

        exec = Executor(
            desc="Generating",
            keep_progress_bar=True,
            raise_exceptions=True,
            is_async=True,
        )

        current_nodes = [
            CurrentNodes(root_node=n, nodes=[n])
            for n in self.docstore.get_random_nodes(k=test_size)
        ]
        for evolution, probability in distributions.items():
            for i in range(round(probability * test_size)):
                exec.submit(
                    evolution.aevolve,
                    current_nodes[i],
                    name=f"{evolution.__class__.__name__}-{i}",
                )

        try:
            test_data_rows = exec.results()
        except ValueError as e:
            raise e
        return TestDataset(test_data=test_data_rows)
