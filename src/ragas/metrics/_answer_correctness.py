from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import (
    FaithfulnessStatements,
    HasSegmentMethod,
    LongFormAnswerPrompt,
)
from ragas.metrics.base import (
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
    get_segmenter,
)
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.metrics._faithfulness import SentencesSimplified


logger = logging.getLogger(__name__)


class QuestionAnswerGroundTruth(BaseModel):
    question: str
    answer: list[str]
    ground_truth: list[str]


class StatementsWithReason(BaseModel):
    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    TP: list[StatementsWithReason]
    FP: list[StatementsWithReason]
    FN: list[StatementsWithReason]


class CorrectnessClassifier(
    PydanticPrompt[QuestionAnswerGroundTruth, ClassificationWithReason]
):
    instruction = "ground truthとanswer statementが与えられたとき、各statementを分析し、以下のカテゴリのいずれかに分類する： TP (true positive): 回答に存在し、ground truthの1つまたは複数のstatementによって直接サポートされているstatement、FP (false positive): 回答に存在するが、ground truthのstatementによって直接サポートされていないstatement、FN (false negative): ground truthに存在するが、回答には存在しないstatement。各statementは、いずれかのカテゴリにのみ属することができます。それぞれの分類の理由を書いてください。"
    input_model = QuestionAnswerGroundTruth
    output_model = ClassificationWithReason
    examples = [
        (
            QuestionAnswerGroundTruth(
                question="太陽には何があり、その主な機能は何なのか？",
                answer=[
                    "太陽は、地球の原子炉と同じように核分裂によって動いている。",
                    "太陽の主な役割は、太陽系に光を供給することである。",
                ],
                ground_truth=[
                    "太陽は核融合によって動いており、水素原子が融合してヘリウムを形成する。",
                    "この核融合プロセスは、太陽のコアで莫大なエネルギーを放出する。",
                    "太陽からのエネルギーは、地球上の生命にとって不可欠な熱と光を供給する。",
                    "太陽の光は地球の気候システムにおいて重要な役割を果たしている。",
                    "太陽の光は天候や海流を動かすのに役立っている。",
                ],
            ),
            ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="太陽の主な役割は、太陽系に光を供給することである。",
                        reason="この声明は、より広く太陽のエネルギーに焦点を当てているが、光を提供する太陽とその役割に言及した基本的な真実によっていくらか裏付けられている。",
                    )
                ],
                FP=[
                    StatementsWithReason(
                        statement="太陽は、地球の原子炉と同じように核分裂によって動いている。",
                        reason="この発言は間違っており、太陽が核融合によって動いているという基本的な真実と矛盾している。",
                    )
                ],
                FN=[
                    StatementsWithReason(
                        statement="太陽は核融合によって動いており、水素原子が融合してヘリウムを形成する。",
                        reason="太陽の動力源に関するこの正確な記述は、答えには含まれていない。",
                    ),
                    StatementsWithReason(
                        statement="この核融合プロセスは、太陽のコアで莫大なエネルギーを放出する。",
                        reason="このプロセスとその意義については、答案には書かれていない。",
                    ),
                    StatementsWithReason(
                        statement="太陽からのエネルギーは、地球上の生命にとって不可欠な熱と光を供給する。",
                        reason="答えは光にしか触れておらず、熱の本質的な側面や生命維持に必要なことは省かれている。",
                    ),
                    StatementsWithReason(
                        statement="太陽の光は地球の気候システムにおいて重要な役割を果たしている。",
                        reason="太陽の光が地球の気候システムに与えるこのような幅広い影響については、回答では触れられていない。",
                    ),
                    StatementsWithReason(
                        statement="太陽の光は天候や海流を動かすのに役立っている。",
                        reason="太陽光が気象パターンや海流に及ぼす影響については、答えの中に省略されている。",
                    ),
                ],
            ),
        ),
        (
            QuestionAnswerGroundTruth(
                question="水の沸点は何度ですか？",
                answer=[
                    "水の沸点は海面で摂氏100度です。"
                ],
                ground_truth=[
                    "水の沸点は海面温度で摂氏100度（華氏212度）です。",
                    "水の沸点は標高によって変化する。",
                ],
            ),
            ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="水の沸点は海面で摂氏100度です。",
                        reason="この発言は、水の沸点が海抜で摂氏100度であるという根拠によって直接的に裏付けられている。",
                    )
                ],
                FP=[],
                FN=[
                    StatementsWithReason(
                        statement="水の沸点は標高によって変化する。",
                        reason="水の沸点が高度によってどのように変化するかについてのこの追加情報は、答えには書かれていない。",
                    )
                ],
            ),
        ),
    ]


@dataclass
class AnswerCorrectness(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

    Attributes
    ----------
    name: string
        The name of the metrics
    weights:
        a list of two weights corresponding to factuality and semantic similarity
        Defaults [0.75, 0.25]
    answer_similarity:
        The AnswerSimilarity object
    """

    name: str = "answer_correctness"  # type: ignore[reportIncompatibleMethodOverride]
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
        }
    )
    correctness_prompt: PydanticPrompt = field(default_factory=CorrectnessClassifier)
    long_form_answer_prompt: PydanticPrompt = field(
        default_factory=LongFormAnswerPrompt
    )
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    answer_similarity: t.Optional[AnswerSimilarity] = None
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1

    def __post_init__(self: t.Self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

        if self.sentence_segmenter is None:
            language = self.long_form_answer_prompt.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def init(self, run_config: RunConfig):
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, embeddings=self.embeddings
            )

    def _compute_statement_presence(
        self, prediction: ClassificationWithReason
    ) -> float:
        tp = len(prediction.TP)
        fp = len(prediction.FP)
        fn = len(prediction.FN)
        score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0
        return score

    async def _create_simplified_statements(
        self, question: str, text: str, callbacks: Callbacks
    ) -> SentencesSimplified:
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"
        assert self.llm is not None, "llm is not set"

        sentences = self.sentence_segmenter.segment(text)
        sentences_with_index = {
            i: sentence
            for i, sentence in enumerate(sentences)
            if sentence.strip().endswith(".")
        }

        statements_simplified = await self.long_form_answer_prompt.generate(
            llm=self.llm,
            data=FaithfulnessStatements(
                question=question, answer=text, sentences=sentences_with_index
            ),
            callbacks=callbacks,
        )
        return statements_simplified

    async def _single_turn_ascore(
        self: t.Self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        score = await self._ascore(row, callbacks)
        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM must be set"

        # extract the statements from the answer and the ground truth
        question = row["user_input"]
        statements: t.Dict[str, t.List[str]] = {}
        for item in ["response", "reference"]:
            simplified_statements = await self._create_simplified_statements(
                question, row[item], callbacks
            )
            _statements_unwrapped = []
            for component in simplified_statements.sentences:
                _statements_unwrapped.extend(component.simpler_statements)
            statements[item] = _statements_unwrapped

        if not all([val == [] for val in statements.values()]):
            ground_truth = [statement for statement in statements["reference"]]
            answer = [statement for statement in statements["response"]]
            answers = await self.correctness_prompt.generate(
                llm=self.llm,
                data=QuestionAnswerGroundTruth(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                ),
                callbacks=callbacks,
            )
            if answers is None:
                return np.nan

            f1_score = self._compute_statement_presence(answers)
        else:
            f1_score = 1.0

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            similarity_score = await self.answer_similarity.ascore(
                row, callbacks=callbacks
            )

        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )

        return float(score)


answer_correctness = AnswerCorrectness()
