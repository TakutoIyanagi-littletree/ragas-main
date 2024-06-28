from __future__ import annotations

import typing as t

from langchain.chains.base import Chain
from langchain.schema import RUN_KEY
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import (
    EvaluationMode,
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
    get_required_columns,
)
from ragas.run_config import RunConfig
from ragas.validation import EVALMODE_TO_COLUMNS

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )


class EvaluatorChain(Chain, RunEvaluator):
    """
    Wrapper around ragas Metrics to use them with langsmith.
    """

    metric: Metric
    column_map: dict[str, str]

    def __init__(self, metric: Metric, **kwargs: t.Any):
        kwargs["metric"] = metric

        # chek if column_map is provided
        if "column_map" in kwargs:
            _column_map = kwargs["column_map"]
        else:
            _column_map = {}
        kwargs["column_map"] = _column_map

        super().__init__(**kwargs)

        # set up the run config
        if "run_config" in kwargs:
            run_config = kwargs["run_config"]
        else:
            run_config = RunConfig()

        if isinstance(self.metric, MetricWithLLM):
            llm = kwargs.get("llm", ChatOpenAI())
            t.cast(MetricWithLLM, self.metric).llm = LangchainLLMWrapper(llm)
        if isinstance(self.metric, MetricWithEmbeddings):
            embeddings = kwargs.get("embeddings", OpenAIEmbeddings())
            t.cast(
                MetricWithEmbeddings, self.metric
            ).embeddings = LangchainEmbeddingsWrapper(embeddings)
        self.metric.init(run_config)

    @property
    def input_keys(self) -> list[str]:
        return [
            self.column_map.get(column_name, column_name)
            for column_name in get_required_columns(self.metric.evaluation_mode)
        ]

    @property
    def output_keys(self) -> list[str]:
        return [self.metric.name]

    def _call(
        self,
        inputs: dict[str, t.Any],
        run_manager: t.Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, t.Any]:
        """
        Call the evaluation chain.
        """
        q, c, a, g = self._validate(inputs)
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        score = self.metric.score(
            {
                "question": q,
                "answer": a,
                "contexts": c,
                "ground_truth": g,
            },
            callbacks=callbacks,
        )
        return {self.metric.name: score}

    async def _acall(
        self,
        inputs: t.Dict[str, t.Any],
        run_manager: t.Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> t.Dict[str, t.Any]:
        """
        Call the evaluation chain.
        """
        q, c, a, g = self._validate(inputs)
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        # TODO: currently AsyncCallbacks are not supported in ragas
        _run_manager.get_child()

        score = await self.metric.ascore(
            {
                "question": q,
                "answer": a,
                "contexts": c,
                "ground_truth": g,
            },
            callbacks=[],
        )
        return {self.metric.name: score}

    def _validate(
        self,
        input: dict[str, t.Any],
    ) -> tuple[str, list[str], str, str]:
        # remap the keys
        question_key = self.column_map.get("question", "question")
        prediction_key = self.column_map.get("answer", "answer")
        context_key = self.column_map.get("contexts", "contexts")
        ground_truth_key = self.column_map.get("ground_truth", "ground_truth")

        # validate if the required columns are present
        required_columns = EVALMODE_TO_COLUMNS[self.metric.evaluation_mode]
        if "question" in required_columns and question_key not in input:
            raise ValueError(
                f'"{question_key}" is required in each example'
                f"for the metric[{self.metric.name}] you have chosen."
            )
        if "answer" in required_columns and prediction_key not in input:
            raise ValueError(
                f'"{prediction_key}" is required in each prediction'
                f"for the metric[{self.metric.name}] you have chosen."
            )
        if "contexts" in required_columns and context_key not in input:
            raise ValueError(
                f'"{context_key}" is required in each prediction for the '
                f"metric[{self.metric.name}] you have chosen."
            )
        if "ground_truth" in required_columns and ground_truth_key not in input:
            raise ValueError(
                f'"ground_truth" is required in each prediction for the '
                f"metric[{self.metric.name}] you have chosen."
            )

        # validate if the columns are of the correct datatype
        for column_name in [question_key, prediction_key, ground_truth_key]:
            if column_name in input:
                if not isinstance(input[column_name], str):
                    raise ValueError(
                        f'Input feature "{column_name}" should be of type string'
                    )

        for column_name in [context_key]:
            if column_name in input:
                if not (isinstance(input[column_name], list)):
                    raise ValueError(
                        f'Input feature "{column_name}" should be of type'
                        f" list[str], got {type(input[column_name])}"
                    )

        q = input.get(question_key, "")
        c = input.get(context_key, [""])
        a = input.get(prediction_key, "")
        g = input.get(ground_truth_key, "")
        return q, c, a, g

    @staticmethod
    def _keys_are_present(keys_to_check: list, dict_to_check: dict) -> list[str]:
        return [k for k in keys_to_check if k not in dict_to_check]

    def _validate_langsmith_eval(self, run: Run, example: t.Optional[Example]) -> None:
        # remap column names
        question_key = self.column_map.get("question", "question")
        ground_truth_key = self.column_map.get("ground_truth", "ground_truth")
        answer_key = self.column_map.get("answer", "answer")
        context_key = self.column_map.get("contexts", "contexts")

        if example is None:
            raise ValueError(
                "expected example to be provided. Please check langsmith dataset and ensure valid dataset is uploaded."
            )
        if example.inputs is None:
            raise ValueError(
                "expected example.inputs to be provided. Please check langsmith dataset and ensure valid dataset is uploaded."
            )
        if example.outputs is None:
            raise ValueError(
                "expected example.inputs to be provided. Please check langsmith dataset and ensure valid dataset is uploaded."
            )
        if (
            question_key not in example.inputs
            or ground_truth_key not in example.outputs
        ):
            raise ValueError(
                f"Expected '{question_key}' and {ground_truth_key} in example."
                f"Got: {[k for k in example.inputs.keys()]}"
            )
        assert (
            run.outputs is not None
        ), "the current run has no outputs. The chain should output 'answer' and 'contexts' keys."
        output_keys = get_required_columns(
            eval_mod=self.metric.evaluation_mode,
            ignore_columns=["question", "ground_truth"],
        )
        # remap output keys with column_map
        output_keys = [
            self.column_map.get(column_name, column_name) for column_name in output_keys
        ]
        # check for missing keys
        missing_keys = self._keys_are_present(output_keys, run.outputs)
        if missing_keys:
            raise ValueError(
                f"Expected {answer_key} and {context_key} in run.outputs."
                f"Got: {[k for k in run.outputs.keys()]}"
            )

    def evaluate_run(
        self, run: Run, example: t.Optional[Example] = None
    ) -> EvaluationResult:
        """
        Evaluate a langsmith run
        """
        self._validate_langsmith_eval(run, example)

        # this is just to suppress the type checker error
        # actual check and error message is in the _validate_langsmith_eval
        assert run.outputs is not None
        assert example is not None
        assert example.inputs is not None
        assert example.outputs is not None

        # remap column key
        ground_truth_key = self.column_map.get("ground_truth", "ground_truth")
        question_key = self.column_map.get("question", "question")

        chain_eval = run.outputs
        chain_eval[question_key] = example.inputs[question_key]
        if self.metric.evaluation_mode in [
            EvaluationMode.gc,
            EvaluationMode.ga,
            EvaluationMode.qcg,
            EvaluationMode.qga,
        ]:
            if example.outputs is None or ground_truth_key not in example.outputs:
                raise ValueError(f"expected `{ground_truth_key}` in example outputs.")
            chain_eval[ground_truth_key] = example.outputs[ground_truth_key]
        eval_output = self.invoke(chain_eval, include_run_info=True)

        evaluation_result = EvaluationResult(
            key=self.metric.name, score=eval_output[self.metric.name]
        )
        if RUN_KEY in eval_output:
            evaluation_result.evaluator_info[RUN_KEY] = eval_output[RUN_KEY]
        return evaluation_result
