from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset, concatenate_datasets

from ragas._analytics import EvaluationEvent, track
from ragas.async_utils import run_async_tasks
from ragas.callbacks import new_group
from ragas.metrics.base import Metric

# from ragas.metrics.critique import AspectCritique
from ragas.validation import (
    remap_column_names,
    validate_column_dtypes,
    validate_evaluation_modes,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


def evaluate(
    dataset: Dataset,
    metrics: list[Metric] | None = None,
    callbacks: Callbacks = [],
    is_async: bool = True,
    column_map: dict[str, str] = {},
) -> Result:
    """
    Run the evaluation on the dataset with different metrics

    Parameters
    ----------
    dataset : Dataset[question: list[str], contexts: list[list[str]], answer: list[str]]
        The dataset in the format of ragas which the metrics will use to score the RAG
        pipeline with
    metrics : list[Metric] , optional
        List of metrics to use for evaluation. If not provided then ragas will run the
        evaluation on the best set of metrics to give a complete view.
    column_map : dict[str, str], optional
        The column names of the dataset to use for evaluation. If the column names of
        the dataset are different from the default ones then you can provide the
        mapping as a dictionary here. Example: If the dataset column name is contexts_v1,
        column_map can be given as {"contexts":"contexts_v1"}

    Returns
    -------
    Result
        Result object containing the scores of each metric. You can use this do analysis
        later. If the top 3 metrics are provided then it also returns the `ragas_score`
        for the entire pipeline.

    Raises
    ------
    ValueError
        if validation fails because the columns required for the metrics are missing or
        if the columns are of the wrong format.

    Examples
    --------
    the basic usage is as follows:
    ```
    from ragas import evaluate

    >>> dataset
    Dataset({
        features: ['question', 'ground_truths', 'answer', 'contexts'],
        num_rows: 30
    })

    >>> result = evaluate(dataset)
    >>> print(result["ragas_score"])
    {'ragas_score': 0.860, 'context_precision': 0.817, 'faithfulness': 0.892,
    'answer_relevancy': 0.874}
    ```
    """
    if dataset is None:
        raise ValueError("Provide dataset!")

    if metrics is None:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [answer_relevancy, context_precision, faithfulness, context_recall]

    # remap column names from the dataset
    dataset = remap_column_names(dataset, column_map)
    # validation
    validate_evaluation_modes(dataset, metrics)
    validate_column_dtypes(dataset)

    # run the evaluation on dataset with different metrics
    # initialize all the models in the metrics
    [m.init_model() for m in metrics]

    # new evaluation chain
    evaluation_rm, evaluation_group_cm = new_group(
        name="ragas evaluation", inputs={}, callbacks=callbacks, is_async=is_async
    )
    # list of chains for each row
    row_chains = []

    scoring_tasks = []
    binary_metrics = []
    for metric in metrics:
        # if isinstance(metric, AspectCritique):
        # binary_metrics.append(metric.name)
        ...
    for i, row in enumerate(dataset):
        row_rm, row_group_cm = new_group(
            name=f"row {i}",
            inputs=row,
            callbacks=evaluation_group_cm,
            is_async=is_async,
        )
        scoring_tasks.extend(
            [metric.ascore(data_row=row, callbacks=row_group_cm) for metric in metrics]
        )
        row_chains.append(row_rm)
    # run the evaluation tasks
    try:
        results = run_async_tasks(
            scoring_tasks, show_progress=True, progress_bar_desc="evaluating dataset"
        )
        # TODO: closing row chains here. handle errors here too
        [chain.on_chain_end({}) for chain in row_chains]

    # run evaluation task
    except Exception as e:
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_error(e)
        raise e
    else:
        if not evaluation_group_cm.ended:
            evaluation_rm.on_chain_end({})

    return results

    # log the evaluation event
    metrics_names = [m.name for m in metrics]
    track(
        EvaluationEvent(
            event_type="evaluation",
            metrics=metrics_names,
            evaluation_mode="",
            num_rows=dataset.shape[0],
        )
    )

    return Result(
        scores=concatenate_datasets(scores, axis=1),
        dataset=dataset,
        binary_columns=binary_metrics,
    )


@dataclass
class Result(dict):
    scores: Dataset
    dataset: Dataset | None = None
    binary_columns: list[str] = field(default_factory=list)

    def __post_init__(self):
        values = []
        for cn in self.scores.column_names:
            value = np.nanmean(self.scores[cn])
            self[cn] = value
            if cn not in self.binary_columns:
                value = t.cast(float, value)
                values.append(value + 1e-10)

    def to_pandas(self, batch_size: int | None = None, batched: bool = False):
        if self.dataset is None:
            raise ValueError("dataset is not provided for the results class")
        assert self.scores.shape[0] == self.dataset.shape[0]
        result_ds = concatenate_datasets([self.dataset, self.scores], axis=1)

        return result_ds.to_pandas(batch_size=batch_size, batched=batched)

    def __repr__(self) -> str:
        scores = self.copy()
        score_strs = [f"'{k}': {v:0.4f}" for k, v in scores.items()]
        return "{" + ", ".join(score_strs) + "}"
