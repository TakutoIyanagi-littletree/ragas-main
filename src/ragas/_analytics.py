from __future__ import annotations

import json
import logging
import os
import typing as t
import uuid
from dataclasses import asdict
from functools import lru_cache, wraps

import requests
from appdirs import user_data_dir
from langchain_core.pydantic_v1 import BaseModel
from pydantic import Field

from ragas.utils import get_debug_mode

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")
    AsyncFunc = t.Callable[P, t.Coroutine[t.Any, t.Any, t.Any]]

logger = logging.getLogger(__name__)


USAGE_TRACKING_URL = "https://t.explodinggradients.com"
RAGAS_DO_NOT_TRACK = "RAGAS_DO_NOT_TRACK"
RAGAS_DEBUG_TRACKING = "__RAGAS_DEBUG_TRACKING"
USAGE_REQUESTS_TIMEOUT_SEC = 1
USER_DATA_DIR_NAME = "ragas"


@lru_cache(maxsize=1)
def do_not_track() -> bool:  # pragma: no cover
    # Returns True if and only if the environment variable is defined and has value True
    # The function is cached for better performance.
    return os.environ.get(RAGAS_DO_NOT_TRACK, str(False)).lower() == "true"


@lru_cache(maxsize=1)
def _usage_event_debugging() -> bool:
    # For BentoML developers only - debug and print event payload if turned on
    return os.environ.get(RAGAS_DEBUG_TRACKING, str(False)).lower() == "true"


def silent(func: t.Callable[P, T]) -> t.Callable[P, T]:  # pragma: no cover
    # Silent errors when tracking
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            if _usage_event_debugging():
                if get_debug_mode():
                    logger.error(
                        "Tracking Error: %s", err, stack_info=True, stacklevel=3
                    )
                else:
                    logger.info("Tracking Error: %s", err)
            else:
                logger.debug("Tracking Error: %s", err)

    return wrapper


@lru_cache(maxsize=1)
def get_userid() -> str:
    user_id_path = user_data_dir(USER_DATA_DIR_NAME)
    uuid_filepath = os.path.join(user_id_path, "uuid.json")
    if os.path.exists(uuid_filepath):
        user_id = json.load(open(uuid_filepath))["userid"]
    else:
        os.mkdir(user_id_path)
        user_id = "a-" + uuid.uuid4().hex
        with open(uuid_filepath, "w") as f:
            json.dump({"userid": user_id}, f)
    return user_id


class BaseEvent(BaseModel):
    event_type: str
    user_id: str = Field(default_factory=get_userid)


class EvaluationEvent(BaseEvent):
    metrics: t.List[str]
    evaluation_mode: str
    num_rows: int


@silent
def track(event_properties: BaseEvent):
    if do_not_track():
        return

    payload = event_properties.dict()

    if _usage_event_debugging():
        # For internal debugging purpose
        logger.info("Tracking Payload: %s", payload)
        return

    requests.post(USAGE_TRACKING_URL, json=payload, timeout=USAGE_REQUESTS_TIMEOUT_SEC)
