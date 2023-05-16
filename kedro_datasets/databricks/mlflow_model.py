"""``MLFlowModel`` implementation to access managed MLFLow
in Databricks.
"""
import importlib
import logging
from typing import Any, Dict

from kedro.io.core import AbstractDataSet
from mlflow.models.signature import ModelSignature

from .mlflow_common import ModelOpsException, parse_model_uri

logger = logging.getLogger(__name__)


class MLFlowModel(AbstractDataSet):
    """_summary_

    Args:
        AbstractDataSet (_type_): _description_
    """

    def __init__(  # pylint: disable=R0913
        self,
        flavor: str,
        model_name: str,
        signature: Dict[str, Dict[str, str]] = None,
        input_example: Dict[str, Any] = None,
        load_version: str = None,
    ):
        self._flavor = flavor
        self._model_name = model_name

        if signature:
            self._signature = ModelSignature.from_dict(signature)
        else:
            self._signature = None
        self._input_example = input_example

        self._load_version = load_version

    def _save(self, data: Any) -> None:
        if self._load_version is not None:
            msg = (
                f"Trying to save an MLFlowModel::{self._describe} which "
                f"was initialized with load_version={self._load_version}. "
                f"This can lead to inconsistency between saved and loaded "
                f"versions, therefore disallowed. Please create separate "
                f"catalog entries for saved and loaded datasets."
            )
            raise ModelOpsException(msg)

        importlib.import_module(self._flavor).log_model(
            data,
            self._model_name,
            registered_model_name=self._model_name,
            signature=self._signature,
            input_example=self._input_example,
        )

    def _load(self) -> Any:
        *_, latest_version = parse_model_uri(f"models:/{self._model_name}")

        model_version = self._load_version or latest_version

        logger.info("Loading model '%s' version '%s'", self._model_name, model_version)

        if model_version != latest_version:
            logger.warning("Newer version %s exists in repo", latest_version)

        model = importlib.import_module(self._flavor).load_model(
            f"models:/{self._model_name}/{model_version}"
        )

        return model

    def _describe(self) -> Dict[str, Any]:
        return {
            "flavor": self._flavor,
            "model_name": self._model_name,
            "signature": self._signature,
            "input_example": self._input_example,
            "load_version": self._load_version,
        }
