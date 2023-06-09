"""Kedro dataset flavour for MLFlow
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, Union

import kedro
import yaml
from kedro.utils import load_obj as load_dataset
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "kedro_dataset"


DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=[f"kedro[all]={kedro.__version__}"],
    additional_pip_deps=None,
    additional_conda_channels=None,
)


def save_model(
    data: Any,
    path: str,
    conda_env: Union[str, Dict[str, Any]] = None,
    mlflow_model: Model = Model(),
    *,
    dataset_type: str,
    dataset_args: Dict[str, Any],
    file_suffix: str,
):
    """_summary_

    Args:
        data (Any): _description_
        path (str): _description_
        dataset_type (str): _description_
        dataset_args (Dict[str, Any]): _description_
        file_suffix (str): _description_
        conda_env (Union[str, Dict[str, Any]], optional): _description_. Defaults to None.
        mlflow_model (Model, optional): _description_. Defaults to Model().

    Raises:
        RuntimeError: _description_
    """
    if os.path.exists(path):
        raise RuntimeError(f"Path '{path}' already exists")
    os.makedirs(path)

    model_data_subpath = f"data.{file_suffix}"
    model_data_path = os.path.join(path, model_data_subpath)

    cls = load_dataset(dataset_type)
    d_s = cls(filepath=model_data_path, **dataset_args)
    d_s.save(data)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = DEFAULT_CONDA_ENV
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r", encoding="utf-8") as file:
            conda_env = yaml.safe_load(file)
    with open(os.path.join(path, conda_env_subpath), "w", encoding="utf-8") as file:
        yaml.safe_dump(conda_env, stream=file, default_flow_style=False)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module=__name__,
        data=model_data_subpath,
        env=conda_env_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        data=model_data_subpath,
        dataset_type=dataset_type,
        dataset_args=dataset_args,
        file_suffix=file_suffix,
    )
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(
    model: Any,
    artifact_path: str,
    conda_env: Dict[str, Any] = None,
    registered_model_name: str = None,
    await_registration_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    *,
    dataset_type: str,
    dataset_args: Dict[str, Any],
    file_suffix: str,
):
    """_summary_

    Args:
        model (Any): _description_
        artifact_path (str): _description_
        dataset_type (str): _description_
        dataset_args (Dict[str, Any]): _description_
        file_suffix (str): _description_
        conda_env (Dict[str, Any], optional): _description_. Defaults to None.
        registered_model_name (str, optional): _description_. Defaults to None.
        await_registration_for (int, optional):
          _description_. Defaults to DEFAULT_AWAIT_MAX_SLEEP_SECONDS.

    Returns:
        _type_: _description_
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
        data=model,
        conda_env=conda_env,
        dataset_type=dataset_type,
        dataset_args=dataset_args,
        file_suffix=file_suffix,
    )


def _load_model_from_local_file(
    local_path: str,
    *,
    dataset_type: str = None,
    dataset_args: Dict[str, Any] = None,
    file_suffix: str = None,
):
    """_summary_

    Args:
        local_path (str): _description_
        dataset_type (str, optional): _description_. Defaults to None.
        dataset_args (Dict[str, Any], optional): _description_. Defaults to None.
        file_suffix (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if dataset_type is not None:
        model_data_subpath = f"data.{file_suffix}"
        data_path = os.path.join(local_path, model_data_subpath)
    else:
        flavor_conf = _get_flavor_configuration(
            model_path=local_path, flavor_name=FLAVOR_NAME
        )
        data_path = os.path.join(local_path, flavor_conf["data"])
        dataset_type = flavor_conf["dataset_type"]
        dataset_args = flavor_conf["dataset_args"]

    cls = load_dataset(dataset_type)
    d_s = cls(filepath=data_path, **dataset_args)
    return d_s.load()


def load_model(
    model_uri: str,
    *,
    dataset_type: str = None,
    dataset_args: Dict[str, Any] = None,
    file_suffix: str = None,
):
    """_summary_

    Args:
        model_uri (str): _description_
        dataset_type (str, optional): _description_. Defaults to None.
        dataset_args (Dict[str, Any], optional): _description_. Defaults to None.
        file_suffix (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if dataset_type is not None or dataset_args is not None or file_suffix is not None:
        assert (
            dataset_type is not None
            and dataset_args is not None
            and file_suffix is not None
        ), ("Please set 'dataset_type', " "'dataset_args' and 'file_suffix'")

    local_path = _download_artifact_from_uri(model_uri)
    return _load_model_from_local_file(
        local_path,
        dataset_type=dataset_type,
        dataset_args=dataset_args,
        file_suffix=file_suffix,
    )


def _load_pyfunc(model_file: str):
    """_summary_

    Args:
        model_file (str): _description_

    Raises:
        MlflowException: _description_

    Returns:
        _type_: _description_
    """
    local_path = Path(model_file).parent.absolute()
    model = _load_model_from_local_file(local_path)
    if not hasattr(model, "predict"):
        try:
            setattr(model, "predict", None)
        except AttributeError as exc:
            raise MlflowException(
                f"`pyfunc` flavor not supported, use " f"{__name__}.load instead"
            ) from exc
    return model
