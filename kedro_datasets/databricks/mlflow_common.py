"""common library for mlflow
"""
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_RUN_ID_ENV_VAR = "mlflow_run_id"


def parse_model_uri(model_uri):
    """_summary_

    Args:
        model_uri (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    parts = model_uri.split("/")

    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(
            "model uri should have the format "
            "'models:/<model_name>' or "
            f"'models:/<model_name>/<version>', got {model_uri}"
        )

    if parts[0] == "models:":
        protocol = "models"
    else:
        raise ValueError(f"model uri should start with `models:/`, got {model_uri}")

    name = parts[1]

    client = MlflowClient()
    if len(parts) == 2:
        results = client.search_model_versions(f"name='{name}'")
        sorted_results = sorted(
            results,
            key=lambda modelversion: modelversion.creation_timestamp,
            reverse=True,
        )
        latest_version = sorted_results[0].version
        version = latest_version
    else:
        version = parts[2]
        if version in ["Production", "Staging", "Archived"]:
            results = client.get_latest_versions(name, stages=[version])
            if len(results) > 0:
                version = results[0].version
            else:
                version = None

    return protocol, name, version


def promote_model(model_name, model_version, stage):
    """_summary_

    Args:
        model_name (_type_): _description_
        model_version (_type_): _description_
        stage (_type_): _description_
    """
    import datetime  # pylint: disable=C0415

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    client = MlflowClient()

    new_model_uri = f"models:/{model_name}/{model_version}"
    _, _, new_model_version = parse_model_uri(new_model_uri)
    new_model = mlflow.pyfunc.load_model(new_model_uri)
    new_model_runid = new_model._model_meta.run_id  # pylint: disable=W0212

    client.set_tag(
        new_model_runid,
        "mlflow.note.content",
        f"```Promoted version {model_version} to {stage}, at {now}```",
    )
    client.set_tag(new_model_runid, "Promoted at", now)

    results = client.get_latest_versions(model_name, stages=[stage])
    if len(results) > 0:
        old_model_uri = f"models:/{model_name}/{stage}"
        _, _, old_model_version = parse_model_uri(old_model_uri)
        old_model = mlflow.pyfunc.load_model(old_model_uri)
        old_model_runid = old_model._model_meta.run_id  # pylint: disable=W0212

        client.set_tag(
            old_model._model_meta.run_id,  # pylint: disable=W0212
            "mlflow.note.content",
            f"```Replaced by version {new_model_version}, at {now}```",
        )
        client.set_tag(old_model_runid, "Retired at", now)
        client.set_tag(old_model_runid, "Replaced by", new_model_version)

        client.set_tag(new_model_runid, "Replaces", old_model_version)

        client.transition_model_version_stage(
            name=model_name, version=old_model_version, stage="Archived"
        )

    client.transition_model_version_stage(
        name=model_name, version=new_model_version, stage=stage
    )


class ModelOpsException(Exception):
    """_summary_

    Args:
        Exception (_type_): _description_
    """

    pass
