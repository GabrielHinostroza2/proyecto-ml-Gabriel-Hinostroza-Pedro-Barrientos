
from kedro.pipeline import Pipeline
from proyecto_ml_gabriel_hinostroza_pedro_barrientos.pipelines import data_processing

def register_pipelines() -> dict[str, Pipeline]:
    dp = data_processing.create_pipeline()
    return {"__default__": dp, "data_processing": dp}
