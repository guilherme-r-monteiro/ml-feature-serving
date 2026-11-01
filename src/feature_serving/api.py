import sqlite3
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feature_serving.modeling import NamedModel
from feature_serving.router import ThompsonRouter
from feature_serving.stores import read_online


class ScoreRequest(BaseModel):
    client_id: int
    as_of: int


class ScoreResponse(BaseModel):
    request_id: str
    model: str
    model_version: str
    feature_version: str
    snapshot: str
    score: float


def create_app(connection: sqlite3.Connection, models: list[NamedModel], router: ThompsonRouter) -> FastAPI:
    app = FastAPI()

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        stored = read_online(connection, request.client_id, request.as_of)
        if stored is None:
            raise HTTPException(status_code=404, detail="no features for client and month")
        values, version, snapshot = stored
        model = models[router.choose()]
        value = float(model.score(pd.DataFrame([values]))[0])
        request_id = uuid.uuid4().hex
        connection.execute(
            "INSERT INTO prediction_log VALUES (?,?,?,?,?,?,?,?)",
            (request_id, request.client_id, request.as_of, model.name, model.version, version, snapshot, value),
        )
        connection.commit()
        return ScoreResponse(
            request_id=request_id, model=model.name, model_version=model.version, feature_version=version, snapshot=snapshot, score=value
        )

    return app
