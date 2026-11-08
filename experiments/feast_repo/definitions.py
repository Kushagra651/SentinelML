from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, String

person = Entity(name="person", join_keys=["person_id"])

source = FileSource(
    path="/app/artifacts/feature_store/train_feast.parquet",
    timestamp_field="event_timestamp",
)

NUM = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
CAT = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

adult_raw = FeatureView(
    name="adult_raw",
    entities=[person],
    ttl=timedelta(days=3650),
    schema=[Field(name=c, dtype=Int64) for c in NUM]
    + [Field(name=c, dtype=String) for c in CAT],
    source=source,
)
