"""Materialize offline -> Redis, then do one online read."""

from datetime import datetime, timezone

from feast import FeatureStore

store = FeatureStore(repo_path="/app/experiments/feast_repo")
store.materialize(
    start_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
    end_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
)
feats = [f"adult_raw:{c}" for c in ["age", "education", "sex", "hours_per_week"]]
print(
    store.get_online_features(features=feats, entity_rows=[{"person_id": 5}]).to_dict()
)
