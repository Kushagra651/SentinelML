"""Great Expectations suite + checkpoint for the 14 features + income label (GE 1.x, pandas only)."""

import os
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

NUM_RANGES = {  # fixed sanity bounds
    "age": (0, 120),
    "fnlwgt": (0, None),
    "education_num": (1, 16),
    "capital_gain": (0, None),
    "capital_loss": (0, None),
    "hours_per_week": (0, 168),
}
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
LABELS = [">50K", "<=50K"]
ALL_COLS = list(NUM_RANGES) + CAT + ["income"]


def categories_from_train() -> dict:
    store = Path(os.getenv("FEATURE_STORE_DIR", "data/feature_store"))
    df = pd.read_parquet(store / "train.parquet")
    return {c: sorted(df[c].unique().tolist()) for c in CAT}


def build_suite(cats: dict) -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="adult_suite")
    add = suite.add_expectation
    add(gxe.ExpectTableColumnsToMatchSet(column_set=ALL_COLS, exact_match=True))
    for c in ALL_COLS:
        add(gxe.ExpectColumnValuesToNotBeNull(column=c))
    for c, (lo, hi) in NUM_RANGES.items():
        add(
            gxe.ExpectColumnValuesToBeInTypeList(
                column=c, type_list=["int64", "int32", "float64"]
            )
        )
        add(gxe.ExpectColumnValuesToBeBetween(column=c, min_value=lo, max_value=hi))
    for c in CAT:
        add(gxe.ExpectColumnValuesToBeInSet(column=c, value_set=cats[c]))
    add(gxe.ExpectColumnValuesToBeInSet(column="income", value_set=LABELS))
    return suite


def run_validation(df: pd.DataFrame, cats: dict | None = None) -> dict:
    """Returns {passed, errors, row_count}. Ephemeral context, no files written."""
    cats = cats or categories_from_train()
    ctx = gx.get_context(mode="ephemeral")
    asset = ctx.data_sources.add_pandas("pd").add_dataframe_asset("adult")
    batch = asset.add_batch_definition_whole_dataframe("whole").get_batch(
        batch_parameters={"dataframe": df}
    )
    res = batch.validate(build_suite(cats))
    errors = []
    for r in res.results:
        if not r.success:
            cfg = r.expectation_config
            errors.append(
                {
                    "check": cfg.type,
                    "column": cfg.kwargs.get("column"),
                    "unexpected_count": (r.result or {}).get("unexpected_count"),
                }
            )
    return {"passed": bool(res.success), "errors": errors, "row_count": len(df)}


def build_data_docs() -> None:
    """File-backed run so Data Docs HTML exists."""
    root = os.getenv("GE_DIR", "/app/artifacts/ge")
    ctx = gx.get_context(mode="file", project_root_dir=root)
    df = pd.read_parquet(
        Path(os.getenv("FEATURE_STORE_DIR", "data/feature_store")) / "train.parquet"
    )
    asset = ctx.data_sources.add_or_update_pandas("pd").add_dataframe_asset("adult")
    bd = asset.add_batch_definition_whole_dataframe("whole")
    suite = ctx.suites.add_or_update(build_suite(categories_from_train()))
    vd = ctx.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="adult_vd", data=bd, suite=suite)
    )
    cp = ctx.checkpoints.add_or_update(
        gx.Checkpoint(
            name="adult_cp",
            validation_definitions=[vd],
            actions=[gx.checkpoint.UpdateDataDocsAction(name="docs")],
        )
    )
    res = cp.run(batch_parameters={"dataframe": df})
    print("checkpoint success:", res.success)
    print("Data Docs:", ctx.build_data_docs())


if __name__ == "__main__":
    import sys

    if "--docs" in sys.argv:
        build_data_docs()
    else:
        train = pd.read_parquet(
            Path(os.getenv("FEATURE_STORE_DIR", "data/feature_store")) / "train.parquet"
        )
        clean = run_validation(train)
        bad = train.head(1000).copy()
        bad.loc[bad.index[0], "age"] = -5
        bad.loc[bad.index[1], "sex"] = "Unknown"
        corrupted = run_validation(bad)
        print("clean :", clean["passed"], clean["row_count"], clean["errors"])
        print("bad   :", corrupted["passed"], corrupted["errors"])
