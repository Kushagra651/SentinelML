import pandas as pd

from experiments.ge_setup import run_validation

CATS = {
    "workclass": ["Private"],
    "education": ["Bachelors"],
    "marital_status": ["Never-married"],
    "occupation": ["Craft-repair"],
    "relationship": ["Not-in-family"],
    "race": ["White"],
    "sex": ["Male", "Female"],
    "native_country": ["United-States"],
}
ROW = {
    "age": 34,
    "fnlwgt": 200000,
    "education_num": 13,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "workclass": "Private",
    "education": "Bachelors",
    "marital_status": "Never-married",
    "occupation": "Craft-repair",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native_country": "United-States",
    "income": "<=50K",
}


def _df(**over):
    return pd.DataFrame([{**ROW, **over}] * 5)


def test_clean_passes():
    assert run_validation(_df(), CATS)["passed"]


def test_negative_age_fails():
    r = run_validation(_df(age=-5), CATS)
    assert not r["passed"] and any(e["column"] == "age" for e in r["errors"])


def test_invalid_category_fails():
    r = run_validation(_df(sex="Unknown"), CATS)
    assert not r["passed"] and any(e["column"] == "sex" for e in r["errors"])
