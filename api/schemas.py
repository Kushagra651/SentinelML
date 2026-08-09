"""
api/schemas.py
Pydantic v2 request/response schemas for the ML monitoring API.
Single source of truth for input features, output structure, and API contracts.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Categorical enums — UCI Adult Income dataset
# ---------------------------------------------------------------------------


class Workclass(str, Enum):
    PRIVATE = "Private"
    SELF_EMP_NOT_INC = "Self-emp-not-inc"
    SELF_EMP_INC = "Self-emp-inc"
    FEDERAL_GOV = "Federal-gov"
    LOCAL_GOV = "Local-gov"
    STATE_GOV = "State-gov"
    WITHOUT_PAY = "Without-pay"
    NEVER_WORKED = "Never-worked"


class Education(str, Enum):
    BACHELORS = "Bachelors"
    SOME_COLLEGE = "Some-college"
    ELEVENTH = "11th"
    HS_GRAD = "HS-grad"
    PROF_SCHOOL = "Prof-school"
    ASSOC_ACDM = "Assoc-acdm"
    ASSOC_VOC = "Assoc-voc"
    NINTH = "9th"
    SEVENTH_EIGHTH = "7th-8th"
    TWELFTH = "12th"
    MASTERS = "Masters"
    FIRST_FOURTH = "1st-4th"
    TENTH = "10th"
    DOCTORATE = "Doctorate"
    FIFTH_SIXTH = "5th-6th"
    PRESCHOOL = "Preschool"


class MaritalStatus(str, Enum):
    MARRIED_CIV_SPOUSE = "Married-civ-spouse"
    DIVORCED = "Divorced"
    NEVER_MARRIED = "Never-married"
    SEPARATED = "Separated"
    WIDOWED = "Widowed"
    MARRIED_SPOUSE_ABSENT = "Married-spouse-absent"
    MARRIED_AF_SPOUSE = "Married-AF-spouse"


class Occupation(str, Enum):
    TECH_SUPPORT = "Tech-support"
    CRAFT_REPAIR = "Craft-repair"
    OTHER_SERVICE = "Other-service"
    SALES = "Sales"
    EXEC_MANAGERIAL = "Exec-managerial"
    PROF_SPECIALTY = "Prof-specialty"
    HANDLERS_CLEANERS = "Handlers-cleaners"
    MACHINE_OP_INSPCT = "Machine-op-inspct"
    ADM_CLERICAL = "Adm-clerical"
    FARMING_FISHING = "Farming-fishing"
    TRANSPORT_MOVING = "Transport-moving"
    PRIV_HOUSE_SERV = "Priv-house-serv"
    PROTECTIVE_SERV = "Protective-serv"
    ARMED_FORCES = "Armed-Forces"


class Relationship(str, Enum):
    WIFE = "Wife"
    OWN_CHILD = "Own-child"
    HUSBAND = "Husband"
    NOT_IN_FAMILY = "Not-in-family"
    OTHER_RELATIVE = "Other-relative"
    UNMARRIED = "Unmarried"


class Race(str, Enum):
    WHITE = "White"
    ASIAN_PAC_ISLANDER = "Asian-Pac-Islander"
    AMER_INDIAN_ESKIMO = "Amer-Indian-Eskimo"
    OTHER = "Other"
    BLACK = "Black"


class Sex(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class NativeCountry(str, Enum):
    UNITED_STATES = "United-States"
    CUBA = "Cuba"
    JAMAICA = "Jamaica"
    INDIA = "India"
    MEXICO = "Mexico"
    SOUTH = "South"
    JAPAN = "Japan"
    PHILIPPINES = "Philippines"
    GERMANY = "Germany"
    PUERTO_RICO = "Puerto-Rico"
    CANADA = "Canada"
    EL_SALVADOR = "El-Salvador"
    ENGLAND = "England"
    DOMINICAN_REPUBLIC = "Dominican-Republic"
    ITALY = "Italy"
    COLUMBIA = "Columbia"
    PORTUGAL = "Portugal"
    CHINA = "China"
    ECUADOR = "Ecuador"
    POLAND = "Poland"
    FRANCE = "France"
    HAITI = "Haiti"
    GUATEMALA = "Guatemala"
    IRAN = "Iran"
    NICARAGUA = "Nicaragua"
    SCOTLAND = "Scotland"
    THAILAND = "Thailand"
    YUGOSLAVIA = "Yugoslavia"
    TRINIDAD_TOBAGO = "Trinidad&Tobago"
    GREECE = "Greece"
    VIETNAM = "Vietnam"
    HONG = "Hong"
    IRELAND = "Ireland"
    HUNGARY = "Hungary"
    CAMBODIA = "Cambodia"
    TAIWAN = "Taiwan"
    LAOS = "Laos"
    PERU = "Peru"
    OUTLYING_US = "Outlying-US(Guam-USVI-etc)"
    HONDURAS = "Honduras"
    HUNGARY2 = "Hungary"
    TRINADAD_TOBAGO = "Trinadad&Tobago"


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class PredictionInput(BaseModel):
    """Input features for a single income prediction."""

    # Numerical features
    age: int = Field(..., ge=17, le=90, description="Age in years")
    fnlwgt: int = Field(..., ge=1, description="Final weight (census sampling weight)")
    education_num: int = Field(..., ge=1, le=16, description="Education level as integer")
    capital_gain: float = Field(..., ge=0, description="Capital gain in USD")
    capital_loss: float = Field(..., ge=0, description="Capital loss in USD")
    hours_per_week: int = Field(..., ge=1, le=99, description="Hours worked per week")

    # Categorical features
    workclass: Workclass
    education: Education
    marital_status: MaritalStatus
    occupation: Occupation
    relationship: Relationship
    race: Race
    sex: Sex
    native_country: NativeCountry

    @model_validator(mode="after")
    def capital_gain_and_loss_not_both_nonzero(self) -> "PredictionInput":
        """capital_gain and capital_loss cannot both be non-zero."""
        if self.capital_gain > 0 and self.capital_loss > 0:
            raise ValueError(
                "capital_gain and capital_loss cannot both be non-zero. "
                "A person cannot simultaneously have capital gains and losses."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "workclass": "Private",
                "fnlwgt": 200000,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PredictionOutput(BaseModel):
    """Response from POST /predict."""

    prediction: int = Field(..., description="0 = <=50K, 1 = >50K")
    probability_class_0: float = Field(..., description="P(income <=50K)")
    probability_class_1: float = Field(..., description="P(income >50K)")
    confidence: float = Field(..., description="Max class probability")
    model_version: str = Field(..., description="Version tag of the serving model")
    model_alias: str = Field(..., description="Registry alias (e.g. 'production')")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    features_used: list[str] = Field(..., description="Feature names used by the model")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings")
    request_id: Optional[str] = Field(None, description="Echo of caller-supplied request ID")


class PredictionLog(BaseModel):
    """Schema for prediction log records returned by GET /logs."""

    request_id: str
    timestamp: str
    model_version: str
    model_alias: str
    features: dict
    prediction: int
    probability_class_0: float
    probability_class_1: float
    confidence: float
    latency_ms: float
    ground_truth: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)
    schema_version: str = "1.0"


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = Field(..., description="'ok' when healthy")
    model_loaded: bool
    model_version: Optional[str] = None
    model_alias: Optional[str] = None
    uptime_seconds: Optional[float] = None


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None