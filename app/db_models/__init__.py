"""
Import models here so Alembic env.py can import database.models and discover everything.
"""

from database.models.city import City, CityAnnualStats
from database.models.sector import Sector
from database.models.emissions import EmissionRecord
from database.models.budgets import CityBudget, FundingSource, BudgetFunding
from database.models.initiatives import (
    Initiative,
    Stakeholder,
    InitiativeStakeholder,
    InitiativeIndicator,
)
from database.models.indicators import Indicator, IndicatorValue, CityTarget
from database.models.tef import TefCategory, InitiativeTef
from database.models.contracts import ClimateCityContract

__all__ = [
    "City",
    "CityAnnualStats",
    "Sector",
    "EmissionRecord",
    "CityBudget",
    "FundingSource",
    "BudgetFunding",
    "Initiative",
    "Stakeholder",
    "InitiativeStakeholder",
    "Indicator",
    "IndicatorValue",
    "CityTarget",
    "InitiativeIndicator",
    "TefCategory",
    "InitiativeTef",
    "ClimateCityContract",
]
