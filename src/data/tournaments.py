"""Global tournament hierarchy: Confederation → Tournament → Stage → Match.

The registry below is the single authority on where a competition sits in
world football, and it exists for one modeling reason: **cross-league
strength normalization**. A team's club form is earned against opposition of
its confederation/tier; when squads meet across that boundary (international
tournaments, Club World Cup), form must be rescaled by the strength prior
before it means anything. `add_tournament_features` stamps every match row
with its hierarchy coordinates so trained models can consume them.

Priors are deliberately coarse (confederation-level, senior club play
anchored at UEFA=1.0) and are meant to be *replaced by fitted coefficients*
once real cross-confederation results are ingested — the schema is the
contract, the numbers are defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Confederation(str, Enum):
    UEFA = "UEFA"
    CONMEBOL = "CONMEBOL"
    CONCACAF = "CONCACAF"
    CAF = "CAF"
    AFC = "AFC"
    OFC = "OFC"
    FIFA = "FIFA"          # world-level competitions


class Scope(str, Enum):
    INTERNATIONAL = "international"   # national teams
    CLUB = "club"


class Gender(str, Enum):
    MEN = "men"
    WOMEN = "women"


class AgeLevel(str, Enum):
    SENIOR = "senior"
    U20 = "u20"
    U17 = "u17"


class Stage(str, Enum):
    GROUP = "group"
    ROUND_OF_32 = "round_of_32"
    ROUND_OF_16 = "round_of_16"
    QUARTERFINAL = "quarterfinal"
    SEMIFINAL = "semifinal"
    THIRD_PLACE = "third_place"
    FINAL = "final"

    @property
    def knockout(self) -> bool:
        return self is not Stage.GROUP


# confederation strength priors (senior men's club level, UEFA anchored 1.0)
CONFEDERATION_STRENGTH: dict[Confederation, float] = {
    Confederation.UEFA: 1.00,
    Confederation.CONMEBOL: 0.92,
    Confederation.CONCACAF: 0.78,
    Confederation.AFC: 0.74,
    Confederation.CAF: 0.72,
    Confederation.OFC: 0.55,
    Confederation.FIFA: 1.00,   # world events mix; per-team priors dominate
}

_AGE_DISCOUNT: dict[AgeLevel, float] = {
    AgeLevel.SENIOR: 1.00, AgeLevel.U20: 0.80, AgeLevel.U17: 0.70,
}


@dataclass(frozen=True)
class Tournament:
    id: str
    name: str
    confederation: Confederation
    scope: Scope
    gender: Gender = Gender.MEN
    age_level: AgeLevel = AgeLevel.SENIOR
    has_group_stage: bool = True
    has_two_legged_ties: bool = False
    neutral_default: bool = True     # False: home/away legs are the norm

    @property
    def strength_prior(self) -> float:
        return (CONFEDERATION_STRENGTH[self.confederation]
                * _AGE_DISCOUNT[self.age_level])


def _t(id_: str, name: str, conf: Confederation, scope: Scope, **kw) -> Tournament:
    return Tournament(id=id_, name=name, confederation=conf, scope=scope, **kw)


_MEN_INTL = [
    _t("uefa_euro", "UEFA European Championship", Confederation.UEFA, Scope.INTERNATIONAL),
    _t("copa_america", "Copa América", Confederation.CONMEBOL, Scope.INTERNATIONAL),
    _t("gold_cup", "CONCACAF Gold Cup", Confederation.CONCACAF, Scope.INTERNATIONAL),
    _t("afcon", "Africa Cup of Nations", Confederation.CAF, Scope.INTERNATIONAL),
    _t("asian_cup", "AFC Asian Cup", Confederation.AFC, Scope.INTERNATIONAL),
    _t("ofc_nations_cup", "OFC Nations Cup", Confederation.OFC, Scope.INTERNATIONAL),
    _t("world_cup", "FIFA World Cup", Confederation.FIFA, Scope.INTERNATIONAL),
]

_CLUB = [
    _t("club_world_cup", "FIFA Club World Cup", Confederation.FIFA, Scope.CLUB),
    _t("uefa_champions_league", "UEFA Champions League", Confederation.UEFA,
       Scope.CLUB, has_two_legged_ties=True, neutral_default=False),
    _t("copa_libertadores", "Copa Libertadores", Confederation.CONMEBOL,
       Scope.CLUB, has_two_legged_ties=True, neutral_default=False),
    _t("concacaf_champions_cup", "CONCACAF Champions Cup", Confederation.CONCACAF,
       Scope.CLUB, has_two_legged_ties=True, neutral_default=False),
    _t("caf_champions_league", "CAF Champions League", Confederation.CAF,
       Scope.CLUB, has_two_legged_ties=True, neutral_default=False),
    _t("afc_champions_league", "AFC Champions League Elite", Confederation.AFC,
       Scope.CLUB, has_two_legged_ties=True, neutral_default=False),
]


def _women(t: Tournament) -> Tournament:
    return Tournament(
        id=f"{t.id}_w", name=f"{t.name} (Women)", confederation=t.confederation,
        scope=t.scope, gender=Gender.WOMEN, age_level=t.age_level,
        has_group_stage=t.has_group_stage,
        has_two_legged_ties=t.has_two_legged_ties,
        neutral_default=t.neutral_default,
    )


def _youth(t: Tournament, age: AgeLevel, gender: Gender) -> Tournament:
    tag = age.value + ("_w" if gender is Gender.WOMEN else "")
    label = age.value.upper().replace("U", "U-")
    suffix = f" ({label} Women)" if gender is Gender.WOMEN else f" ({label})"
    return Tournament(
        id=f"{t.id}_{tag}", name=t.name + suffix,
        confederation=t.confederation, scope=t.scope, gender=gender,
        age_level=age,
    )


REGISTRY: dict[str, Tournament] = {t.id: t for t in _MEN_INTL + _CLUB}
REGISTRY.update({w.id: w for w in map(_women, _MEN_INTL)})
for base in _MEN_INTL:
    for age in (AgeLevel.U20, AgeLevel.U17):
        for gender in (Gender.MEN, Gender.WOMEN):
            y = _youth(base, age, gender)
            REGISTRY[y.id] = y


def get_tournament(competition_id: str) -> Tournament:
    try:
        return REGISTRY[competition_id]
    except KeyError:
        raise KeyError(
            f"unknown competition {competition_id!r}; register it in "
            "src/data/tournaments.py before ingesting its matches"
        ) from None


def tournament_features(competition_id: str, stage: str) -> dict:
    """Feature columns for one match: hierarchy coordinates + priors."""
    t = get_tournament(competition_id)
    s = Stage(stage)
    return {
        "confederation": t.confederation.value,
        "tournament_id": t.id,
        "scope": t.scope.value,
        "gender": t.gender.value,
        "age_level": t.age_level.value,
        "strength_prior": t.strength_prior,
        "stage_knockout": s.knockout,
        "two_legged_tie": t.has_two_legged_ties and s.knockout and s is not Stage.FINAL,
        "neutral_default": t.neutral_default,
    }


def add_tournament_features(
    df: pd.DataFrame, *, competition_col: str = "competition",
    stage_col: str = "stage",
) -> pd.DataFrame:
    """Stamp every match row with its hierarchy features (validates that all
    competitions are registered — unknown ids fail loudly, never default)."""
    feats = pd.DataFrame(
        [tournament_features(c, s)
         for c, s in zip(df[competition_col], df[stage_col])],
        index=df.index,
    )
    return pd.concat([df, feats], axis=1)
