"""Unit tests: tournament hierarchy registry and feature stamping."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.tournaments import (
    REGISTRY,
    AgeLevel,
    Confederation,
    Gender,
    Scope,
    Stage,
    add_tournament_features,
    get_tournament,
    tournament_features,
)


def test_registry_covers_required_competitions() -> None:
    # men's continental + world
    for tid in ["uefa_euro", "copa_america", "gold_cup", "afcon",
                "asian_cup", "ofc_nations_cup", "world_cup"]:
        assert tid in REGISTRY
        assert f"{tid}_w" in REGISTRY, f"missing women's {tid}"
        for youth in [f"{tid}_u20", f"{tid}_u17", f"{tid}_u20_w", f"{tid}_u17_w"]:
            assert youth in REGISTRY, f"missing youth {youth}"
    # club competitions
    for tid in ["club_world_cup", "uefa_champions_league", "copa_libertadores",
                "concacaf_champions_cup", "caf_champions_league",
                "afc_champions_league"]:
        assert tid in REGISTRY
        assert REGISTRY[tid].scope is Scope.CLUB


def test_hierarchy_coordinates() -> None:
    ucl = get_tournament("uefa_champions_league")
    assert ucl.confederation is Confederation.UEFA
    assert ucl.has_two_legged_ties and not ucl.neutral_default

    copa_w = get_tournament("copa_america_w")
    assert copa_w.gender is Gender.WOMEN
    assert copa_w.confederation is Confederation.CONMEBOL

    u17 = get_tournament("world_cup_u17")
    assert u17.age_level is AgeLevel.U17
    # youth discounted below the senior prior of the same confederation
    assert u17.strength_prior < get_tournament("world_cup").strength_prior


def test_strength_priors_ordered_and_uefa_anchored() -> None:
    assert get_tournament("uefa_champions_league").strength_prior == 1.0
    assert (get_tournament("copa_libertadores").strength_prior
            > get_tournament("ofc_nations_cup").strength_prior)


def test_stage_knockout_and_two_legged() -> None:
    assert not Stage.GROUP.knockout
    assert Stage.SEMIFINAL.knockout
    semi = tournament_features("uefa_champions_league", "semifinal")
    assert semi["stage_knockout"] and semi["two_legged_tie"]
    final = tournament_features("uefa_champions_league", "final")
    assert final["stage_knockout"] and not final["two_legged_tie"]  # one-off
    group_intl = tournament_features("uefa_euro", "group")
    assert not group_intl["two_legged_tie"] and group_intl["neutral_default"]


def test_add_tournament_features_stamps_rows() -> None:
    df = pd.DataFrame({
        "match_id": ["m1", "m2"],
        "competition": ["uefa_champions_league", "afcon"],
        "stage": ["group", "final"],
    })
    out = add_tournament_features(df)
    assert out.loc[0, "confederation"] == "UEFA"
    assert out.loc[1, "confederation"] == "CAF"
    assert out.loc[1, "stage_knockout"]
    assert out.loc[0, "strength_prior"] > out.loc[1, "strength_prior"]


def test_unknown_competition_fails_loudly() -> None:
    with pytest.raises(KeyError, match="register it"):
        get_tournament("sunday_league")
