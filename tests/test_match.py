"""Matching rules — the part of the metric most likely to produce wrong numbers silently."""

import pytest

from sourcecode.match import (
    lemma_match,
    normalize_text,
    surface_match,
    term_present,
    tokenize,
)
from sourcecode.normalize import is_unspaced_language, normalize_language


def test_word_boundary_prevents_substring_false_positives():
    assert surface_match("a category of things", "cat", "en-us") is False
    assert surface_match("the cat sat", "cat", "en-us") is True


def test_accents_compare_under_nfc_regardless_of_input_normalization():
    decomposed = "le café est ouvert"   # e + combining acute
    composed = "café"                     # é
    assert surface_match(decomposed, composed, "fr-fr") is True


def test_accented_word_boundary_is_respected():
    # "café" must not match inside "cafés" on surface form alone.
    assert surface_match("trois cafés ouverts", "café", "fr-fr") is False


def test_regex_metacharacters_in_terms_are_literal():
    assert surface_match("the C++ compiler", "C++", "en-us") is True
    assert surface_match("the C-- compiler", "C++", "en-us") is False


def test_terms_with_trailing_punctuation_still_match():
    # This is why \b is unusable: it needs a \w/\W transition, which "C++ " does not provide.
    assert surface_match("use C++ here", "c++", "en-us") is True


def test_unspaced_languages_fall_back_to_containment():
    assert is_unspaced_language("ja-jp") is True
    assert surface_match("自動車のエンジン", "エンジン", "ja-jp") is True


def test_casefold_handles_non_ascii_case():
    assert surface_match("DIE STRASSE", "strasse", "de-de") is True


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  a   b \n c ") == "a b c"
    assert normalize_text(None) == ""


def test_lemma_match_requires_contiguous_run():
    assert lemma_match(["le", "moteur", "electrique", "etre"], ["moteur", "electrique"]) is True
    assert lemma_match(["le", "moteur", "etre"], ["moteur", "electrique"]) is False
    # non-contiguous must not match
    assert lemma_match(["moteur", "de", "electrique"], ["moteur", "electrique"]) is False


def test_lemma_match_accepts_strings_or_sequences():
    assert lemma_match("le moteur electrique", "moteur electrique") is True


def test_term_present_reports_how_the_match_was_made():
    surface = term_present(text="the brake pad", term="brake pad", language_code="en-us")
    assert (surface.found, surface.via) == (True, "surface")

    lemma = term_present(
        text="les moteurs electriques",
        term="moteur electrique",
        language_code="fr-fr",
        text_lemmas="le moteur electrique",
        term_lemmas="moteur electrique",
    )
    assert (lemma.found, lemma.via) == (True, "lemma")

    missing = term_present(text="rien ici", term="moteur", language_code="fr-fr")
    assert (missing.found, missing.via) == (False, None)


def test_lemma_fallback_is_skipped_for_unspaced_languages():
    # Token alignment is meaningless without word separation; must not report a lemma match.
    result = term_present(
        text="全然違う", term="エンジン", language_code="ja-jp",
        text_lemmas="全然 違う", term_lemmas="エンジン",
    )
    assert result.found is False


@pytest.mark.parametrize(
    "source,target,expected_source,expected_target",
    [
        ("English (United Kingdom)", "French (France)", "en-gb", "fr-fr"),
        ("en-us", "de-de", "en-us", "de-de"),
        ("Chinese (Simplified, China)", "Japanese (Japan)", "zh-cn", "ja-jp"),
    ],
)
def test_language_names_and_codes_both_resolve(source, target, expected_source, expected_target):
    out = normalize_language({"source_language": source, "target_language": target})
    assert out["clean_source_language_code"] == expected_source
    assert out["clean_target_language_code"] == expected_target


def test_unknown_languages_pass_through_rather_than_raising():
    out = normalize_language({"source_language": "xx-yy", "target_language": "Klingon"})
    assert out["clean_source_language_code"] == "xx-yy"
    assert out["clean_target_language_code"] == "klingon"


def test_tokenize_drops_empties():
    assert tokenize("  a   b  ") == ["a", "b"]
