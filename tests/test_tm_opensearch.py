"""The OpenSearch-backed TM component. No test touches a cluster: the index client is driven by a
fake search transport."""

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sourcecode import tm_benchmark, tm_embed, tm_gold, tm_index, tm_match, tm_report, tm_score


ROW = {
    'query_id': 'q_en_zh_0001',
    'cat_project_id': 'R-0322-23',
    'source': 'Make sure all statements are using the same base',
    'raw_mt': '請確保所有陳述式使用相同的基準',
    'reference': '請確保所有語句都使用相同的基座標系',
    'source_language': 'en-US',
    'target_language': 'zh-TW',
    'domain': 'software',
    'tm_match': {'aaa': 2, 'bbb': 3},
    'hard_negatives': ['neg-1', 'neg-2'],
}


def write_gold(tmp_path, rows):
    path = tmp_path / 'gold.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    return path


def test_row_is_declared_when_it_lists_entry(tmp_path):
    gold = tm_gold.load_gold_set(write_gold(tmp_path, [ROW]))
    assert gold.declared == gold.rows


def test_row_with_no_listed_entry_is_kept_undeclared(tmp_path):
    gold = tm_gold.load_gold_set(write_gold(tmp_path, [{**ROW, 'tm_match': {}}]))
    assert len(gold.rows) == 1
    assert gold.declared == []


def test_candidates_ranked_by_grade(tmp_path):
    """4 is the most relevant entry and 1 the least, whatever order the file writes them in."""
    row = {**ROW, 'tm_match': {'least': 1, 'most': 4, 'low': 2, 'middle': 3}}
    gold = tm_gold.load_gold_set(write_gold(tmp_path, [row]))
    assert gold.rows[0].candidate_ids == ['most', 'middle', 'low', 'least']


def test_equal_grades_keep_written_order(tmp_path):
    row = {**ROW, 'tm_match': {'first': 3, 'second': 3, 'third': 2}}
    gold = tm_gold.load_gold_set(write_gold(tmp_path, [row]))
    assert gold.rows[0].candidate_ids == ['first', 'second', 'third']


def test_rows_group_into_submissions_by_project(tmp_path):
    rows = [
        ROW,
        {**ROW, 'query_id': 'q2'},
        {**ROW, 'query_id': 'q3', 'cat_project_id': 'R-9999-99'},
    ]
    gold = tm_gold.load_gold_set(write_gold(tmp_path, rows))
    submissions = gold.submissions()
    assert len(submissions) == 2
    assert sorted(len(s.rows) for s in submissions) == [1, 2]


def test_raw_mt_goes_out_as_mt_column(tmp_path):
    """post-mt edits the MT the gold set carries, so no translating step runs ahead of APE."""
    gold = tm_gold.load_gold_set(write_gold(tmp_path, gold_rows()))
    dataset = tm_benchmark.as_dataset(gold, gold.submissions()[0])

    assert [segment['target_content'] for segment in dataset.segments] == [
        row['raw_mt'] for row in gold_rows()
    ]
    assert dataset.steps == ['AQE', 'APE']


def test_row_without_raw_mt_rejected(tmp_path):
    """With no MT to edit, post-mt returns nothing to score against the entries."""
    with pytest.raises(ValueError, match='raw_mt'):
        tm_gold.load_gold_set(write_gold(tmp_path, [{**ROW, 'raw_mt': ''}]))


def test_gold_set_spanning_two_language_pairs_rejected(tmp_path):
    """A result carries one stratum, so a mixed file would report German rows as French ones."""
    rows = [ROW, {**ROW, 'query_id': 'q2', 'target_language': 'de-DE'}]
    with pytest.raises(ValueError, match='strata'):
        tm_gold.load_gold_set(write_gold(tmp_path, rows))


def test_gold_set_spanning_two_domains_rejected(tmp_path):
    rows = [ROW, {**ROW, 'query_id': 'q2', 'domain': 'legal'}]
    with pytest.raises(ValueError, match='strata'):
        tm_gold.load_gold_set(write_gold(tmp_path, rows))


def test_repeated_query_id_rejected(tmp_path):
    """Post-edited text is keyed by query id: a duplicate would score both rows against one."""
    with pytest.raises(ValueError, match='repeats the query_id'):
        tm_gold.load_gold_set(write_gold(tmp_path, [ROW, {**ROW}]))


def test_blank_query_id_falling_back_to_line_number_cannot_collide(tmp_path):
    rows = [{**ROW, 'query_id': '2'}, {**ROW, 'query_id': ''}]
    with pytest.raises(ValueError, match='repeats the query_id'):
        tm_gold.load_gold_set(write_gold(tmp_path, rows))


def test_submission_parameters_carry_clean_language_codes(tmp_path):
    gold = tm_gold.load_gold_set(write_gold(tmp_path, [ROW]))
    parameters = gold.submissions()[0].parameters
    assert parameters['clean_target_language_code'] == 'zh-tw'
    assert parameters['cat_project_id'] == 'R-0322-23'


def test_row_missing_reference_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='reference'):
        tm_gold.load_gold_set(write_gold(tmp_path, [{**ROW, 'reference': ''}]))


BANDING = {'fuzzy_floor': 0.75, 'semantic_floor': 0.80}


def test_100_percent_match_bands_exact():
    assert tm_match.classify(char=1.0, semantic=None, **BANDING) == tm_match.EXACT


def test_semantic_is_residual_band():
    """A candidate the character channel reaches is banded on characters, never on meaning."""
    assert tm_match.classify(char=0.9, semantic=0.99, **BANDING) == tm_match.FUZZY
    assert tm_match.classify(char=0.4, semantic=0.9, **BANDING) == tm_match.SEMANTIC


def test_candidate_below_both_floors_gets_no_band():
    assert tm_match.classify(char=0.4, semantic=0.5, **BANDING) is None


def test_best_candidate_chosen_by_band_before_score():
    def candidate(entry_id, band, char):
        return tm_score.Candidate(
            entry=tm_index.Entry(entry_id=entry_id, source='s', target='t'),
            grade=3, char=char, band=band)

    best = tm_score._best([
        candidate('fuzzy-99', tm_match.FUZZY, 0.99),
        candidate('exact', tm_match.EXACT, 1.0),
        candidate('semantic', tm_match.SEMANTIC, 0.95),
    ])
    assert best.entry_id == 'exact'


FLOORS = {'reference_floor': 0.90, 'semantic_floor': 0.85}


def test_identical_text_is_applied():
    evidence = tm_match.match_evidence('Mettez la machine hors tension.',
                                       'Mettez la machine hors tension.', **FLOORS)
    assert evidence.tier == tm_match.TIER_EXACT
    assert evidence.identical
    assert evidence.verdict == tm_match.APPLIED


def test_casing_drift_is_near_verbatim_not_applied():
    evidence = tm_match.match_evidence('mettez la machine hors tension.',
                                       'Mettez la machine hors tension.', **FLOORS)
    assert evidence.matched and not evidence.identical
    assert evidence.verdict == tm_match.NEAR_VERBATIM


def test_whitespace_alone_does_not_break_applied():
    evidence = tm_match.match_evidence('Mettez  la machine\nhors tension.',
                                       'Mettez la machine hors tension.', **FLOORS)
    assert evidence.verdict == tm_match.APPLIED


def test_edited_entry_above_floor_is_adapted():
    evidence = tm_match.match_evidence(
        'Mettez la machine hors tension avant toute intervention de maintenance lourde.',
        'Mettez la machine hors tension avant toute intervention de maintenance.', **FLOORS)
    assert evidence.tier == tm_match.TIER_FUZZY
    assert evidence.verdict == tm_match.ADAPTED


def test_short_segment_falls_below_floor_on_small_edit():
    """The floor is length-sensitive: the same edit that is noise in a long segment is decisive in
    a short one."""
    evidence = tm_match.match_evidence('Mettez la machine hors tension avant tout.',
                                       'Mettez la machine hors tension.', **FLOORS)
    assert not evidence.matched
    assert 0.75 < evidence.score < 0.90


def test_unrelated_text_without_embedder_reports_unavailable():
    """Unanswerable is not the same as answered no."""
    evidence = tm_match.match_evidence('Something else entirely.',
                                       'Mettez la machine hors tension.', **FLOORS)
    assert not evidence.matched
    assert evidence.tier == tm_match.TIER_UNAVAILABLE
    assert not evidence.conclusive


def test_semantic_tier_answers_when_embedder_given():
    def embedder(texts):
        return [[1.0, 0.0], [1.0, 0.02]]

    evidence = tm_match.match_evidence('Something else entirely.',
                                       'Mettez la machine hors tension.',
                                       embedder=embedder, **FLOORS)
    assert evidence.matched and evidence.tier == tm_match.TIER_SEMANTIC
    assert evidence.verdict == tm_match.ADAPTED


def test_semantic_tier_answers_no_conclusively():
    def embedder(texts):
        return [[1.0, 0.0], [0.0, 1.0]]

    evidence = tm_match.match_evidence('a', 'b', embedder=embedder, **FLOORS)
    assert not evidence.matched and evidence.tier == tm_match.TIER_NONE
    assert evidence.conclusive


def test_empty_side_never_matches():
    assert not tm_match.match_evidence('', 'something', **FLOORS).matched


def test_exact_match_must_be_applied():
    assert tm_match.is_compliant(tm_match.EXACT, tm_match.APPLIED)
    assert tm_match.is_compliant(tm_match.EXACT, tm_match.NEAR_VERBATIM)
    assert not tm_match.is_compliant(tm_match.EXACT, tm_match.ADAPTED)


def test_fuzzy_match_must_be_adapted():
    assert tm_match.is_compliant(tm_match.FUZZY, tm_match.ADAPTED)
    assert not tm_match.is_compliant(tm_match.FUZZY, tm_match.APPLIED)


def test_violations_are_named():
    assert tm_match.violation(tm_match.FUZZY, tm_match.APPLIED) == 'unedited_fuzzy'
    assert tm_match.violation(tm_match.SEMANTIC, tm_match.APPLIED) == 'unedited_semantic'
    assert tm_match.violation(tm_match.UNBANDED, tm_match.APPLIED) == 'unedited_unbanded'
    assert tm_match.violation(tm_match.EXACT, tm_match.ADAPTED) == 'edited_exact'
    assert tm_match.violation(tm_match.EXACT, tm_match.NOT_USED) == 'not_used'
    assert tm_match.violation(tm_match.EXACT, tm_match.APPLIED) is None


class FakeSearch:
    """Stands in for the transport: the ids it holds, and the ones it does not."""

    def __init__(self, documents: dict[str, dict]) -> None:
        self.documents = documents
        self.calls: list[list[str]] = []

    def mget(self, index, ids):
        self.calls.append(list(ids))
        return [
            {'_id': entry_id, 'found': entry_id in self.documents,
             '_source': self.documents.get(entry_id, {})}
            for entry_id in ids
        ]


def tm_document(**overrides):
    return {
        'source_text': 'No sobrepasar la carga máxima.',
        'target_text': 'Nicht die Maximalbeladung überschreiten.',
        'source_lang': 'es-ES', 'target_lang': 'de-DE',
        **overrides,
    }


def client(documents):
    return tm_index.TmIndexClient(FakeSearch(documents), 'benchmark_tm')


def test_entries_come_back_keyed_by_id():
    entries, report = client({'a': tm_document()}).fetch_entries_by_id(['a'])
    assert entries['a'].source.startswith('No sobrepasar')
    assert report.found == 1 and report.missing == 0


def test_id_index_does_not_hold_is_reported_not_raised():
    entries, report = client({'a': tm_document()}).fetch_entries_by_id(['a', 'ghost'])
    assert set(entries) == {'a'}
    assert report.missing == 1


def test_entry_in_another_language_is_not_usable():
    entries, report = client({'a': tm_document(target_lang='fr-FR')}).fetch_entries_by_id(
        ['a'], source_language='es-ES', target_language='de-DE')
    assert entries == {}
    assert report.wrong_language == 1


def test_regional_variant_still_matches():
    entries, _ = client({'a': tm_document()}).fetch_entries_by_id(
        ['a'], source_language='es-419', target_language='de-AT')
    assert set(entries) == {'a'}


def test_entry_with_empty_target_is_not_usable():
    entries, report = client({'a': tm_document(target_text='   ')}).fetch_entries_by_id(['a'])
    assert entries == {} and report.empty_text == 1


def test_ids_are_fetched_in_batches():
    search = FakeSearch({str(n): tm_document() for n in range(5)})
    tm_index.TmIndexClient(search, 'benchmark_tm', batch_size=2).fetch_entries_by_id(
        [str(n) for n in range(5)])
    assert [len(call) for call in search.calls] == [2, 2, 1]


@dataclass(frozen=True)
class FakeTmConfig:
    """The floors the assertions below are written against. The real Config reads `.env`, so a
    documented setting on the developer's machine would silently rewrite these expectations."""

    fuzzy_floor: float = 0.75
    semantic_floor: float = 0.80
    reference_floor: float = 0.90
    reference_semantic_floor: float = 0.85
    partial_floor: float = 0.95
    length_guard: float = 0.5
    min_relevance: int = 0
    hard_negatives: bool = True


@dataclass(frozen=True)
class FakeBenchmarkConfig:
    batch_size: int = 10


@dataclass(frozen=True)
class FakeConfig:
    tm: FakeTmConfig = field(default_factory=FakeTmConfig)
    benchmark: FakeBenchmarkConfig = field(default_factory=FakeBenchmarkConfig)


class FakePostMt:
    """Returns each segment with an APE text taken from a table, keyed by segment id."""

    def __init__(self, edited: dict[str, str]) -> None:
        self.edited = edited

    def run(self, *, parameters, segments, steps, on_progress=None):
        from sourcecode.postmt import Usage

        class Result:
            error = None
            usage = Usage()

        result = Result()
        result.segments = [
            {**segment, 'ape_results': {'text': self.edited.get(segment['source_segment_id'], '')}}
            for segment in segments
        ]
        return result


def gold_rows():
    """Two segments: one the human applied verbatim, one the TM could not serve."""
    return [
        {'query_id': 'q1', 'cat_project_id': 'P-1',
         'source': 'Switch off the machine before any maintenance work.',
         'raw_mt': 'Eteignez la machine avant tout travail de maintenance.',
         'reference': 'Mettez la machine hors tension avant toute intervention.',
         'source_language': 'en-GB', 'target_language': 'fr-FR', 'domain': 'forestry',
         'tm_match': {'e1': 3}, 'hard_negatives': ['neg-1']},
        {'query_id': 'q2', 'cat_project_id': 'P-1',
         'source': 'Nothing in the memory is close to this sentence at all.',
         'raw_mt': 'Rien dans la memoire n est proche de cette phrase.',
         'reference': 'Rien dans la mémoire ne ressemble à cette phrase.',
         'source_language': 'en-GB', 'target_language': 'fr-FR', 'domain': 'forestry',
         'tm_match': {}, 'hard_negatives': []},
    ]


def benchmark_entries():
    return {
        'e1': tm_document(
            source_text='Switch off the machine before any maintenance work.',
            target_text='Mettez la machine hors tension avant toute intervention.',
            source_lang='en-GB', target_lang='fr-FR'),
        'neg-1': tm_document(
            source_text='Switch on the machine after maintenance.',
            target_text='Remettez la machine sous tension après la maintenance.',
            source_lang='en-GB', target_lang='fr-FR'),
    }


def run_benchmark(tmp_path, edited=None, skip_pipeline=False, rows=None, config=None,
                  entries=None, embedder=None):
    gold = tm_gold.load_gold_set(write_gold(tmp_path, rows or gold_rows()))
    return tm_benchmark.TmBenchmark(
        postmt=None if skip_pipeline else FakePostMt(edited or {}),
        index=client(entries or benchmark_entries()),
        config=config or FakeConfig(),
        embedder=embedder,
    ).run(gold, skip_pipeline=skip_pipeline)


def test_dry_run_scores_reference_alone(tmp_path):
    """The baseline figure needs no post-mt call at all."""
    result = run_benchmark(tmp_path, skip_pipeline=True)
    assert result.scored_versions == (tm_score.REF,)
    assert result.aggregate.sampled == 2
    assert result.aggregate.declared == 1
    assert result.aggregate.reference_rate(tm_score.REF) == 1.0


def test_not_listed_separated_from_not_fetched(tmp_path):
    result = run_benchmark(tmp_path, skip_pipeline=True)
    assert result.aggregate.sampled - result.aggregate.declared == 1
    assert result.aggregate.not_fetched == 0
    assert result.aggregate.below_relevance == 0
    assert result.aggregate.eligibility_rate == 0.5


def test_relevance_threshold_not_reported_as_index_failure(tmp_path):
    """A grade the operator excluded is their threshold at work, not an entry the index lost."""
    config = FakeConfig(tm=FakeTmConfig(min_relevance=4))
    result = run_benchmark(tmp_path, skip_pipeline=True, config=config)

    assert result.aggregate.not_fetched == 0
    assert result.aggregate.below_relevance == 1
    rendered = tm_report.tm_scorecard(result).as_markdown()
    assert 'TM_MIN_RELEVANCE' in rendered
    assert 'the index could not deliver' not in rendered


def test_entry_rejected_for_one_pair_stays_out_of_another(tmp_path):
    """Two pairs cannot share a gold set, so the id can only leak across two runs of one index."""
    gold = tm_gold.load_gold_set(write_gold(tmp_path, gold_rows()))
    by_pair, report = tm_benchmark.TmBenchmark(
        postmt=None, index=client(benchmark_entries()), config=FakeConfig()
    ).fetch(gold)

    assert set(by_pair) == {('en-GB', 'fr-FR')}
    assert set(by_pair[('en-GB', 'fr-FR')]) == {'e1', 'neg-1'}
    assert report.found == 2


def test_human_applied_exact_match(tmp_path):
    result = run_benchmark(tmp_path, skip_pipeline=True)
    scored = next(s for s in result.segments if s.query_id == 'q1')
    assert scored.band == tm_match.EXACT
    assert scored.versions[tm_score.REF].verdict == tm_match.APPLIED
    assert all(use.compliant for use in scored.versions[tm_score.REF].uses)


def test_version_taking_weaker_entry_violates_under_that_entry_band(tmp_path):
    """The band on offer is exact and the rule applied is fuzzy's. The segment counts on the exact
    row, so an ignored exact stays visible there; the entry and its violation count on the fuzzy
    row, so the fuzzy compliance rate is the pass rate of the rule that was actually applied."""
    rows = [{
        'query_id': 'q1', 'cat_project_id': 'P-1',
        'source': 'Switch off the machine before any maintenance work.',
        'raw_mt': 'Eteignez la machine avant tout travail de maintenance.',
        'reference': 'Mettez la machine hors tension avant toute intervention.',
        'source_language': 'en-GB', 'target_language': 'fr-FR', 'domain': 'forestry',
        'tm_match': {'strong': 3, 'weak': 2}, 'hard_negatives': [],
    }]
    entries = {
        'strong': tm_document(
            source_text='Switch off the machine before any maintenance work.',
            target_text='Mettez la machine hors tension avant tout travail de maintenance.',
            source_lang='en-GB', target_lang='fr-FR'),
        'weak': tm_document(
            source_text='Switch off the machine before maintenance.',
            target_text='Mettez la machine hors tension avant toute intervention.',
            source_lang='en-GB', target_lang='fr-FR'),
    }
    result = run_benchmark(tmp_path, skip_pipeline=True, rows=rows, entries=entries)

    scored = result.segments[0]
    assert scored.band == tm_match.EXACT
    reference = scored.versions[tm_score.REF]
    assert reference.violation == 'unedited_fuzzy'
    assert reference.matched_lesser

    bands = result.aggregate.bands
    assert bands[tm_match.EXACT].eligible == 1
    assert bands[tm_match.EXACT].used[tm_score.REF] == 1
    assert bands[tm_match.EXACT].reference_rate(tm_score.REF) == 1.0
    assert bands[tm_match.EXACT].entries[tm_score.REF] == 0
    assert bands[tm_match.EXACT].compliance_rate(tm_score.REF) is None
    assert bands[tm_match.FUZZY].eligible == 0
    assert bands[tm_match.FUZZY].entries[tm_score.REF] == 1
    assert bands[tm_match.FUZZY].compliance_rate(tm_score.REF) == 0.0
    assert bands[tm_match.FUZZY].violations[tm_score.REF]['unedited_fuzzy'] == 1


def test_ape_that_ignores_entry_costs_reference_rate_not_compliance(tmp_path):
    """Declining a match is a translator's choice, so it lowers the reference rate and leaves the
    compliance rate without a denominator rather than scoring zero on it."""
    result = run_benchmark(tmp_path, edited={'q1': 'Arrêtez la machine avant l entretien.'})
    tally = result.aggregate.bands[tm_match.EXACT]
    assert tally.used[tm_score.REF] == 1
    assert tally.used[tm_score.APE] == 0
    assert tally.reference_rate(tm_score.APE) == 0.0
    assert tally.compliance_rate(tm_score.APE) is None
    assert tally.ref_only == 1
    assert result.aggregate.carry_over_rate == 0.0


def test_ape_that_reproduces_entry_carries_over(tmp_path):
    result = run_benchmark(
        tmp_path, edited={'q1': 'Mettez la machine hors tension avant toute intervention.'})
    assert result.aggregate.carry_over_rate == 1.0
    assert result.aggregate.bands[tm_match.EXACT].compliant[tm_score.APE] == 1


def test_hard_negative_does_not_register_as_match(tmp_path):
    result = run_benchmark(tmp_path, skip_pipeline=True)
    assert result.calibration.tested == 1
    assert result.calibration.matched == 0
    assert result.calibration.false_match_rate == 0.0


def test_fetched_entry_count_reaches_result(tmp_path):
    result = run_benchmark(tmp_path, skip_pipeline=True)
    assert result.totals['entries_fetched'] == 2


def test_scorecard_says_pipeline_has_no_tm(tmp_path):
    result = run_benchmark(tmp_path, skip_pipeline=True)
    rendered = tm_report.tm_scorecard(result).as_markdown()
    assert 'does not retrieve TM matches' in rendered
    assert 'ICE is not reported' in rendered


def test_band_table_has_no_ice_row_and_keeps_unbanded(tmp_path):
    rendered = tm_report.render_tm_bands(run_benchmark(tmp_path, skip_pipeline=True))
    assert '| ICE |' not in rendered
    assert '| exact |' in rendered and '| unbanded |' in rendered


def test_every_block_renders(tmp_path):
    result = run_benchmark(tmp_path, edited={'q1': 'Mettez la machine hors tension.'})
    for block in (
        tm_report.tm_scorecard(result).as_console(),
        tm_report.render_tm_bands(result, console=True),
        tm_report.render_tm_compliance(result, console=True),
        tm_report.render_tm_segments_console(result),
        tm_report.render_tm_strata([result], console=True),
        tm_report.tm_scorecard(result).as_markdown(),
        tm_report.render_tm_bands(result),
        tm_report.render_tm_compliance(result),
        tm_report.render_tm_calibration(result),
        tm_report.render_tm_segments(result),
        tm_report.render_tm_strata([result]),
    ):
        assert block.strip()


def test_unmeasured_semantic_band_never_prints_zero(tmp_path):
    """No embedder is wired, so the row is unmeasured - a 0 there would read as a finding."""
    result = run_benchmark(tmp_path, skip_pipeline=True)
    assert not result.semantic_measured

    for rendered in (tm_report.render_tm_bands(result), tm_report.render_tm_compliance(result)):
        semantic = next(line for line in rendered.splitlines() if line.startswith('| semantic |'))
        assert 'unmeasured' in semantic
        assert '| 0 |' not in semantic

    scorecard = tm_report.tm_scorecard(result).as_markdown()
    assert 'semantic unmeasured' in scorecard
    assert 'was not measured' in scorecard


def test_semantic_band_measured_once_embedder_given(tmp_path):
    """The floors are live settings the moment something can answer them, not decoration."""
    def embedder(texts):
        return [[1.0, 0.0] for _ in texts]

    result = run_benchmark(tmp_path, skip_pipeline=True, embedder=embedder)
    assert result.semantic_measured
    assert all(
        candidate.semantic is not None
        for segment in result.segments for candidate in segment.candidates
    )
    assert 'unmeasured' not in tm_report.render_tm_bands(result)


def test_unfinished_match_test_named_beside_rate(tmp_path):
    """With no embedder a `not_used` rests on characters alone, so the rate is a lower bound."""
    rows = [{
        'query_id': 'q1', 'cat_project_id': 'P-1',
        'source': 'Switch off the machine before any maintenance work.',
        'raw_mt': 'Eteignez la machine avant tout travail de maintenance.',
        'reference': 'Rien a voir avec cette entree.',
        'source_language': 'en-GB', 'target_language': 'fr-FR', 'domain': 'forestry',
        'tm_match': {'e1': 3}, 'hard_negatives': [],
    }]
    result = run_benchmark(tmp_path, skip_pipeline=True, rows=rows)

    assert result.aggregate.inconclusive(tm_score.REF) == 1
    assert 'lower bound' in tm_report.tm_scorecard(result).as_markdown()


def test_segments_listed_worst_first(tmp_path):
    """The console shows the head of the table, so the worst has to be at the head."""
    result = run_benchmark(tmp_path, edited={'q1': 'Arretez la machine avant l entretien.'})
    rows = tm_report.segment_rows(result)
    assert [row['query_id'] for row in rows] == ['q1', 'q2']
    assert 'worst first' in tm_report.render_tm_segments(result)


class FakeProxy:
    """Stands in for LiteLLM: records what each call asked for, and answers out of order."""

    def __init__(self, data=None, error=None) -> None:
        self.batches = []
        self.data = data
        self.error = error

    def embedding(self, *, model, input, **kwargs):
        self.batches.append(list(input))
        if self.error:
            raise self.error
        if self.data is not None:
            return {'data': self.data}
        answer = [{'index': number, 'embedding': [float(len(text)), 1.0]}
                  for number, text in enumerate(input)]
        return {'data': list(reversed(answer))}


def embedder(monkeypatch, proxy):
    monkeypatch.setitem(sys.modules, 'litellm', types.SimpleNamespace(embedding=proxy.embedding))
    return tm_embed.TmEmbedder('https://litellm.test', 'azure/azure/text-embedding-3-small')


def test_embedder_keeps_vector_order(monkeypatch):
    """The proxy is not guaranteed to answer in order, so `index` pairs a vector with its text."""
    vectors = embedder(monkeypatch, FakeProxy())(['a', 'bb', 'ccc'])
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]


def test_embedder_embeds_each_text_once(monkeypatch):
    """One version's text is tested against every candidate, and each re-embedding is a call."""
    proxy = FakeProxy()
    embed = embedder(monkeypatch, proxy)
    embed(['ref', 'entry-1'])
    embed(['ref', 'entry-2'])
    assert proxy.batches == [['ref', 'entry-1'], ['entry-2']]


def test_embedder_rejects_incomplete_answer(monkeypatch):
    """A short answer would pair vectors with the wrong texts, so it raises rather than zips."""
    proxy = FakeProxy(data=[{'index': 0, 'embedding': [1.0]}])
    with pytest.raises(ValueError):
        embedder(monkeypatch, proxy)(['a', 'b'])


def test_embedder_ping_reports_unusable_proxy(monkeypatch):
    proxy = FakeProxy(error=RuntimeError('401 Unauthorized'))
    assert embedder(monkeypatch, proxy).ping() is False


PARTIAL = {'floor': 0.95}

FIRST = 'Mettez la machine hors tension avant toute intervention.'
SECOND = 'Portez des gants de protection en permanence.'


def test_entry_found_inside_version_is_a_partial_match():
    evidence = tm_match.contained_evidence(f'{FIRST} {SECOND}', SECOND, **PARTIAL)
    assert evidence.matched and evidence.tier == tm_match.TIER_PARTIAL
    assert evidence.verdict == tm_match.APPLIED
    assert evidence.span == (len(FIRST) + 1, len(FIRST) + 1 + len(SECOND))


def test_rewritten_window_is_adapted_not_applied():
    text = f'{FIRST} Portez les gants de protection en permanence.'
    evidence = tm_match.contained_evidence(text, SECOND, **PARTIAL)
    assert evidence.matched and not evidence.identical
    assert evidence.verdict == tm_match.ADAPTED


def test_window_below_floor_does_not_match():
    text = f'{FIRST} Portez des gants de protection a tout moment.'
    assert not tm_match.contained_evidence(text, SECOND, **PARTIAL).matched


def test_entry_longer_than_version_is_never_contained():
    """Containment is the shorter text inside the longer one; the other way round is the whole test."""
    evidence = tm_match.contained_evidence(SECOND, f'{FIRST} {SECOND}', **PARTIAL)
    assert not evidence.matched and evidence.span is None


def stitched_rows():
    """One segment answered by two entries, each covering one of its sentences."""
    return [{
        'query_id': 'q1', 'cat_project_id': 'P-1',
        'source': 'Switch off the machine before any maintenance work. '
                  'Wear protective gloves at all times.',
        'raw_mt': 'Eteignez la machine avant tout travail. Portez des gants.',
        'reference': f'{FIRST} {SECOND}',
        'source_language': 'en-GB', 'target_language': 'fr-FR', 'domain': 'forestry',
        'tm_match': {'first': 3, 'second': 3}, 'hard_negatives': [],
    }]


def stitched_entries(**extra):
    return {
        'first': tm_document(
            source_text='Switch off the machine before any maintenance work.',
            target_text=FIRST, source_lang='en-GB', target_lang='fr-FR'),
        'second': tm_document(
            source_text='Wear protective gloves at all times.',
            target_text=SECOND, source_lang='en-GB', target_lang='fr-FR'),
        **extra,
    }


def test_entry_covering_part_of_source_is_banded_on_that_part(tmp_path):
    """Against the whole SRC it reaches no floor; against the sentence it answers it is exact."""
    result = run_benchmark(tmp_path, skip_pipeline=True, rows=stitched_rows(),
                           entries=stitched_entries())
    candidates = result.segments[0].candidates

    assert all(candidate.partial for candidate in candidates)
    assert all(candidate.band == tm_match.EXACT for candidate in candidates)
    assert all(candidate.char < 0.75 < candidate.covered for candidate in candidates)
    assert result.segments[0].band == tm_match.EXACT


def test_both_entries_serving_one_segment_are_credited(tmp_path):
    result = run_benchmark(tmp_path, skip_pipeline=True, rows=stitched_rows(),
                           entries=stitched_entries())
    reference = result.segments[0].versions[tm_score.REF]

    assert [use.candidate.entry_id for use in reference.uses] == ['first', 'second']
    assert reference.stitched and not reference.whole_match
    assert all(use.verdict == tm_match.APPLIED for use in reference.uses)
    assert all(use.compliant for use in reference.uses)

    tally = result.aggregate.bands[tm_match.EXACT]
    # One segment used an entry, out of two entries used.
    assert tally.used[tm_score.REF] == 1
    assert tally.entries[tm_score.REF] == 2
    assert tally.stitched[tm_score.REF] == 1


def test_two_entries_cannot_be_credited_with_the_same_words(tmp_path):
    """A near duplicate of an entry already used covers words that are spoken for."""
    rows = stitched_rows()
    rows[0]['tm_match'] = {'first': 3, 'second': 3, 'duplicate': 3}
    entries = stitched_entries(duplicate=tm_document(
        source_text='Switch off the machine before maintenance work.',
        target_text='Mettez la machine hors tension avant toute intervention !',
        source_lang='en-GB', target_lang='fr-FR'))

    reference = run_benchmark(
        tmp_path, skip_pipeline=True, rows=rows, entries=entries,
    ).segments[0].versions[tm_score.REF]

    assert [use.candidate.entry_id for use in reference.uses] == ['first', 'second']


def test_whole_text_match_leaves_no_room_for_a_second_entry(tmp_path):
    """One entry explaining the whole version is the answer it was before, unchanged."""
    reference = next(
        segment for segment in run_benchmark(tmp_path, skip_pipeline=True).segments
        if segment.query_id == 'q1'
    ).versions[tm_score.REF]

    assert len(reference.uses) == 1 and reference.whole_match
    assert not reference.stitched


MOSTLY = ('Mettez la machine hors tension et consignez le sectionneur avant toute '
          'intervention de maintenance sur les organes mobiles.')
TAIL = 'Portez des gants.'


def lopsided_rows():
    """One segment answered by two entries, one of them covering almost all of it."""
    return [{
        'query_id': 'q1', 'cat_project_id': 'P-1',
        'source': 'Switch off the machine and lock out the isolator before any maintenance work '
                  'on moving parts. Wear gloves.',
        'raw_mt': 'Eteignez la machine avant la maintenance. Portez des gants.',
        'reference': f'{MOSTLY} {TAIL}',
        'source_language': 'en-GB', 'target_language': 'fr-FR', 'domain': 'forestry',
        'tm_match': {'mostly': 3, 'tail': 3}, 'hard_negatives': [],
    }]


def lopsided_entries():
    return {
        'mostly': tm_document(
            source_text='Switch off the machine and lock out the isolator before any maintenance '
                        'work on moving parts.',
            target_text=MOSTLY, source_lang='en-GB', target_lang='fr-FR'),
        'tail': tm_document(
            source_text='Wear gloves.',
            target_text=TAIL, source_lang='en-GB', target_lang='fr-FR'),
    }


def test_dominant_entry_really_does_clear_the_whole_text_floor():
    """The premise of the test below: covering 82% of a segment is enough to match the whole of it
    on characters, so the whole-text pass fires on an entry that answers only part of the text."""
    reference = f'{MOSTLY} {TAIL}'
    assert tm_match.char_score(reference, MOSTLY) >= FakeTmConfig.reference_floor
    assert len(MOSTLY) < len(reference)


def test_entry_covering_most_of_the_segment_does_not_swallow_the_rest(tmp_path):
    """A whole-text match is not proof the entry accounts for every word: where another entry
    answers the part it leaves free, both were used and both are credited."""
    result = run_benchmark(tmp_path, skip_pipeline=True, rows=lopsided_rows(),
                           entries=lopsided_entries())
    reference = result.segments[0].versions[tm_score.REF]

    assert [use.candidate.entry_id for use in reference.uses] == ['mostly', 'tail']
    assert reference.stitched and not reference.whole_match
    assert all(use.evidence.tier == tm_match.TIER_PARTIAL for use in reference.uses)


def test_dominant_entry_alone_still_takes_the_whole_segment(tmp_path):
    """With no second entry to complete it, the same match is the single whole-text use it was."""
    rows = lopsided_rows()
    rows[0]['tm_match'] = {'mostly': 3}
    reference = run_benchmark(
        tmp_path, skip_pipeline=True, rows=rows, entries=lopsided_entries(),
    ).segments[0].versions[tm_score.REF]

    assert [use.candidate.entry_id for use in reference.uses] == ['mostly']
    assert not reference.stitched


WHOLE = ('Portez des gants de protection et un casque avant toute intervention sur la machine.')
EDITED = ('Portez des gants de protection et un casque avant chaque intervention sur la machine.')
FRAGMENT = 'Portez des gants de protection'


def swallowed_rows():
    """One segment adapted from one entry, with a short second entry sitting verbatim inside it."""
    return [{
        'query_id': 'q1', 'cat_project_id': 'P-1',
        'source': 'Wear protective gloves and a helmet before each intervention on the machine.',
        'raw_mt': 'Portez des gants et un casque avant chaque intervention.',
        'reference': EDITED,
        'source_language': 'en-GB', 'target_language': 'fr-FR', 'domain': 'forestry',
        'tm_match': {'whole': 3, 'fragment': 3}, 'hard_negatives': [],
    }]


def swallowed_entries():
    return {
        'whole': tm_document(
            source_text='Wear protective gloves and a helmet before any work on the machine.',
            target_text=WHOLE, source_lang='en-GB', target_lang='fr-FR'),
        'fragment': tm_document(
            source_text='Wear protective gloves',
            target_text=FRAGMENT, source_lang='en-GB', target_lang='fr-FR'),
    }


def test_fragment_really_does_outscore_the_entry_the_segment_was_built_from():
    """The premise of the test below: a verbatim window scores 1.0 by construction, above the score
    of the whole-text match the version was actually adapted from."""
    window = tm_match.contained_score(EDITED, FRAGMENT)
    assert window == 1.0 > tm_match.char_score(EDITED, WHOLE) >= FakeTmConfig.reference_floor


def test_verbatim_fragment_does_not_outrank_the_segment_it_sits_inside(tmp_path):
    """Claims are settled on the characters they explain, not on their score alone: a short entry
    cannot claim more words than it has, so it never displaces the entry that covers the segment."""
    result = run_benchmark(tmp_path, skip_pipeline=True, rows=swallowed_rows(),
                           entries=swallowed_entries())
    reference = result.segments[0].versions[tm_score.REF]

    assert [use.candidate.entry_id for use in reference.uses] == ['whole']
    assert not reference.stitched
    assert reference.verdict == tm_match.ADAPTED


def test_hard_negative_found_as_a_window_is_a_false_match(tmp_path):
    """The negatives are tested the way the uses are found, or they calibrate a test nothing runs."""
    rows = stitched_rows()
    rows[0]['hard_negatives'] = ['neg-1']
    entries = stitched_entries(**{'neg-1': tm_document(
        source_text='Switch on the machine after maintenance.',
        target_text='la machine hors tension avant toute intervention.',
        source_lang='en-GB', target_lang='fr-FR')})

    calibration = run_benchmark(
        tmp_path, skip_pipeline=True, rows=rows, entries=entries,
    ).calibration

    assert calibration.tested == 1 and calibration.matched == 1
    assert calibration.by_tier[tm_match.TIER_PARTIAL].matched == 1


def test_worklist_names_every_entry_the_segment_used(tmp_path):
    result = run_benchmark(tmp_path, skip_pipeline=True, rows=stitched_rows(),
                           entries=stitched_entries())
    row = tm_report.segment_rows(result)[0]

    assert row['entry'] == 'first+second'
    assert 'stitched' in row['flags'] and 'partial' in row['flags']
    assert 'more than one entry' in tm_report.tm_scorecard(result).as_markdown()
