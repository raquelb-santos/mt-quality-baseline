# mt-quality-baseline

Benchmarks the translation quality of the `post-mt` pipeline.

A run takes segments with their source, their raw MT and the human translation delivered for them,
sends them through post-mt, and scores the translation against the human reference.

| Name    | The text                                                    |
| ------- | ----------------------------------------------------------- |
| **SRC** | the source-language original that was sent to MT            |
| **MT**  | the raw machine translation, the baseline post-mt was given |
| **APE** | the post-edited output post-mt returned                     |
| **REV** | the reverted output the DNT service returned (DNT only)     |
| **REF** | the human translation delivered for the segment             |

Three components are implemented:

**Terminology adherence** — how often each version uses the glossary terms the human used. **MT**
and **APE** are both scored against **REF**.

**DNT preservation** — how often each version keeps the items that must not be translated. **APE**
is sent to the DNT service's **revert endpoint**, which returns both the items it weighed in a
segment and **REV**. **MT**, **APE** and **REV** are each scored against **REF**.

**TM reference** — how often each version reuses the translation memory entries the segment had,
and whether that reuse is what the match asks for: an exact match is to be kept, a weaker one
edited. The entries are fetched from the **TM index** and banded on how
close each entry's source is to **SRC**. **REF** and **APE** are each scored against them.

It reports each version's score and the **delta** from post-editing, **repairs and regressions**, a
worst-first **per-item worklist** tracing numbers back to segments, and **pooling by stratum**
across datasets.

## Quick start

Terminology reads the glossary from the term-bases index post-mt itself queries, so it needs
`SEARCH_ENGINE_URL` (and AWS SigV4 for the dev domain) in a `.env`.

DNT gets its items from the DNT service, so
it needs `DNT_BASE_URL` and `DNT_API_KEY`.

TM fetches its entries from the TM index on the same
cluster, so it needs `SEARCH_ENGINE_URL` and `TM_INDEX` beside its floors.

```bash
# scores the configured dataset, post-editing each segment through post-mt
python sourcecode/run.py

# the same, but scores MT only — never calls post-mt
python sourcecode/run.py --dry-run
```

`pip install -e .` also puts the same entry point on the path as `mt-quality-baseline`.

What a run scores is configured in `.env`, as `GLOSSARY_PATH`, `DNT_PATH` and `TM_PATH`. Each may
name a dataset file or a folder of them (`.json`,
`.csv`, `.mxliff`, `.xliff`, `.xlf`). A file that cannot describe itself needs a `<name>.params.json`
beside it giving its own `parameters`, `steps` and `glossary_ids`.
Components are configured separately because pooling adds counts within a stratum.

A run prints each component's scorecard as it is measured and writes one Markdown report under
`reports/`, named for what was evaluated and when it ran — `glossary+dnt_20260826-142207.md`, with
`_dry-run` in the name when it was one.

## How it works

```
 one component's dataset: SRC · MT · REF · the component's own fields
     │
     ├─ 1. run segments ──────────────────────► post-mt  POST /api/workflow/async
     │      (REF stripped from payload)                  GET  /api/workflow/async/:id
     │      returns: MT, the baseline it was given
     │               APE, the post-edited text
     │               per-segment step failures
     │               whether a glossary reached the model
     │
     ├─ 2. score every version against REF, with the dataset's component
     │      terminology ─ a. lemmatize ───────► Stanza        (as post-mt does)
     │                    b. resolve terms ───► term-bases    (same query post-mt sends)
     │                         (percolate each source against the pinned glossary
     │                          ids, then follow the hit's concept id to the
     │                          target term the segment is held to)
     │                    c. count each term in REF, then in MT and APE
     │
     │      DNT ───────── a. detect and revert ► DNT service  POST /v1/revert
     │                         (one call returns the items it weighed and REV,
     │                          a third version to score)
     │                    b. count each item in SRC and REF,
     │                       then in MT, APE and REV
     │
     │      TM ────────── a. fetch the entries the row lists ────► benchmark_tm
     │                    b. band each on the recomputed character score
     │                    c. check if REF and APE used any of them,
     │                       and whether that use is what the band expects
     │
     └─ 3. aggregate, pool by stratum, write the report
```

Everything about the translation itself comes from post-mt's own API, so the benchmark measures the
pipeline that actually runs rather than a reimplementation of it.

***

## Terminology adherence

Terms come from the **term-bases index**, selected with the same Elasticsearch query post-mt sends,
so both resolve the same terms from the same data. Which term bases are queried is pinned per dataset
as `glossary_ids`.

### Metrics

For every glossary term matched in a segment, the **REF count** is how often the target term
appears in the **human reference** — so a term used three times is not discharged by one — and the
**version count** is how often it appears in the **version being scored**, capped at the REF count.
Summed over every term in every segment:

```
expected_instances = sum of the REF counts
adherent_instances = sum of the version counts

adherence_rate     = adherent_instances / expected_instances

violations         = expected_instances - adherent_instances
```

The rate is never averaged. At every grain it is recomputed from the pooled counts, so a term
matched once cannot outweigh the same term matched forty times. It is reported again **term by
term** over that term's own REF count and its violation count.

The same terms are also counted **once each instead of per occurrence**, split four ways by how the
version count compares with REF's:

| Bucket                       | When                | Scored as                                    |
| ---------------------------- | ------------------- | -------------------------------------------- |
| **never used**               | none in the version | missed, adherence 0                          |
| **used, not everywhere**     | fewer than REF      | missed, adherence partial                    |
| **matched REF**              | as many as REF      | adherent                                     |
| **used more than the human** | more than REF       | adherent — **never a violation but flagged** |

### What is reported

Everything below is reported for **MT** and for **APE**:

* **Adherence rate** — the share of expected instances where the target term actually appears,
  computed at three grains: **per term**, **per dataset** and **per stratum**.

* **Violations** — the sum of three failures, counted over the whole corpus rather than segment by
  segment, because one of them is only visible once every segment has been read:

  * **miss** — output carries no approved target, surface form nor lemma of reference term.

  * **inconsistency** — output carries an approved target for a term the glossary proposed in
    that segment, but not the wording the reference used there.

  * **over-application** — output carries a target that the glossary did not propose in that
    segment, and the reference did not use that wording there either.

* **Violation rate** — segments carrying at least one violation, over *every* segment. Several
  violations in one segment count once.

* **Repairs and regressions** — counted separately rather than netted against each other: a
  **repair** is a term MT got wrong and APE corrected; a **regression** is a term MT got **right**
  and APE broke.

**Strict and permissive** are the two kinds of glossary instruction, scored separately.
`` `X` should be translated to: `Y` `` names one target term and is **strict**, so only `Y` counts;
`` `X` may be translated as: `Y`, `Z` `` names several and is **permissive**, so any of them does.

***

## DNT preservation

Items come from the **DNT service**: `POST /v1/revert` per batch returns both the items it weighed
in a segment and the text it produced after reverting them. Reversion is asked for over **APE**, so
**REV** is a third scored column beside MT and APE. `--dry-run` skips post-mt but not the DNT
service: reversion then runs over MT, and MT and APE read alike.

The item must appear **verbatim in SRC** and **REF** must keep it. Failing either leaves the item
flagged: **`not in SRC`** when the service named a string the source does not carry, and **`not in
REF`** when the source carries it but the human translated it.

### Metrics

For every scored item in a segment, the **REF count** is how often it appears verbatim in the
**human reference** — the number of times it had to survive — and the **version count** is how often
it appears in the **version being scored**, capped at the REF count. Summed over every item in every
segment:

```
expected_instances  = sum of the REF counts
preserved_instances = sum of the version counts

preservation_rate   = preserved_instances / expected_instances

leaked_instances    = expected_instances - preserved_instances
over_kept           = instances kept beyond the reference's own count
```

A leak can be a **case drift** — the item present but cased differently — or **translated**. An
over-keep freezes a word the human legitimately translated.

The same items counted **once each instead of per instance**, split four ways by how the version
count compares with REF's:

| Bucket          | When                | Meaning                                     |
| --------------- | ------------------- | ------------------------------------------- |
| **never kept**  | none in the version | REF kept it, the version has none of it     |
| **kept partly** | fewer than REF      | kept in one place and translated in another |
| **matched REF** | as many as REF      | preserved exactly as often as it was owed   |
| **over-kept**   | more than REF       | frozen more often than the human froze it   |

Unlike terminology, where using a term more than the human is legitimate and never scored,
**over-keeping is an error here**: it is only ever measured on items the reference did keep
somewhere, so it means the version froze an occurrence the human had translated.

### What is reported

Everything below is reported for **MT**, for **APE** and for **REV**:

* **Preservation rate** — the share of expected instances the version kept verbatim, **per item**, **per dataset** and **per stratum**.

* **Leaks and over-keeps** — counts beside the rate, each split by kind.

* **Segments clean** — the share of item-bearing segments with no leak and no over-keep in either
  direction.

* **Repairs and regressions** — what post-editing moved preservation by and then what
  reversion moved it by, with the items the next version broke and the ones it fixed counted
  separately.

***

## TM reference

Each gold row carries **SRC**, **MT** and **REF**, and names the entries referenced for its segment 
in `tm_match`, a map of entry id to relevance grade (4 the most relevant and 1 the least). Alongside 
it, `hard_negatives` lists entry ids a human annotator judged to look relevant and not be.

Four steps per segment:

1. **Fetch** the listed entries from `TM_INDEX` by id.

2. **Band** each one by comparing the entry's **source** with **SRC** on a character score. The floors decide which band it is
   reported under:

   | Band         | When                                                         |
   | ------------ | ------------------------------------------------------------ |
   | **exact**    | the character score is 100%                                  |
   | **fuzzy**    | below 100%, at or above `TM_FUZZY_FLOOR`                     |
   | **semantic** | below `TM_FUZZY_FLOOR`, reached only by embedding similarity |
   | **unbanded** | listed by the gold set, but no candidate reaches any floor   |

   A segment can be served by more than one entry, so where the whole comparison bands nothing the
   entry is banded on **the part of SRC it covers** — its best window there, at `TM_PARTIAL_FLOOR`.
   A fragment matching its own part 100% bands **exact**, counts in that band like any other, and is
   expected to be applied.

3. **Match** each version — **REF** and **APE** — against the entry's **target**, to see whether it
   was used.

4. **Judge** that use against what its band expects.

### Metrics

Two questions are asked of each version on each eligible segment. Their answers are the counts every
TM rate is built from, kept per band and per version.

A **band** compares the entry's **source** with **SRC** at `TM_FUZZY_FLOOR` and says what the TM
offered. A **tier** compares the version's text with the entry's **target** at `TM_REFERENCE_FLOOR`
and says what was done with it. The two are independent. A **fuzzy** entry copied verbatim matches
at the **exact** tier — its source was never identical, its target was — and that pairing is the
`unedited_fuzzy` violation.

**`used`: Did it use an entry?**

Every listed candidate is tested separately, against the entry's **target**. Each test is
a cascade, first tier to answer wins. Each candidate is looked for twice over, as the **whole** of
the version's text and as a **window** inside it. What a segment reports is then a single entry that
took all of the text, or the entries that between them cover its parts.

| Tier         | Matched when                                               | Verdict         |
| ------------ | ---------------------------------------------------------- | --------------- |
| **exact**    | equal            | `applied`       |
| **exact**    | equal only once casefolded | `near_verbatim` |
| **fuzzy**    | the character score reaches `TM_REFERENCE_FLOOR`            | `adapted`       |
| **semantic** | the embedding cosine reaches `TM_REFERENCE_SEMANTIC_FLOOR`  | `adapted`       |
| **partial**  | the entry's target is a window of the version at `TM_PARTIAL_FLOOR`, verbatim or edited | `applied` or `adapted` |
| —            | no candidate matched at any tier                            | `not_used`      |

Every test that answers becomes a **claim** on a span of the version's text: a whole-text match
claims all of it and a window claims only itself, and a window is only looked for where the entry's
target is **shorter** than the version. Claims are settled strongest first and a claim overlapping
one already settled is dropped, so two entries are never credited with the same words and an entry
that matched both ways is credited once. Ties keep the **gold set's own order**, most relevant grade
first and within a grade the order the entries are written in.

**A claim's weight is its score over the characters it explains**, capped for a whole-text claim at
the length of the entry's own **target** so the score is not improved by words the entry does not
contain.

**`compliant`: Did it use it the way the band asks?**

The verdict is checked against what the band expects. **exact** expects `applied` or
`near_verbatim`; **fuzzy**, **semantic** and **unbanded** expect `adapted`, because the entry
answers a source that is not the segment's **SRC**. Anything else is a violation. It is asked **per
entry** rather than per segment, so a segment stitched from several is right about the ones it got
right, and each violation is named under the band of the entry that produced it.

`not_used` is **not** among them. A grade says an entry is relevant, not that it had to be taken.
Declining a fuzzy match and translating from scratch is a translator's decision, not an error. So
non-use is measured by the reference rate, which is what it is a fact about, and left out of the
compliance rate, which is asked only of the entries a version did use. The two mislead apart — a
version that used nothing has no compliance denominator at all — so the scorecard prints them
together, each with its own count beside it.

The two rates count under different bands. The **reference rate** counts segments under the band the
TM **offered**, whichever entry the version went on to work from, so an exact match ignored in
favour of a weaker entry stays visible on the exact row and `Lesser` counts it. The **compliance
rate** counts entries under the band of the entry a version **used**, so each row is the pass rate
of that band's own rule. The denominators differ on the same row deliberately, `Eligible` being an
opportunity and `Entries` a decision.

Summed over every segment, and computed for each version scored, **APE** and **REF**:

```
eligibility_rate  = listed / sampled

reference_rate    = used / eligible           per band offered, and per grade
reference_rate    = used / eligible           overall, over every eligible segment

compliance_rate   = compliant / entries       per band used

carry_over_rate   = used by REF and by APE / used by REF
false_match_rate  = matches against a hard negative / hard negatives tested
```


### What is reported

Every rate below is reported for **APE** and **REF**. The false-match rate is a property of the gold set rather than of a version:

* **Reference rate** — the share of the segments the band was offered on in which the version used
  an entry, computed **per band**, **per dataset** and **per stratum**. Beside it the **carry-over
  rate**, the share of the segments REF used an entry in where APE used one too, and a breakdown
  **per relevance grade**: a grade-1 entry counts as an opportunity like a grade-4 one, so the split
  is what says whether a low rate is marginal entries going untaken or good ones being missed.

* **Compliance rate** — per entry used, per band, with the applied and adapted counts it is made of
  and the violations named by kind. An entry the version declined shows in the reference rate only.

* **False match rate** — what the match test does on the hard negatives, per tier.

* **Per-segment worklist** — every segment, worst first: its band, the score its best candidate
  was banded on, how many candidates it had, the entries REF used, a verdict per entry each version
  used with the tier that answered it, and its flags: `stitched` — REF used more than one entry — `partial` — an
  entry covered part of the segment only, on the source side, the target side or both — `lesser`,
  `length` — the version and the entry's target differ in length by more than `TM_LENGTH_GUARD` —
  and `no-match`.


## Configuration

Create a `.env` in the repo root with the following variables:

| Variable                                              | Purpose                                                                                                                                                                                 |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POSTMT_BASE_URL`                                     | post-mt instance to drive                                                                                                                                                               |
| `POSTMT_API_KEY`                                      | sent as `X-API-KEY`                                                                                                                                                                     |
| `STANZA_BASE_URL`                                     | Stanza lemmatizer, `https://stanza.acolad.build` — no credential                                                                                                                        |
| `SEARCH_ENGINE_URL`                                   | term-bases index                                                                                                                                                                        |
| `SEARCH_ENGINE_USERNAME` / `SEARCH_ENGINE_PASSWORD`   | HTTP basic auth, if the cluster uses it                                                                                                                                                 |
| `GLOSSARY_PATH` / `DNT_PATH` / `TM_PATH`              | the single source of what each component scores: a dataset file, or a folder of that component's datasets                                                                               |
| `TM_INDEX`                                            | the TM index entries are fetched from — `benchmark_tm` today, post-mt's own index once they are the same                                                                                |
| `BENCH_COMPONENT`                                     | which components a run measures, one or more of `glossary`, `dnt` and `tm`                                                                                                              |
| `TM_FUZZY_FLOOR` / `TM_SEMANTIC_FLOOR`                | the match rate a candidate needs to band as fuzzy, and the embedding similarity it needs to band as semantic — currently `0.75` and `0.80`                                                |
| `TM_REFERENCE_FLOOR` / `TM_REFERENCE_SEMANTIC_FLOOR`  | how close a version must be to the entry's target to count as having used it — currently `0.90` and `0.85`                                                                                |
| `TM_PARTIAL_FLOOR`                                    | what an entry covering only part of a segment must reach on that part, banding and matching alike — currently `0.95`, above the whole-text floors because a window is easier to reach                     |
| `TM_MIN_RELEVANCE`                                    | lowest relevance grade a listed entry needs to enter the candidate pool — currently `0`, which accepts every entry the gold set lists                                                     |
| `TM_LENGTH_GUARD`                                     | length difference beyond which a use is flagged for review, never excluded — currently `0.5`                                                                                              |
| `TM_HARD_NEGATIVES`                                   | whether the match test is calibrated against the gold set's hard negatives — currently `true`                                                                                             |
| `TM_SEARCH_ENGINE_URL`                                | the cluster holding the TM index, when it is not the term-bases one                                                                                                                     |
| `TM_SOURCE_FIELD` / `TM_TARGET_FIELD` / `TM_SOURCE_LANG_FIELD` / `TM_TARGET_LANG_FIELD` | the index's own field names — `source_text`, `target_text`, `source_lang` and `target_lang`                                                                                     |
| `TM_EMBEDDING_URL`                                    | the LiteLLM proxy the semantic band and the semantic match tier embed through, `https://litellm-dev.acolad.build`                                                              |
| `TM_EMBEDDING_MODEL`                                  | the model asked for, `azure/azure/text-embedding-3-small` as in `rag_benchmark`                                                                                               |
| `TM_EMBEDDING_API_KEY`                                | the proxy key, and the switch for the semantic band: unset means no embedder, and its rows read `unmeasured`                                                                            |
| `DNT_BASE_URL` / `DNT_API_KEY`                        | the DNT service and its key, sent as `X-Api-Key` — note the casing, post-mt's own key is not accepted                                                                                   |
| `ES_AWS_SIGV4_ENABLED` / `AWS_REGION` / `AWS_PROFILE` | sign requests with AWS SigV4 instead — required by AWS-managed domains, which reject basic auth. Needs `pip install -e ".[aws]"` and a live login (`aws sso login --profile <profile>`) |

The *APE* column only means anything if post-mt was shown the same terms, and it finds them by
asking the CAT tool which term bases are attached to `cat_project_id`. Every component therefore
requires **`tempo_task_id`** and **`cat_project_id`**. Terminology also
requires **`cat_tool_provider`** and **`ecosystem_id`**, without which retrieval is skipped and APE
runs blind. Since a well-formed `cat_project_id` naming no real project passes every field check and
still retrieves nothing, terminology submits **one AQE-only segment** and reads `has_glossary` off
the reply, stopping the run for the price of one segment rather than billing the dataset for a
measurement that means nothing. Both ids come from the CAT tool and cannot be invented, and the term
base named in `glossary_ids` has to be attached to that project.
