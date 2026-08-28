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

Two components are implemented:

**Terminology adherence** — how often each version uses the glossary terms the human used. **MT**
and **APE** are both scored against **REF**.

**DNT preservation** — how often each version keeps the items that must not be translated. **APE**
is sent to the DNT service's **revert endpoint**, which returns both the items it weighed in a
segment and **REV**. **MT**, **APE** and **REV** are each scored against **REF**.

It reports each version's score and the **delta** from post-editing, **repairs and regressions**, a
worst-first **per-item worklist** tracing numbers back to segments, and **pooling by stratum**
across datasets.

## Quick start

Terminology reads the glossary from the term-bases index post-mt itself queries, so it needs
`SEARCH_ENGINE_URL` (and AWS SigV4 for the dev domain) in a `.env`.

DNT gets its items from the DNT service, so
it needs `DNT_BASE_URL` and `DNT_API_KEY`.

```bash
# scores the configured dataset, post-editing each segment through post-mt
python sourcecode/run.py

# the same, but scores MT only — never calls post-mt
python sourcecode/run.py --dry-run
```

`pip install -e .` also puts the same entry point on the path as `mt-quality-baseline`.

What a run scores is configured in `.env`, as `GLOSSARY_PATH` and `DNT_PATH`. Each may name a single
dataset file (`.json`, `.csv`, `.mxliff`, `.xliff`, `.xlf`) or a folder of them; a CSV or XLIFF
needs a `<name>.params.json` beside it giving its own `parameters`, `glossary_ids` and `steps`.
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
     │                         (glossary ids pinned in the dataset)
     │                    c. count each term in REF, then in MT and APE
     │
     │      DNT ───────── a. detect and revert ► DNT service  POST /v1/revert
     │                         (one call returns the items it weighed and REV,
     │                          a third version to score)
     │                    b. count each item in SRC and REF,
     │                       then in MT, APE and REV
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

## Configuration

Create a `.env` in the repo root with the following variables:

| Variable                                              | Purpose                                                                                                                                                                                 |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POSTMT_BASE_URL`                                     | post-mt instance to drive                                                                                                                                                               |
| `POSTMT_API_KEY`                                      | sent as `X-API-KEY`                                                                                                                                                                     |
| `STANZA_BASE_URL`                                     | Stanza lemmatizer, `https://stanza.acolad.build` — no credential                                                                                                                        |
| `SEARCH_ENGINE_URL`                                   | term-bases index                                                                                                                                                                        |
| `SEARCH_ENGINE_USERNAME` / `SEARCH_ENGINE_PASSWORD`   | HTTP basic auth, if the cluster uses it                                                                                                                                                 |
| `GLOSSARY_PATH` / `DNT_PATH`                          | the single source of what each component scores: a dataset file, or a folder of that component's datasets                                                                               |
| `BENCH_COMPONENT`                                     | which components a run measures, `glossary` and/or `dnt`                                                                                                                                |
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
