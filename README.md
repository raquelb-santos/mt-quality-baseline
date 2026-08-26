# mt-quality-baseline

Benchmarks the translation quality of the `post-mt` pipeline.

A run takes segments with their source, their raw MT and the human translation delivered for them,
sends them through post-mt, and scores two versions of every segment against one or more **quality
components**:

* the **MT baseline** post-mt was given,

* the **post-edited output** it returned.

Both are scored **against the human reference for the same segment**.

A run reports:

* **the score for each version**, and the **delta** post-editing moved it by

* **repairs and regressions** — what post-editing fixed, and what it broke

* **a per-term worklist**, worst first, so a number traces back to the terms that produced it

* **pooling by stratum** when a run covers several datasets

## Components

| Component                                       | Measures                                                        | Status      |
| ----------------------------------------------- | --------------------------------------------------------------- | ----------- |
| [Terminology adherence](#terminology-adherence) | how much of the glossary the human used each version reproduced | implemented |

**Each component has its own dataset**. One run measures one component.

## Quick start

Terminology reads the glossary from the term-bases index post-mt itself queries, so a run needs
`SEARCH_ENGINE_URL` (and AWS SigV4 for the dev domain) in a `.env`.

```bash
# scores BENCH_DATASET, post-editing each segment through post-mt
python sourcecode/cli.py

# the same, but scores the MT baseline only — never calls post-mt, so it costs nothing
python sourcecode/cli.py --dry-run
```

`pip install -e .` also puts the same entry point on the path as `mt-quality-baseline`.

What gets scored is configured in `.env`, as `BENCH_DATASET`. A run prints its report and
writes nothing: the numbers belong to the run that produced them.

`BENCH_DATASET` may name a single dataset file, or a folder — a folder contributes every
`.json` / `.csv` / `.mxliff` directly inside it, scored in one run and pooled by stratum.  If the files don't state which term bases they apply to, those
ids have to be supplied with the `--glossary-ids` flag, along with the language pair the file also
cannot state:

```bash
python sourcecode/cli.py --dry-run \
    --glossary-ids 0xdsJEaDoENXui2rxMdZE3 \
    --source-lang en-gb --target-lang fr-fr
```

Pass several ids as one comma-separated value (`--glossary-ids a,b,c`). The flag overrides a JSON
descriptor's pinned ids too.Running `cli.py` prints the scorecard and the worklist. Under `--dry-run` the post-edited column mirrors the MT baseline and the APE (automatic
post-editing) counters stay at zero, because post-mt was never called.

***

## How it works

```
 one component's dataset: source · raw MT · human reference · the component's own fields
     │
     ├─ 1. run segments ──────────────────────► post-mt  POST /api/workflow/async
     │      (reference stripped from payload)            GET  /api/workflow/async/:id
     │      returns: the raw MT it was given (the baseline)
     │               the post-edited text
     │               per-segment step failures
     │               whether a glossary reached the model
     │
     ├─ 2. score MT and post-edited against the reference, with the dataset's component
     │      terminology ─ a. lemmatize ───────► Stanza        (as post-mt does)
     │                    b. resolve terms ───► term-bases    (same query post-mt sends)
     │                         (glossary ids pinned in the dataset)
     │                    c. count each term in the reference, then in each version
     │
     └─ 3. aggregate, pool by stratum, print
```

Everything about the translation itself comes from post-mt's own API, so the benchmark measures the
pipeline that actually runs rather than a reimplementation of it..

***

## Terminology adherence

How much of the glossary the human applied each version of the translation reproduced.

Terms come from the **term-bases index**, selected with the same Elasticsearch query post-mt sends —
so the benchmark scores against the terms the pipeline was actually shown. Which term bases are
queried is pinned per dataset as `glossary_ids`.

### The metric

For every glossary term matched in a segment:

| Count   | Is                                                                                                                                     |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **`R`** | occurrences of the **target term in the human reference** — what the version owed, so a term used three times is not discharged by one |
| **`T`** | occurrences in the **version being scored** — what it delivered against that demand, capped at `R`                                     |

Summed over every term in every segment:

```
expected_instances = sum of R
adherent_instances = sum of T

adherence_rate     = adherent_instances / expected_instances
missed_instances   = expected_instances - adherent_instances
```

The rate is over what the reference demanded, so it pools across datasets; the shortfall stays a
count, which is what there is to go and fix. The rate is never averaged — at each grain, **per
term**, **per segment** and **per stratum**, it is recomputed from the counts of the grain below, so
a term matched once cannot outweigh one matched forty times. The per-term and per-segment counts
each sum to the dataset total.

`missed_instances` is subtraction and says nothing about why. A **violation** is an instance missed
where the segment has text but not the target term; an **omission** is one missed where the segment
came back empty — reported, but never a violation. They sum to `missed_instances` at every grain and
pool by addition. Keeping them apart is what stops a pipeline that returned nothing from reading
like one that translated everything with the wrong words.

**Term breakdown.** The same terms counted **once each instead of per occurrence**, split four ways
by how `T` compares with `R`. Exhaustive, so the buckets sum to the distinct terms scored; counts,
not a rate.

| Bucket                       | When        | Scored as                                                                               |
| ---------------------------- | ----------- | --------------------------------------------------------------------------------------- |
| **never used**               | `T = 0`     | missed, adherence 0                                                                     |
| **used, not everywhere**     | `0 < T < R` | missed, adherence partial — *used but not everywhere* is inconsistency inside a segment |
| **matched the reference**    | `T = R`     | adherent                                                                                |
| **used more than the human** | `T > R`     | adherent — never a violation, **flag for review**                                       |

**Strict and permissive slices.** The two kinds of glossary instruction, scored separately.

| Prompt rendering                         | Meaning              | Scored as                     |
| ---------------------------------------- | -------------------- | ----------------------------- |
| `` `X` should be translated to: `Y` ``   | one target term      | **strict** — only `Y` counts  |
| `` `X` may be translated as: `Y`, `Z` `` | several target terms | **permissive** — *any* counts |

**Segment-level adherence.** The share of glossary-bearing segments with *zero* violations.

**Repairs and regressions.** Counted separately rather than netted against each other: a **repair**
is a term the raw MT got wrong and post-editing corrected, a **regression** one it got **right** and
post-editing broke.

***

## Configuration

Create a `.env` in the repo root with the following variables:

| Variable                                              | Purpose                                                                                                                                                                                 |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POSTMT_BASE_URL`                                     | post-mt instance to drive                                                                                                                                                               |
| `POSTMT_API_KEY`                                      | sent as `X-API-KEY`                                                                                                                                                                     |
| `STANZA_BASE_URL`                                     | Stanza lemmatizer, `https://stanza.acolad.build` — no credential                                                                                                                        |
| `SEARCH_ENGINE_URL`                                   | term-bases index                                                                                                                                                                        |
| `SEARCH_ENGINE_USERNAME` / `SEARCH_ENGINE_PASSWORD`   | HTTP basic auth, if the cluster uses it                                                                                                                                                 |
| `BENCH_DATASET`                                       | the single source of what gets scored: a dataset file, or a folder of one component's datasets                                                                                          |
| `BENCH_LEMMA_MATCHING`                                | count inflected forms as matches (default `true`); `false` scores surface forms only                                                                                                    |
| `ES_AWS_SIGV4_ENABLED` / `AWS_REGION` / `AWS_PROFILE` | sign requests with AWS SigV4 instead — required by AWS-managed domains, which reject basic auth. Needs `pip install -e ".[aws]"` and a live login (`aws sso login --profile <profile>`) |

The benchmark resolves the glossary itself, but the *post-edited* column is only meaningful if
post-mt was also shown those terms. It looks them up by asking the CAT tool which term bases are
attached to `cat_project_id`.

***

## CLI reference

```bash
python sourcecode/cli.py [options]
```

There is no dataset argument: what gets scored is `BENCH_DATASET` in `.env`. The options below
change *behaviour*, not what is measured.

| Option                            | Effect                                              |
| --------------------------------- | --------------------------------------------------- |
| `--source-lang` / `--target-lang` | language code or name, instead of a `--params` file |
| `--domain D`                      | domain label, recorded in reports                   |
| `--dry-run`                       | scores the MT baseline only; never calls post-mt    |
| `--glossary-ids IDS`              | comma-separated, overrides the dataset              |
| `--params FILE`                   | JSON parameters block for `.csv`/`.mxliff` inputs   |
| `--steps STEPS`                   | pipeline steps (default `AQE,APE`)                  |
| `--batch-size N`                  | segments per post-mt task                           |

###
