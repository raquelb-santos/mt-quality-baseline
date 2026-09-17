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

**Tag and placeholder integrity** — how much of a segment's markup each version carries through
unbroken. The tags are read off the segments themselves, and **MT**, **APE** and **REF** are each
scored against **SRC**.

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

What a run scores is configured in `.env`, as `GLOSSARY_PATH`, `DNT_PATH` and `TAGS_PATH`.
Each may name a dataset file or a folder of them (`.json` or `.csv`).
Components are configured separately because pooling adds counts within a stratum.

`BENCH_LANGUAGE` picks the language pairs, one slot per `BENCH_COMPONENT` slot, and a blank slot
scores every pair. A code without a region matches all of its regions, so `en_es` takes every
English–Spanish pair and `en-gb_es-es` takes only that one.

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
     │      tags ──────── a. count each tag in SRC, then in MT, APE and REF,
     │                       and check pairing and ordering in each
     │
     └─ 3. aggregate, pool by stratum, write the report
```

Everything about the translation itself comes from post-mt's own API, so the benchmark measures the
pipeline that actually runs rather than a reimplementation of it.

***

## Terminology adherence

Terms come from the **CAT tool that owns them**. The tool named in `cat_tool_provider` is asked
which term bases are attached to `cat_project_id` — `v1/projects/{id}/termBases` in Phrase, the
project's `termCustomerIds` in XTM — and then for the terms those term bases hold.

Terms and segments are both lemmatized
through Stanza and matched on the lemmas, which is what percolating the term-bases index does, and
the source language matches permissively over the full code and the base code.

A CAT export is one row per segment, recognised by its header rather than by its extension. It is
read by the columns `SEGMENTID`, `SOURCECONTENT`, `TARGETCONTENT` and `HUMAN_TARGET`, where
`TARGETCONTENT` is the MT being scored and `HUMAN_TARGET` the human reference. 

`ISOSOURCELANGUAGE`, `ISOTARGETLANGUAGE`, `CATTOOL`, `CATPROJECTID`, `TEMPOTASKCODE`,
`DOMAIN` and `OPERATION` describe the job rather than the text, and become the
dataset's parameters.


### Metrics

For every glossary term matched in a segment, the **REF count** is how often the target term
appears in the **human reference** — so a term used three times is not discharged by one — and the
**version count** is how often it appears in the **version being scored**, capped at the REF count.
Summed over every term in every segment:

```
expected_instances = sum of the REF counts
adherent_instances = sum of the version counts

adherence_rate     = adherent_instances / expected_instances

exact_instances    = adherent instances worded as in the human reference
exact_rate         = exact_instances / expected_instances

violations         = expected_instances - adherent_instances
```

Every adherent instance is either an **exact match**, using the same words as the reference, or a
**lemma match only**, where just the lemmas agree, as with `contrat` against the human's
`contrats`. The lemma matches only are therefore `adherent_instances - exact_instances`. 

Where one target sits inside a longer one, as `crédito` sits inside `mercado de crédito`, an
occurrence of the longer wording counts for the longer term alone.

REF counts a term only where it uses an approved target as written, or inflected when lemmas can be
compared. A term REF renders any other way is not scored, and its segment is listed for review.

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

* **Exact match and lemma match only** — the adherence rate split in two for each dataset, with
  the exact count also given per term.

* **Violations** — the sum of three failures, counted over the whole corpus rather than segment by
  segment, because one of them is only visible once every segment has been read:

  * **miss** — output carries no approved target, surface form nor lemma of reference term.

  * **inconsistency** — output carries an approved target for a term the glossary proposed in
    that segment, but not the wording the reference used there.

  * **over-application** — output carries a target that the glossary did not propose in that
    segment, for a term the source does not contain, and the reference did not use that wording
    there either.

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

## Tag and placeholder integrity

Tags come from **the segments themselves**. **SRC is the answer key**, **REF** is scored
beside **MT** and **APE** to show how many of them a human really keeps.

A fully correct segment therefore carries the same tags and placeholders, in the same numbers, in
SRC, in MT and in REF. 

Seven notations are read:

| Family         | Written as                          |
| -------------- | ----------------------------------- |
| `paired_open`  | `{1>` `{b>` `{b^>`                  |
| `paired_close` | `<1}` `<b}` `<b^}`                  |
| `standalone`   | `{1}` `{2}` `{name}`                |
| `double_brace` | `{{var}}`                           |
| `xml`          | `<ph x="1"/>` `<span ...>` `</span>` |
| `printf`       | `%s` `%1$s` `%.2f`                  |
| `entity`       | `&brand;` `&#160;`                  |


### Metrics

For every tag in a segment, the **SRC count** is how often it appears in the **source**, and the
**version count** is how often it appears in the **version being scored**, capped at the SRC count
so that a tag emitted four times where SRC had one cannot cover three dropped elsewhere. Summed over
every tag in every segment:

```
expected_instances = sum of the SRC counts
present_instances  = sum of the version counts

integrity_rate     = present_instances / expected_instances

dropped_instances  = expected_instances - present_instances
```

Comparison is **case-sensitive and exact** and no tag is
read as a renamed version of another. A version that turns `{1>` into `{2>` is therefore charged
twice, once for the `{1>` it dropped and once for the `{2>` it invented. 

The same tags counted **once each instead of per instance**, split four ways by how the version
count compares with SRC's:

| Bucket             | When                | Meaning                                        |
| ------------------ | ------------------- | ---------------------------------------------- |
| **never carried**  | none in the version | SRC carried it, the version has none of it     |
| **carried partly** | fewer than SRC      | kept in one place and lost in another          |
| **matched SRC**    | as many as SRC      | present exactly as often as it was owed        |
| **duplicated**     | more than SRC       | an id repeated, which re-imports as two spans  |

### What is reported

Everything below is reported for **MT**, for **APE** and for **REF**:

* **Integrity rate** — the share of expected instances the version carries, **per tag**, **per tag
  family**, **per dataset** and **per stratum**.

* **Error kinds**, counted separately over the whole corpus rather than netted against each other:

  * **dropped** — the source carried the tag and the version has fewer of it.

  * **duplicated** — the version has more of it than the source did.

  * **hallucinated** — a tag the source never carried at all, which is the receiving half of a
    renumbering.

  * **unpaired** — an opener with no closer, or a closer standing before its opener. Ids that
    already arrive broken in SRC are left out, so no version is charged for markup it was handed
    broken.

  * **mis-ordered** — the surviving tags stand in a different order than the source put them in,
    with nothing lost.

* **Segments clean** — the share of tag-bearing segments with no error of any kind. A segment
  whose source carries no tag is left out of the denominator, since nothing to preserve is not
  evidence that preservation works. An invented tag is still counted there.

* **Repairs and regressions** — a **repair** is a tag MT got wrong and APE corrected, a
  **regression** one MT got right and APE broke.


## Configuration

Create a `.env` in the repo root with the following variables:

| Variable                                              | Purpose                                                                                                                                                                                 |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POSTMT_BASE_URL`                                     | post-mt instance to drive                                                                                                                                                               |
| `POSTMT_API_KEY`                                      | sent as `X-API-KEY`                                                                                                                                                                     |
| `STANZA_BASE_URL`                                     | Stanza lemmatizer, `https://stanza.acolad.build` — no credential                                                                                                                        |
| `SEARCH_ENGINE_USERNAME` / `SEARCH_ENGINE_PASSWORD`   | HTTP basic auth, if the cluster uses it                                                                                                                                                 |
| `PHRASE_BASE_URL` / `PHRASE_USERNAME` / `PHRASE_PASSWORD` | the Phrase TMS that says which term bases a Memsource project has                                                                                                                   |
| `XTM_BASE_URL` / `XTM_CLIENT` / `XTM_USER_ID` / `XTM_PASSWORD` | the same for XTM projects                                                                                                                                                     |
| `GLOSSARY_PATH` / `DNT_PATH` / `TAGS_PATH`            | the single source of what each component scores — a dataset file, or a folder of that component's datasets                                                                              |
| `BENCH_COMPONENT`                                     | which components a run measures, one or more of `glossary`, `dnt` and `tags`                                                                                                            |
| `BENCH_LANGUAGE`                                      | which language pairs each component scores, one slot per `BENCH_COMPONENT` entry and blank for all of them — `en_es`, or `en-gb_es-es` to pin the regions                               |
| `DNT_BASE_URL` / `DNT_API_KEY`                        | the DNT service and its key, sent as `X-Api-Key` — note the casing, post-mt's own key is not accepted                                                                                   |
| `ES_AWS_SIGV4_ENABLED` / `AWS_REGION` / `AWS_PROFILE` | sign requests with AWS SigV4 instead — required by AWS-managed domains, which reject basic auth. Needs `pip install -e ".[aws]"` and a live login (`aws sso login --profile <profile>`) |

*APE* column only means anything if post-mt was shown the same terms, and post-mt
finds them by asking the CAT tool which term bases are attached to `cat_project_id`. Terminology
therefore also requires **`cat_project_id`** and **`cat_tool_provider`**, and needs one of the two
CAT tools configured, since the CAT tool holds both the list of term bases and the terms themselves.
Before any segment is billed, a run checks that these fields are present and that
`cat_tool_provider` names a CAT tool post-mt supports, and a failure stops the run. A dataset none
of whose projects has a term base attached, such as a well-formed `cat_project_id` naming no real
project, is skipped with a warning before it is billed. A glossary the benchmark found but post-mt
did not retrieve surfaces only once the dataset has been billed, as a scorecard warning raised from
the `has_glossary` flag post-mt returns. Both ids come from the CAT tool and cannot be invented.
