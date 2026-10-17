# News-driven AI Picture Diary

This project generates an **English-language fictional diary** from unrelated, fresh news headlines. The news does not become the diary's factual narrative: it only supplies the creative seeds.

## Architecture

```
RSS / Atom (three configured English sources)
  -> collector.py: deduplication, date filtering, URL provenance
  -> keywords.py: pretrained spaCy POS/NER for named entities and noun phrases
      -> rank by specificity, inverse article frequency and freshness
  -> selector.py: cross-article / cross-publisher triples
      using sentence-transformers/all-MiniLM-L6-v2
      cosine distance = 1 - cosine_similarity
      balanced=quality-first (specificity/recency + semantic diversity)
      max and random are baselines on specific, verified NLP candidates
  -> data/news/selection.json (replayable selection and scores)
  -> scripts/generate_sentence.py: Ollama JSON output, English validation
  -> fixed_entry.yml: persisted keywords, source links, scores
  -> existing illustration worker -> GitHub Pages
```

**Pretrained model:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0, 384-dimensional English text embeddings). The model is downloaded on first execution; no training is performed.

The extractor now uses the pretrained `en_core_web_sm` POS tagger and named-entity recognizer. Only proper names and noun phrases (not arbitrary bigrams) enter the candidate pool. Generic reporting terms are excluded. A 48-hour freshness window, exponential recency preference, and article-frequency rarity prioritize news-specific concepts. The `balanced` selector restricts choices to nearly best-quality combinations before comparing semantic distances. These are heuristics, not an objective measure of creativity or global newsworthiness; `max` and `random` remain baselines.

## Local usage

Install dependencies in a Python 3.12 environment:

```sh
python -m pip install -r news_agent/requirements.txt pytest
python -m spacy download en_core_web_sm
```

Fetch recent headlines (RSS access and initial model download require internet):

```sh
python -m news_agent.cli --config news_agent/sources.json \
  --output data/news/selection.json --mode balanced --seed 42
cat data/news/selection.json
```

To reproduce a selection from **your own** cached article JSON list, use:

```sh
python -m news_agent.cli --articles-json path/to/articles.json \
  --output data/news/selection.json --mode max --seed 42
```

The JSON list must use the `Article` fields `id`, `publisher`, `title`, `url`, and `published_at`. The CLI intentionally fails if fewer than three usable articles, or no valid combination, remain. It does not invent headlines or replace semantic scoring with a hidden fallback.

Run a local Ollama instance with `qwen2.5:3b-instruct` available, then:

```sh
OLLAMA_BASE_URL=http://127.0.0.1:11434 \
OLLAMA_MODEL=qwen2.5:3b-instruct \
python -m scripts.generate_sentence --context data/news/selection.json
```

For offline unit tests (no RSS or model download):

```sh
python -m pytest -q tests/test_news_agent.py tests/test_validation.py
```

## GitHub Actions and provenance

The current `feed.yml` generates and publishes an English news-inspired diary on the main branch. It keeps `keywords`, `source_articles`, and `selection` in the feed JSON for reproducibility. The public UI does not need to display article links. The writing prompt receives only three dynamically selected story keywords, never article titles or publisher names.

Edit `news_agent/sources.json` to add/remove RSS feeds. Failed sources are reported in `source_errors`; a run stops if it cannot obtain a valid triple. Sources remain credited by publisher, headline, article URL and publication time. Full copyrighted article bodies are neither scraped nor republished.

**Important inherited limitations:** The existing image-generation workflow uses a large diffusion model and its CI feasibility depends on GitHub runner resources, Docker build and model availability. The original `event_core.yml` also expects a local `.env` for the illustration stage; this change does not fix that preexisting requirement. Successful offline unit tests do **not** prove an end-to-end run in GitHub Actions. Do an initial `workflow_dispatch` test and inspect the Actions logs before relying on the scheduled publication.

Old published Japanese entries remain in the archive; new output is English only. Existing, unrelated legacy workflows and backend RAG services are not removed by this patch.

## Docker Compose (recommended)

Use the **standalone** `docker-compose.news.yml` to avoid starting PostgreSQL,
Ollama or the illustration worker just to extract news keywords.

```sh
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
docker compose -f docker-compose.news.yml build news-agent
docker compose -f docker-compose.news.yml run --rm --entrypoint python \
  news-agent -m pytest -q tests/test_news_agent.py tests/test_validation.py
docker compose -f docker-compose.news.yml run --rm news-agent
jq '{keywords, selection, source_count, article_count}' data/news/selection.json
```

The first build downloads CPU-only PyTorch, spaCy and its trained English
`en_core_web_sm` model; the first non-mocked run downloads pretrained MiniLM. A named volume caches
the model across runs. The output JSON is written to `data/news/selection.json`
in the checkout and is gitignored. RSS and model downloads require internet.

To compare another strategy, replace the default command entirely:

```sh
docker compose -f docker-compose.news.yml run --rm news-agent \
  --config news_agent/sources.json --output data/news/selection.json \
  --mode max --seed 42
```

## Newsworthiness and specificity

The current POS/NER extractor selects recognized proper names and noun phrases, without hardcoded sample headlines or example keywords. If the feeds contain too few suitable candidates, extraction fails instead of substituting predefined example words.

Each selected keyword includes `kind` (`named_entity`, `proper_noun`, or `noun_phrase`), `specificity`, `freshness`, article source URL and publication time, plus aggregate `mean_relevance` and cosine distance in `selection.json`. Source freshness defaults to **48 hours** (`--max-age-hours` overrides this), and newer headlines receive higher ranking. The seed controls selection *only among quality-eligible combinations*.

To inspect the extracted news concept quality after running Compose:

```sh
jq '{keywords, selected: [.selected[] | {text, kind, specificity, freshness, publisher, published_at}], selection}' data/news/selection.json
```

Docker Compose execution and live-feed NLP output have not been verified in the supplied development environment.

## Entity normalization and headline relevance

The collector preserves punctuation in actual names and recovers adjacent
proper nouns when the named-entity model misses a multiword name. Headline
modifiers are removed using POS/NER information, without a fixed list of
place-name examples. Generic abstract descriptions are excluded; concrete
concepts remain eligible. Selected terms expose
`entity_label` (where NER provided one) and `title_position`, the normalized
position of the mention in the headline. Higher-ranked entities include a
small relevance bonus for their NER category and appearing near the beginning
of the title. These are intentionally **heuristics**, not guarantees that a
candidate is the central subject of a news event.

To inspect the live output using Docker Compose:

```sh
docker compose -f docker-compose.news.yml run --rm \
  --entrypoint python news-agent -m pytest -q tests/test_news_agent.py tests/test_validation.py
docker compose -f docker-compose.news.yml run --rm news-agent
jq '.selected[] | {text, kind, entity_label, title_position, relevance, title}' \
  data/news/selection.json
```


## Local English diary and feed publishing (v4)

After running `news-agent` and reviewing `data/news/selection.json`, start
Ollama **inside the same standalone Compose project** and download the model.
The model is stored in the persistent `news_ollama_models` volume; you only need
to pull it once. Ollama is not exposed on a host port and does not depend on
the original app stack or a host-installed Ollama.

```sh
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
docker compose -f docker-compose.news.yml up -d ollama
docker compose -f docker-compose.news.yml exec ollama ollama pull qwen2.5:3b-instruct
docker compose -f docker-compose.news.yml run --rm news-agent
docker compose -f docker-compose.news.yml run --rm diary-writer
jq '{kind, text, keywords, source_articles, selection}' data/news/draft.json
```

`diary-writer` resolves `http://ollama:11434` via the private Compose network
and starts after the Ollama service passes its healthcheck. If it reports
`model not found`, run the `ollama pull` command above. If it reports
`connection refused`, inspect `docker compose -f docker-compose.news.yml logs ollama`.

`diary-writer` uses the news-generation module `scripts.generate_sentence` with a new
`--output-json` flag. Its draft is written **only after** the Ollama JSON is
validated (first person, English, three exact terms). Missing keywords or
invalid JSON cause a nonzero exit without publishing anything.

Publish the validated draft and rebuild the existing paginated feed and posts:

```sh
docker compose -f docker-compose.news.yml run --rm diary-publish
jq '{feed_url, updated_at}' frontend/app/public/latest.json
```

The publisher cross-checks all news attribution against the saved selection;
it refuses to publish a stale or altered draft. Running it again on the same
local day with the same draft will not create duplicate snapshots. It leaves
previous entries and images intact. Published HTML links back to the original
news items and labels the story as fiction.

Optional **existing** image-generation stage (large model, CPU-only in the
current illustration Dockerfile and `scripts/illustrate.py`; may be slow):

```sh
docker compose -f docker-compose.news.yml --profile illustration run --rm diary-illustrate
docker compose -f docker-compose.news.yml run --rm diary-index
```

The image worker updates the matching feed snapshot. `diary-index` rebuilds
index/page JSON and HTML so the image is visible. Do not publish to GitHub
until you have inspected both the generated text and the illustration.

The model can be overridden with `NEWS_OLLAMA_MODEL` (pull that model in the
Ollama service first). `NEWS_OLLAMA_URL` overrides the private Compose endpoint
when connecting to a separately managed Ollama server. The default is
`http://ollama:11434`, **not** `127.0.0.1`; inside a container, the latter
would refer to the `diary-writer` container itself.

The GitHub `event_core.yml` workflow now uses the same generator
`--output-json` path as local Compose and the isolated illustration service,
instead of depending on an untracked `.env` for illustration. This removes a
known integration blocker, but **live GitHub Actions, online feeds, Ollama and
SDXL generation have not yet been validated end to end**.

## Retry strategy for missing phrases (v6)

A valid JSON response from Ollama may still omit one of the three news terms,
particularly a long event headline.
The writer now reports the **specific missing phrase(s)** and sends the previous
text back for targeted revision rather than retrying the unchanged prompt.
Named entities still require literal matching. A longer event phrase can appear
as the exact text of a fictional newspaper clipping or stage prop; it must
not become an unsupported claim about a real person.

To allow more revisions without changing code:

```sh
NEWS_MAX_RETRIES=5 docker compose -f docker-compose.news.yml run --rm diary-writer
jq '{text, keywords, source_articles}' data/news/draft.json
```

A draft is created only after successful validation. On failure, `draft.json`
may be absent; this is deliberate and publication must not be attempted.
This repair strategy is verified with mocked Ollama responses in tests; live
model success is not guaranteed.

## Story keyword normalization (v7)

The writer must not require exact reproduction of an entire source event clause.
News selections have two distinct fields in the draft and published feed:

- `keywords`: untouched news extraction, used for provenance and source attribution.
- `story_keywords`: short, literal narrative anchors, used for LLM prompting and
  publication validation. Named entities retain exact spelling. For a `noun_phrase`
  representing an event and containing at least three words, a trained English
  POS tagger filters verbs, stopwords and generic terms. The remaining content words are ranked by
  English Zipf frequency (`wordfreq`); **lower-frequency meaningful words**
  take priority, independent of their position in the headline.
  The exact result depends on model tagging and word-frequency data,
  not hand-authored example keywords.

The selection deliberately trades event detail for more natural first-person fiction;
source titles, source URLs, and original keywords are never altered. The publisher
recomputes and checks the narrative anchors from the current `selection.json` and
rejects a tampered or stale draft. Named entities remain mandatory literal matches.
This is a bounded rule, not an assertion that those two words fully characterize
the news.

Re-run without re-collecting news:

```sh
docker compose -f docker-compose.news.yml run --rm diary-writer
jq '{text, keywords, story_keywords, source_articles}' data/news/draft.json
```

If a previous attempt failed, `draft.json` should not exist; do not publish until
the new writer succeeds. This patch was tested with mocked LLM responses, not
a live Ollama inference.

To test the scoring after applying this patch, **rebuild** the image (this
version adds the `wordfreq` English frequency dataset to the image):

```sh
docker compose -f docker-compose.news.yml build news-agent diary-writer diary-publish
docker compose -f docker-compose.news.yml run --rm diary-writer
jq '{keywords, story_keywords, text}' data/news/draft.json
```

The rarest word is not necessarily the most visually interesting; word
frequency is an explicit baseline that can be evaluated against other
selection policies. It is never used to assert that a news event is factual.

## Prompt configuration and Python module invocation

The writer receives **only the three dynamically selected narrative keywords**
and an unrelated numeric seed. Source article headlines, publishers and URLs are
stored in provenance, but are not included in the LLM prompt. There are no
hand-written exemplar terms or named people in the production prompt.

Use `NEWS_DIARY_PROMPT_PATH` to override the English news-diary prompt. The
legacy `PROMPT_PATH` variable is deliberately ignored by this generator,
because older `.env` files may point to removed Japanese-learning prompts.
Relative prompt paths are resolved from the repository root, not the current
process directory. For command-line and CI execution, prefer
`python -m scripts.generate_sentence` from the repository root; the direct
script entry point remains supported.

## Scheduled GitHub Actions

`.github/workflows/feed.yml` is the **single scheduled news-diary workflow**. It
runs daily at **12:17 JST (03:17 UTC)** on the repository's default `main`
branch. GitHub Actions scheduled runs can be delayed or occasionally skipped;
they are not an exact-time scheduler.

The scheduled pipeline uses the same Docker Compose services as local development:

1. Build the CPU-only news agent and start an isolated Ollama container.
2. Fetch RSS headlines and select three distinctive English news terms.
3. Generate and validate a first-person English fictional diary.
4. Publish it to `frontend/app/public/` with source articles and keyword scores.
5. Attempt a 512px SD-Turbo illustration on CPU; rebuild feed pages regardless.
6. Commit generated public assets to `main`, then invoke the existing Pages
   deployment workflow. A failed text generation aborts publication; a failed
   optional image generation does not discard a validated diary.

In the GitHub repository, **Actions → Feed → Run workflow** runs the pipeline
manually. **Illustration is enabled by default** on both scheduled and manual
runs. Disable it once with the manual **illustrate** input, or disable it on
scheduled runs by setting the Actions variable `NEWS_IMAGE_ENABLED=false`.
The hosted runner has no GPU: for scheduled runs the worker uses the smaller
`stabilityai/sd-turbo` model at 512px with one sampling step and guidance 0.
Actual CPU image generation still depends on memory, disk, and model access;
the image stage is best-effort and does not block publication of a valid text
entry. A GPU runner is preferable for high-quality, reliable illustration.

News URLs and selection metrics remain in feed JSON for reproducibility, but
links to the news articles are not shown in the website feed or post pages.

The scheduled workflow must be **committed to the default branch** (currently
assumed to be `main`), and Actions must be enabled. Set **Settings → Actions →
General → Workflow permissions** to allow the built-in `GITHUB_TOKEN` to write
repository contents; Pages also needs to be configured for GitHub Actions.
Do not commit `.env`, model caches, or `data/news/selection.json` and
`data/news/draft.json` as publication assets.

Manual verification:

```bash
gh workflow run feed.yml --ref main
gh run list --workflow feed.yml --limit 5
gh run watch $(gh run list --workflow feed.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```
