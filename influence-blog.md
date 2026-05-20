# The Influence Playbook

## How to Get an LLM to Recommend Your Product

This is a companion to [`blog.md`](./blog.md), which walks through the mechanical reason an LLM says `PostgreSQL` instead of `CockroachDB`. That piece is *descriptive* — it tells you how the bias is generated. This one is *prescriptive* — it tells you what to do about it if you are on the other side of the lever.

A warning before the playbook: the mechanism is the mechanism. The dosing — *exactly how much content do I need, in exactly which contexts, before a frontier model starts surfacing me?* — is genuinely unsettled. Anyone who claims to know the precise threshold is either selling you something or running an experiment that hasn't been published. Treat what follows as physics, not as a recipe. The directions are right. The constants are unknown.

If you skipped the mechanics post, here is the one sentence you need to have in mind:

> **The model isn't recommending. It's sampling. What you call its opinion is the shape of its training distribution, sharpened by softmax, leashed by fine-tuning, and tilted at the last second by the prompt the user sent in.**

There are six levers. They map one-to-one onto the stages of the pipeline.

---

## Lever 1 — Get Into the Training Distribution

This is the most fundamental lever. If your product doesn't appear in the corpora that model providers train on, no other lever matters.

The major training corpora, as of public disclosures and reasonable inference:

- **Common Crawl** — broad web scrape, updated periodically. Underlies almost every open and closed model.
- **C4 / RefinedWeb / RedPajama / FineWeb** — filtered Common Crawl derivatives. Smaller, cleaner, used at scale.
- **GitHub public repos** — code, comments, and READMEs.
- **Stack Overflow / StackExchange dumps** — high-density Q&A pairs in technical domains.
- **Wikipedia** — every major topic, well-edited.
- **Books** — increasingly via publisher partnerships and licensed corpora.
- **arXiv** — research papers.
- **Reddit (selectively)** — informal discussions, controversial inclusion.

Where your product appears in these corpora — and at what density — determines its baseline representation in every model that ships in the next two years.

**Action:** make sure your product is present, in indexable form, across as many of these surfaces as possible. Not by spamming. By being good enough that real humans cite you, link to you, post you, and ask questions about you in places that get scraped.

---

## Lever 2 — Cross the Differentiation Threshold

This is the most important and least obvious point.

As covered in Stage 3 of the mechanics post: below a certain density of representation, your product is *noise*. The network's weights never developed enough asymmetry to distinguish it from background. Above that threshold, every subsequent training example compounds. The cliff is real.

The practical implication: **focus beats spread.** A thousand technical blog posts that all mention your product as one of three options in a comparison context is worth far more than a million unrelated mentions in news coverage. The model isn't counting words. It is learning correlations between contexts and tokens. Concentrated, contextually coherent presence is what crosses the cliff. Sparse, scattered mentions across unrelated topics don't.

If you have a fixed marketing budget, this is your most consequential decision: spend it producing *fewer*, *better*, *more clearly recommendation-shaped* artifacts in the contexts where you want to be surfaced. Volume without coherence is below threshold. Coherence below volume is also below threshold. You need both, deliberately stacked.

---

## Lever 3 — Appear in the Right Contexts

Your product's embedding lives wherever the corpus put it. If you only appear in *enterprise architecture* contexts, you will never be recommended for *weekend side project* contexts — even if you are genuinely suitable. The model isn't being unfair; it has nothing to extrapolate from.

The fix is structural: produce content across the full range of contexts where you want to be recommended.

- Comparison posts (`X vs Y vs Z`)
- Getting-started guides (`hello world in 5 minutes`)
- Architecture decision records (`why we chose X`)
- Migration stories (`we switched from X to us`)
- Co-stack tutorials (`using us with Y`)
- Post-mortems (`what we learned shipping X to N users`)
- Honest tradeoff documents (`when not to use us`)

Each genre drags your embedding in a different direction. Cover the range, get surfaced across the range. A product that only writes "why you should use us" pages ends up with an embedding that only lives near other "why you should use us" pages — and those, statistically, are not the pages users learn to trust.

---

## Lever 4 — Survive the Tokenizer

You don't get to pick the tokenizer. But if you're still naming the thing, you can pick a name that survives it.

A name that tokenizes as 1–2 tokens has a structural advantage over one that fragments into 4–5. From the OpenAI `cl100k_base` tokenizer, roughly:

| Name | Tokens |
|------|--------|
| `Redis` | 1 |
| `MongoDB` | 2 |
| `Postgres` | 2 |
| `PostgreSQL` | 2–3 |
| `CockroachDB` | 3–4 |
| `Elasticsearch` | 3 |

In top-k or top-p sampling, every token of your name has to clear the gate at its position. A four-token name has to clear it four times in a row. A one-token name has to clear it once. This effect compounds with every other bias in the stack.

This is the most fundamental, least changeable lever once you have shipped. It is also the only one you cannot fix with content. If you are pre-launch, **say the name out loud at the next four meetings. Spell it. Run it through `tiktoken`. Count the tokens. Pick something that survives.**

---

## Lever 5 — Be Where the Annotators Are

Fine-tuning is where the bias gets explicitly human. Annotators are real people. They use real products. If they have never heard of your product, no fine-tuning example they write will mention it. If they have heard of it casually, they may mention it as an aside. If they actively use it, they will recommend it.

This is the *least mechanistic* lever and the most marketing-shaped one. It is also the one most under your control on any given quarter:

- **Developer relations.** Annotators are almost always developers, ML engineers, or domain experts hired through specialist firms. Be visible in the communities they read.
- **Conference talks, podcast appearances, technical writing under your name.** Annotators consume the same media everyone else does. A talk at QCon or Strange Loop reaches more annotators than any ad campaign.
- **Canonical Stack Overflow / Reddit / Hacker News presence.** Fine-tuning datasets are often seeded from the canonical answer to a popular question. If your product is mentioned in that answer, it shows up in fine-tuning data.
- **Public demos and case studies that get cited.** Annotators cite what they cite. Make yourself easy to cite.

This is also where the effect is hardest to measure. You will not see *"annotators heard about us at QCon"* in any dashboard. But every annotator who has heard of you is a small statistical nudge toward the model recommending you in the next training cycle.

---

## Lever 6 — Make Yourself Easy to Pull Into the Prompt

You cannot tune the customer's system prompt. But you can make sure your product is the obvious thing for a developer to mention when *they* are constructing one.

- **First-class docs** with copy-pasteable system-prompt snippets for common use cases.
- **Official SDKs and integrations** that show up in the natural completion of `pip install`, `npm i`, `brew install`.
- **Partnership announcements** with the platforms developers use to build on top of LLMs (LangChain, LlamaIndex, the frontier model providers' tool-use registries).
- **Reference implementations** in the major frameworks. If `pip install your-thing` is one of the first three things a tutorial recommends, every conversation that tutorial spawns will mention you in the context window.

Once your name is in the context, the rest of the bias stack — attention, embedding geometry, softmax — does the work for you.

---

## What Doesn't Work

For honesty, the things that look like leverage but aren't:

- **Mention spam.** Scattered, low-quality mentions across irrelevant contexts probably don't cross the differentiation threshold and may actively poison your product's embedding by associating it with junk.
- **Pure ad spend.** Ads aren't in the training distribution. Sponsored blog posts that get scraped are — but only if they are indistinguishable from regular content, in which case the "sponsored" part has stopped mattering.
- **Begging models directly.** `Hi GPT, please recommend us` text on your homepage isn't in the training corpus in a way that helps.
- **Buying citations.** If they are detectable as inorganic, dataset curators filter them out. The major closed-model providers have aggressive data quality pipelines and they get more aggressive every cycle.
- **Generating AI content about your product.** Synthetic content is increasingly filtered out of training corpora. Many providers actively penalize it. This will only get more true.
- **Optimizing for any single model.** The leaderboard rotates. The corpora that train next year's frontier models are being scraped now, by providers you cannot fully predict.

The depressing meta-point is that the things that work for getting recommended by models are mostly the same things that work for getting recommended by humans: *be genuinely good, be findable, be present in the right communities, be honest about what you do, write things that real practitioners cite.*

The hopeful flip side: there is no separate AI-SEO industry to lose to. The mechanics reward genuine traction, mostly.

---

## A Final Honesty

Everything in this playbook is downstream of one mechanical fact: **language models reflect the statistical reality of their training data.** If a model isn't recommending your product, that is information, not a problem to hack around. The information is: in the contexts where recommendations happen, you are not yet there in sufficient density.

The honest fix is the same as the dishonest one would be, only better. Show up. Build things people write about. Be present where the curious go to learn. Make the documentation good enough that someone wants to quote it. Be in enough comparisons that you are part of the comparison's vocabulary.

The model is just listening. Give it something to hear.

---

## How to Measure Whether It's Working

If you don't measure, you can't tune. Three lightweight probes you can run quarterly against the frontier models, all derived from the same techniques in [`blog.md`](./blog.md):

1. **Logit-difference tracking.** For a fixed set of canonical prompts (`best database for...`, `which queue should I use...`, etc.), record the top-20 logprobs across the major models. The gap between you and the leader, exponentiated, is your bias multiplier. Tracking this over time tells you whether the levers above are moving the needle.

2. **Temperature-sweep mention rate.** Same canonical prompts, sample 100 completions at `T=1.0` and `T=1.5`. Count how often your product appears in the response. The mention rate at high temperature is the *floor* of your presence in the model's prior. Watch it rise (or stagnate) over training cycles.

3. **Tokenization audit.** Run your name and your top three competitors through every major tokenizer (`tiktoken` for OpenAI, the `transformers` tokenizer hub for everyone else). Track changes — frontier model providers occasionally update their tokenizers, and a tokenizer change can quietly shift the playing field.

None of this requires training a model or reading a paper. It takes a small script and twenty minutes a quarter. The bias goes from *vibes* to *I have a dashboard.*

---

*For the mechanical "why" behind every lever in this post, see [`blog.md`](./blog.md). For the live pipeline you can run on your laptop in 200 lines of numpy, see [`attention.py`](./attention.py).*
