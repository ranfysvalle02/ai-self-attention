# Why Does the Model Recommend Postgres Over CockroachDB?

## The Mechanics of Bias in Large Language Models

Ask any major LLM *"which database should I use for my new project?"* and you'll get an answer. Maybe it says PostgreSQL. Maybe MongoDB. Maybe Redis if the conversation drifted. But *why* that answer? Not the prose justification the model writes for you — the actual mechanical reason that specific token sequence won out over another.

This isn't a philosophy question. It's a math question.

A large language model is a prediction engine. Every word it outputs is sampled from a probability distribution it just computed over its entire vocabulary. That distribution didn't come from anywhere magical. It came from a corpus, a loss function, a few learned weight matrices, and a handful of inference-time knobs. Each of those contributes to why `PostgreSQL` comes out of the model's mouth instead of `CockroachDB`.

If you want to understand the bias, you have to walk the pipeline in order, input to output. That's what this piece does — seven stages, end to end, the exact same seven stages [`attention.py`](./attention.py) in this repo runs in front of your eyes.

If you only remember one sentence:

> **The model isn't recommending. It's sampling. What you call its opinion is the shape of its training distribution, sharpened by softmax, leashed by fine-tuning, and tilted at the last second by the prompt you sent in.**

The rest is the long version.

---

## Stage 1 — Tokens & Embeddings: Where Words Become Geometry

Before the model can think about your prompt, it has to break the prompt into **tokens**.

A token is not a word. Modern models use sub-word tokenizers (BPE, WordPiece, SentencePiece) that split rare or novel words into pieces. In OpenAI's `cl100k_base` tokenizer (used by GPT-4 / GPT-3.5):

| Word | Approx. token count |
|------|---------------------|
| `Redis` | 1 |
| `MongoDB` | 2 |
| `PostgreSQL` | 2–3 |
| `CockroachDB` | 3–4 |
| `Elasticsearch` | 3 |

Before any math happens, the playing field is uneven.

> A product name that fragments into five tokens has to win five independent statistical battles to be generated. A product name that's a single token has to win one. The first place where "Postgres beats CockroachDB" gets quietly decided is the tokenizer.

Once tokenized, each token is looked up in a giant **embedding matrix**. Every token in the vocabulary lives somewhere in a high-dimensional space — typically `d_model` of 4096–16384 in production models, `8` in our demo. During training, tokens that appear in similar contexts drift toward each other. The geometry encodes the statistics.

`PostgreSQL`, `MySQL`, and `database` end up in a neighborhood. `CockroachDB` is nearby — but its exact position depends on what it appeared next to during training. If it mostly showed up near `distributed`, `multi-region`, `Spanner`, `consensus`, its embedding lives in that part of the space. Ask a question about *"a simple side project"* and the attention mechanism is going to find Postgres geometrically closer to your query than Cockroach. Not because Postgres is better. Because that's where the corpus put it.

**The training contexts of a product determine which future questions will surface it.** A database mentioned exclusively in enterprise architecture papers is functionally invisible to "I'm building a weekend project" queries — even if it would be perfectly suitable.

---

## Stage 2 — Self-Attention: Shaping the Question

Self-attention is the operation that lets every token in your prompt look at every other token and decide how much it cares.

```
scores            = (Q · K^T) / √d_k
attention_weights = softmax(scores)
context_vector    = attention_weights · V
```

That's the whole mechanism. Each token projects itself through three learned matrices (`W_Q`, `W_K`, `W_V`) into three roles: what it's looking for (Query), what it advertises (Key), and what it carries (Value). The dot product of Queries and Keys produces a similarity matrix. Softmax turns those similarities into weights. A weighted sum of Values produces a new, context-aware embedding for every position.

For the prompt *`which database should I use for`*, the Query at the `use` position dot-products strongly against the Key at `database`. The model has learned, across billions of training examples, that these tokens belong to the same kind of conversation. The attention weight for `database` goes up. The context vector emerging from this position is now heavily flavored by what `database` means to the model.

The key insight, which took me longer than I'd like to admit to internalize:

> **Attention doesn't choose the answer. It shapes the question.**

By the time attention is done, the model is no longer thinking about your literal prompt. It is thinking about a context-laden, re-weighted, *interpreted* version of your prompt — one that already leans toward the answers most strongly associated with the words you used. The bias gets pre-installed at this stage, before any specific token gets picked.

The actual numbers from running `attention.py` on `the cat sat on the mat`:

```
Attention weights (each row sums to 1.0):
          the    cat    sat     on    the    mat
   the  0.243  0.112  0.165  0.136  0.243  0.101
   cat  0.062  0.197  0.367  0.117  0.062  0.194
   sat  0.106  0.179  0.237  0.151  0.106  0.219
    on  0.137  0.172  0.230  0.176  0.137  0.147
   the  0.243  0.112  0.165  0.136  0.243  0.101
   mat  0.049  0.294  0.238  0.153  0.049  0.216
```

`cat` attends to `sat` at `0.367` — more than 3x what it gives to `the`. `mat` attends to `cat` at `0.294`. These asymmetric weights aren't properties of the words. They are properties of the *learned matrices* `W_Q`, `W_K`, `W_V`. Same words, different training run, different attention pattern. Same training run, different prompt, different attention pattern.

Attention is the lens. Training ground it.

### The Same Operation at Every Scale: Self-Attention, In-Context Learning, RAG

Look at what `Q · K^T` actually *is*: a similarity search. The query vector dot-products against every key vector and finds the ones it lines up with. Softmax turns those into weights. A weighted sum of values produces "the input, biased toward whatever this query was asking about."

That is not a transformer-specific trick. That is the entire family of context-injection techniques in modern LLMs, sitting on top of one another like Russian dolls. Self-attention, in-context learning (few-shot prompting), and retrieval-augmented generation (RAG) are not three different mechanisms. **They are the same mechanism, with progressively larger choices of `K` and `V`.**

| Mechanism | Where `K`, `V` live | What ends up shaping the question |
|---|---|---|
| **Self-attention** | The current prompt | Other tokens already in this input |
| **In-context learning** | The current prompt + few-shot examples | The examples you assembled |
| **RAG** | An external vector database | Whatever your retriever returned |
| **Conversation history** | Earlier turns concatenated into context | Whatever was said before |
| **System prompts** | A privileged region of the prompt | The persona/instructions you injected |

All five are the same four-step recipe:

1. Project the current position into a **query**.
2. Compute similarity against a set of **keys**.
3. Softmax similarities into **weights**.
4. Weighted-sum the corresponding **values** back into the context vector.

The only thing that changes is *whose keys and values are eligible*.

#### In-context learning is attention on your few-shot examples

When you paste five worked examples before your real question, the model's weights do not update. *Nothing inside the network changes.* What changes is the **set of keys and values the answer position is allowed to attend to.** The few-shot examples become high-similarity neighbors of the query in attention space. The context vector at the answer position becomes a weighted blend of those examples. The "lesson" the model appears to have learned is just the geometry of the prompt you assembled.

This is also why few-shot examples that *feel* relevant to a human sometimes do nothing. If their tokens don't dot-product strongly against the answer-position query, attention skips them. The model isn't ignoring you. Its query simply isn't matching your keys.

#### RAG is attention with the `K` matrix swapped out

Retrieval-augmented generation is what you get when you take Step 2 — "compute similarity against a set of keys" — and run it *outside* the model on a keystore too big to ever fit in the attention window. The vector database stores pre-computed embeddings of document chunks. `top-k` retrieval is just **hard-attention**: instead of softmax-weighting every chunk continuously, you keep the `k` highest-similarity ones and discard the rest by setting their weights to zero.

Once those `k` chunks are concatenated into the prompt, **regular self-attention takes over.** The answer-position query dot-products against the retrieved chunks the same way it dot-products against any other context. RAG is not a parallel pathway bolted onto the transformer. It is a feeder system that decides which keys and values get the privilege of being inside the attention window in the first place.

> **RAG is attention. The vector database is just a `K` matrix too big to keep on the GPU.**

#### So who actually asked the question?

By the time the LM head fires, the model is responding to a query vector that has been re-shaped by:

- the rest of the prompt (self-attention),
- the few-shot examples (in-context learning),
- the retrieved documents (RAG),
- the conversation history (multi-turn attention),
- and the system prompt (privileged context).

The string the user typed is *one input* into that re-shaping. It is not the question the model answers. It is the question the model *starts from*, before five layers of attention finish editing it.

> Self-attention shapes the question with the prompt. In-context learning shapes it with examples. RAG shapes it with documents. They are the same operation, scaled outward.

**Every system that "adds context" to an LLM is mechanically an attention mechanism whose `K` and `V` matrices it gets to author.** That is enormous leverage — and it is why prompt engineering works, why prompt injection works, why a stale RAG index silently lobotomizes a deployed assistant, and why a single well-placed few-shot example can flip a model's answer for an entire class of queries.

You don't need new architecture to steer an LLM. You just need to control its keys and values.

---

## Stage 3 — Feed-Forward + Activation: The Nonlinear Amplifier

The context-aware vector from attention is still a smooth, linear combination of its inputs. To learn anything genuinely nonlinear — anything that requires `if A and B but not C` logic — the model has to pass through a **feed-forward network** with an **activation function**.

GPT-2/3/4 use GeLU. Llama uses SwiGLU. The shape is always:

```python
hidden = activation(x · W1 + b1)
output = hidden · W2 + b2
```

Activation functions are gatekeepers. They suppress weak or negative signals and pass strong positive ones through. From the demo:

```
Before activation (sample from 'cat' position):
  Linear:     [ 0.627   0.053  -0.729   0.244 ]
  After GeLU: [ 0.461   0.028  -0.170   0.145 ]
```

The negative value (`-0.729`) gets crushed to roughly a quarter of its magnitude. The strong positive (`0.627`) passes through at ~73% strength. Everything in between is re-weighted nonlinearly. Stack this dozens of times across 96+ transformer blocks and the network can carve out arbitrarily complex decision surfaces.

### The Differentiation Threshold

Here is the most consequential thing nobody tells you about neural networks:

> **If you initialize weights too small, the entire system produces uniform mush.**

Run the demo with `init_scale = 0.01` and every word gets roughly equal probability. Attention weights flatten. Softmax has nothing to amplify. Temperature has no effect. The model is technically functional and practically useless — a coin flip dressed up in linear algebra.

This isn't a quirk of the demo. It's the fundamental insight about how these systems work:

**The network only produces value when its weights carry enough differentiation to create signal. Training is the process that creates that differentiation.**

Every gradient update during training moves weights further from uniform, encoding asymmetries. "In contexts like this, Postgres is more relevant than CockroachDB" is — literally — a small region of weight space where the matrices have grown asymmetric enough to project Postgres-ward vectors more strongly than Cockroach-ward ones.

This means the relationship between *frequency in training data* and *probability of generation* is **not linear**. There is a critical mass of representation below which a product is statistically indistinguishable from noise — the weights never developed enough asymmetry to encode it as a distinct concept. Above that threshold, every subsequent training example compounds. The "rich get richer" effect of neural networks. There is no gentle slope between obscurity and relevance. There is a cliff.

Empirically, you can see this in any frontier model's tendency to *confidently* recommend a small set of well-represented products and *completely fail to mention* technically equivalent alternatives that didn't cross the threshold. It is not a bug. It is the architecture.

---

## Stage 4 — The Language Modeling Head: From Vectors to Words

Stage 3 leaves you with a contextual vector at every position. That vector is still in embedding space — it doesn't correspond to any word yet. The **language modeling head** is the matrix multiplication that finally turns geometry into language.

```python
logits = context_vector · W_vocab     # one raw score per token in vocab
probs  = softmax(logits)              # turn scores into probabilities
```

For the last position in the sequence (the position that predicts what comes next), this projection produces one number for every token in the model's ~100k-token vocabulary. Then softmax.

Softmax is where small advantages become decisive ones.

$$P(\text{token}_i) = \frac{e^{\text{logit}_i / T}}{\sum_j e^{\text{logit}_j / T}}$$

A logit difference of `2.0` between two tokens translates to roughly a **7.4×** probability difference after softmax at `T=1`. A difference of `5.0` becomes ~**148×**. A difference of `10.0` becomes ~**22,000×**.

From the demo:

```
Token       Logit  Probability
the        -0.844     0.090
cat         0.712     0.426  <-- predicted
sat        -0.221     0.168
on          0.131     0.239
mat        -0.994     0.077
```

The logit gap between `cat` and `on` is `0.58` — a barely-existing difference in raw score. After softmax, `cat` gets 1.8× the probability. The signal didn't dominate. It just edged out, and the exponential reshape did the rest.

> **The model doesn't need to be overwhelmingly biased toward a token for it to dominate the output. It needs a modest logit edge. Softmax does the rest.**

This is exactly what bias in LLMs looks like. It is not the model "deciding" anything. It is a small logit asymmetry — produced by Stage 1 (tokenization), Stage 2 (attention pattern), Stage 3 (nonlinear amplification) — getting exponentially sharpened into a confident-sounding recommendation.

If you have ever wondered why every major LLM tends to converge on the same handful of "safe" answers across radically different prompts, this is most of the answer. Training has built up edges. Softmax has amplified them. The model isn't repeating itself because it's lazy. It's repeating itself because that's what a sharp distribution does when you sample from its mode.

---

## Stage 5 — The Loss Function and the Corpus: Where Bias Is Born

Everything above is *what happens at inference*. It is the consequence. Now go upstream.

During pretraining, the model is doing exactly one thing: next-token prediction. For every position in the training corpus, the network produces a probability distribution over the vocabulary, and the loss function asks how much probability mass it placed on the *actual* next token.

$$\mathcal{L} = -\log P(\text{actual next token})$$

If the training example reads *`For most web applications, I recommend PostgreSQL because...`*:

1. At the position before `PostgreSQL`, the model produces a distribution over its vocabulary.
2. The loss penalizes the model proportional to how *little* probability it assigned to the tokens that spell `PostgreSQL`.
3. Backpropagation pushes every weight in the network — embeddings, `W_Q`, `W_K`, `W_V`, feed-forward weights, `W_vocab` — in whatever direction makes that probability higher next time.

**Every single training example that mentions a product in a recommendation context is a small vote.** The loss function fires on every one of them. There is no neutral example. There is no "the model already knows this, skip it." The corpus is the model's entire moral universe and every gradient update is the universe whispering *this is what comes next, learn it.*

So bias isn't injected. Bias *is* training. It is what the loss function does, mathematically and unavoidably, when the corpus is non-uniform — and corpora are *always* non-uniform.

Concrete asymmetries you can verify yourself in under five minutes:

- **Stack Overflow tag counts** (Stack Exchange Data Explorer): the `postgresql` tag carries hundreds of thousands of questions; `cockroachdb` carries low thousands. Roughly two orders of magnitude.
- **GitHub repo counts**: same shape — searching `language:sql postgres` returns vastly more repos than `cockroachdb`.
- **Common Crawl mentions**: pages mentioning `PostgreSQL` outnumber `CockroachDB` by roughly two orders of magnitude in any recent crawl you sample.

Now compound that asymmetry with subword tokenization (Stage 1) and softmax sharpening (Stage 4). The output is structurally inevitable.

> **The loss function doesn't care about truth. It cares about prediction accuracy against the training distribution. The model isn't being optimized to give you the best answer. It is being optimized to reproduce the statistical patterns of its corpus.**

That is the most important sentence in this entire post. Read it twice.

---

## Stage 6 — Fine-Tuning: The Final Shaping

Pretraining gives you a base model that has absorbed the statistical fingerprint of the internet. It is not yet useful. Out of the box, it will happily complete prompts in directions you do not want, in styles nobody asked for, and with no instinct for what an "assistant" is.

Fine-tuning is what makes a base model a *product*. It is also where the bias story gets explicitly human, in two distinct phases.

### Phase A — Supervised Fine-Tuning (SFT)

A team of human annotators writes (or curates) thousands to millions of instruction → response pairs:

```
USER:      What's a good database for a multi-tenant SaaS?
ASSISTANT: It depends on scale and consistency requirements.
           PostgreSQL with row-level security works well for most
           teams under ~10TB...
```

The model is fine-tuned on these with the same cross-entropy loss as pretraining — but now the "correct" next token is no longer *whatever the internet happens to say*. It is *whatever this annotator wrote*. The model shifts toward the style and content the annotators preferred.

The dataset is small (~10k–1M examples) but it operates on a model that already has trillions of tokens of priors. The leverage is enormous. Annotator preferences — even ones the annotators didn't realize they had — propagate directly into the model's behavior.

If the annotator guidelines say *prefer well-known, battle-tested tools*, that becomes a gradient signal. If the annotators personally use Postgres at work, the examples they write will mention Postgres. The model learns the preference *from the data*, not from the guideline document. Nobody types `recommend Postgres` into a config file. They just write examples in which Postgres is what gets recommended.

### Phase B — RLHF (or DPO)

SFT only teaches the model to imitate one written response per prompt. It cannot teach the model to prefer one response *over* another. For that you need preference learning.

The standard recipe (RLHF, *Reinforcement Learning from Human Feedback*) has four steps:

1. Sample two model completions for the same prompt.
2. Ask a human which one is better.
3. Train a separate **reward model** `R(x, y)` to predict human preferences from those pairs.
4. Use policy optimization (typically PPO) to push the policy toward higher reward, while keeping a leash on how far it can drift from its pretrained behavior:

$$\max_{\pi_\theta}\ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \bigl[\, R(x, y) \,\bigr] \;-\; \beta\, \mathbb{D}_{\mathrm{KL}}\bigl[\pi_\theta(y|x)\ \|\ \pi_{\text{ref}}(y|x)\bigr]$$

Don't let the notation scare you. The equation has two halves and one knob.

- **First half** — *maximize expected reward.* Produce outputs the reward model rates highly. This is what bends the model toward responses humans liked.
- **Second half** — *the KL leash.* A penalty for drifting too far from the pretrained reference policy `π_ref`. Without it, the policy will collapse onto whatever degenerate output the reward model happens to score highest, because reward models are imperfect and gameable. (This failure mode has a name: *reward hacking.*)
- **β** — *the leash's tightness.* As `β → 0` the policy chases reward and forgets how to be a language model. As `β → ∞` the policy ignores reward and stays exactly as pretrained. Production RLHF runs spend a lot of compute looking for `β` in the middle.

### RLHF Is a Logit Shift — Here's the Math

The equation above looks intimidating, but it has a closed-form optimum that collapses the whole story onto one line. Solve the KL-regularized objective for the optimal policy and you get:

$$\pi^{\*}(y \mid x) \;\propto\; \pi_{\text{ref}}(y \mid x)\; \exp\!\left(\frac{R(x, y)}{\beta}\right)$$

Take the logarithm. The exponent becomes additive. The proportionality constant disappears into the softmax that's coming next:

$$\text{new\_logit}_y \;=\; \text{base\_logit}_y \;+\; \frac{R(y)}{\beta}$$

That is the entire mechanism. **A reward model is, mechanically, a learned bias vector added to the base model's logits, leashed by `1/β`.** PPO is the procedure that approximately discovers this bias vector through sampling and gradient updates. DPO derives the same shift in closed form from preference pairs without ever instantiating an `R(x, y)` network. The shape of the answer is identical.

This collapses RLHF from "mysterious reinforcement-learning recipe" to "addition." Which means we can demonstrate it on the demo's actual numbers.

### Watching the KL Leash Tighten

[`attention.py`](./attention.py) STEP 6 runs this exact computation. The base model's distribution from STEP 4 is:

```
the     base_logit = -0.8443   P = 0.0899
cat     base_logit =  0.7124   P = 0.4264   <-- pretraining winner
sat     base_logit = -0.2205   P = 0.1678
on      base_logit =  0.1315   P = 0.2385
mat     base_logit = -0.9943   P = 0.0774
```

Now pretend a varied preference dataset was collected. The annotators were action-verb people. They consistently preferred completions where the next token was `sat` and consistently disliked `cat`. The reward model trained on those preferences outputs:

```
R(sat) = +3.0      strongly preferred
R(cat) = -2.0      mildly disliked
R(*)   =  0.0      everything else
```

Sweep `β` (the KL-leash tightness) and watch the policy distribution shift:

```
 beta  leash         top    P(sat)   P(cat)   KL(π‖π_ref)
─────────────────────────────────────────────────────────
10.00  very tight    cat    0.2307   0.3557        0.0169
 1.00  tight         sat    0.8791   0.0151        1.2634
 0.30  loose         sat    0.9999   0.0000        1.7840
 0.10  slack         sat    1.0000   0.0000        1.7852
```

Read across the rows. Three things are happening at once.

1. **At `β = 10`** the leash is tight. The reward `R/β = 0.3` is too small to overcome `cat`'s pretraining advantage. The base model wins. KL divergence from the reference is `0.017` — barely moved.
2. **At `β = 1`** the reward gets multiplied by `1`. `R(sat)/β = 3.0` is enough to flip the top token from `cat` to `sat`. The distribution has now drifted `1.26` nats of KL from the base. RLHF has done its job.
3. **At `β = 0.1`** the leash is essentially gone. `R(sat)/β = 30` makes `sat` the only token with non-vanishing probability. The policy has collapsed onto the reward's favorite. KL plateaus around `1.78`. This is the *reward hacking* regime.

This is what people mean when they say *"RLHF tuned the model toward $X$."* It is literally adding a bias vector to the logits, scaled by `1/β`, computed by a network trained on preference pairs. Nothing more mysterious.

### Varied Datasets Produce Varied Outputs

This is the lever the question was actually about. The `R(y)` vector above came from one specific set of annotators. Change the annotator pool — vary the dataset — and you get a different `R(y)`, which means a different additive shift, which means a different output distribution from the same base model.

Three plausible annotator pools, three different reward signatures, three different deployed models:

| Annotator pool | Likely `R(PostgreSQL)` | Likely `R(CockroachDB)` | Production effect |
|---|---|---|---|
| Hobbyist web devs | `+2.0` | `-0.5` | Reinforces the pretraining bias toward Postgres |
| Distributed-systems engineers | `-0.5` | `+2.5` | Flips the recommendation for "scale" prompts |
| Mixed enterprise + indie pool | `+0.5` | `+0.5` | Both surface, model hedges between them |
| RLAIF (model-as-judge) | inherits judge's bias | inherits judge's bias | Whatever the judge model thought, amplified |

Every cell in that table is a hyperparameter you can move just by changing who you put in the annotation pool — or, with [Constitutional AI / RLAIF](https://arxiv.org/abs/2212.08073), which model you point at the preference-labeling job. **The output bias of a deployed LLM is partly a sourcing decision masquerading as a training decision.**

Combine that with the KL-leash sweep above and you have the full toolkit:

- *Who* you ask for preferences sets the **direction** of `R(y)`.
- *How tight* you set `β` sets the **magnitude** of the shift.
- *What the base model already knew* sets the **support** within which RLHF can move probability mass.

### What RLHF Cannot Do

What this means in practice: RLHF is fundamentally a *re-weighting* of the pretrained distribution. It can amplify tokens the base model already knew about. It can suppress others. **It cannot teach the model genuinely new content.** A token with base logit near `−∞` (never seen in pretraining) cannot be reached for any finite reward — the KL term blows up the moment you try.

> **Fine-tuning doesn't paint a new picture. It chooses which parts of the existing picture to spotlight.**

If CockroachDB is a faint pixel in the base distribution, no reasonable amount of RLHF can make it a default recommendation. If Postgres is already a dominant pixel, RLHF tends to make it more so — because annotators who don't know what else to suggest will rate Postgres-mentioning responses higher than Postgres-omitting ones. Bias compounds *exactly* where it was already strong.

Newer variants (DPO, IPO, KTO, RLAIF) skip the explicit reward model and optimize directly on preference pairs — but the structural conclusion is the same: fine-tuning sharpens the priors. It does not invent them.

A useful frame, borrowed from predictive-coding neuroscience: **the priors are inherited. The next update is the only thing anyone gets to author.** It is true of brains and it is true of language models. The pretraining corpus is the model's childhood. Fine-tuning is its adolescence. By the time it ships, the personality is set; you are only tuning style.

---

## Stage 7 — Inference-Time Controls: The Last Tilt

By the time you query the model, the weights are frozen. But there are still several knobs that can tip a near-tied logit race one way or the other.

### Temperature

You met `T` in Stage 4. Temperature divides the logits before softmax. Concrete effect from the demo:

```
T=0.5:  predicted='cat',  top_prob=0.647,  entropy=1.020
T=1.0:  predicted='cat',  top_prob=0.426,  entropy=1.419
T=2.0:  predicted='cat',  top_prob=0.307,  entropy=1.559
```

At `T=0.5`, the model's existing bias is *amplified*: the most probable token under the base distribution gets even more probability mass. At `T=2.0`, the distribution flattens and underdog tokens get a real chance.

A model served at `T=0.0` (deterministic, argmax) will always produce its single most-biased response. The same model at `T=1.5` will produce a diverse range of responses including ones that mention less-common products. **The bias you observe is partially a function of the temperature your provider chose.**

### Top-k and Top-p (Nucleus) Sampling

Most production systems don't sample from the full softmaxed distribution. They truncate it first.

- **Top-k** — only consider the `k` highest-probability tokens.
- **Top-p (nucleus)** — only consider the smallest set of tokens whose cumulative probability ≥ `p`.

If any of CockroachDB's tokens fall outside the top-50 most likely continuations at any position in its name, top-50 sampling can never produce it. **Every token has to clear the truncation gate, every time.** A long product name has to clear it once per token. Stage 1 comes back to bite: fragmented names are not just disadvantaged in probability, they are disadvantaged in *gates passed*.

### System Prompts and Personas

A system prompt is just more tokens in the context. But its position is privileged — the model attends to it on *every* generation step. A system prompt that says *`You are an experienced PostgreSQL DBA`* doesn't force the model to recommend Postgres. It tilts the attention distribution toward Postgres-adjacent regions of embedding space at every layer.

The empirical effect is enormous. The same base model with *`You are a database architect`* vs. *`You are a startup CTO who values simplicity`* will produce systematically different recommendation distributions for the same user question. Same weights. Different attention. Different output.

### Few-Shot Priming, RAG, and Conversation Context

Everything earlier in the conversation is part of the input. If turn 1 mentions Kubernetes, the attention mechanism at turn 4 will weight tokens that co-occurred with Kubernetes in training — including whatever databases the training corpus paired with Kubernetes. A conversation that already named one technology biases recommendations for adjacent ones, because the embeddings of mentioned tokens get pulled into every subsequent context vector.

This is the Stage 2 punchline coming back at inference time. Few-shot examples, prior turns, and RAG-retrieved documents are all the *same* attention-keystore swap: they change which keys and values the answer-position query gets to dot-product against. A RAG retriever that pulls Postgres docs into context is, mechanically, indistinguishable from a user who pasted those same docs as a few-shot example — both end up as keys in the same attention computation. The only difference is who chose them.

This is why *asking the same question fresh* and *asking it after a long conversation* (or with a different RAG index, or with different few-shots) can yield meaningfully different answers from the same model. The weights are identical. The keys are not.

### Decoding Strategies

Greedy decoding (always take the most probable token) maximally surfaces the model's biases. Beam search finds high-probability sequences globally but tends to over-recommend frequent patterns. Stochastic sampling (with `T > 0`) widens the distribution. Repetition penalties artificially suppress tokens the model has already used.

None of these are neutral. Every decoding choice is an editorial choice. The model's "opinion" is partly a function of how you choose to read it.

---

## The Full Picture: Why X Beats Y

So why does a model recommend Database X over Database Y? Stack the stages:

1. **Tokens & Embeddings** — X's name fragments into fewer subwords, and its geometric neighbors in embedding space sit closer to the typical query.
2. **Attention** — the learned `W_Q`/`W_K`/`W_V` route attention toward X's associated tokens for the typical prompt.
3. **Feed-forward** — nonlinear amplification crosses the differentiation threshold for X and not for Y.
4. **LM head + softmax** — even a small logit edge becomes a dominant probability after exponentiation.
5. **Training corpus + loss** — X appeared more often, more recently, in more recommendation-shaped contexts.
6. **Fine-tuning** — human annotators preferred X, often implicitly. RLHF sharpens that preference via a logit shift leashed by β.
7. **Inference-time** — temperature, sampling cutoffs, and system prompts collectively favor whatever the rest of the stack already favors.

None of this requires a conspiracy. No one at the AI company decided *"recommend Postgres."* The bias is emergent — but emergent from machinery so well-characterized that, once you have seen it, you cannot un-see it.

This is also why the bias is so hard to *remove*. Every stage compounds in the same direction. Patching one stage (say, fine-tuning the model to mention more alternatives) without changing the upstream stages (training distribution, embedding geometry) produces a model that *mentions* alternatives but *recommends* Postgres anyway. The leash is too short.

---

## How to See This for Yourself

Theory is cheap. Three concrete probes you can run today, in increasing order of effort. Each one converts "the model is biased" from a vibe into a number.

### Probe 1 — The Logit-Difference Test

Pick any model that exposes logprobs (OpenAI API, vLLM, llama.cpp, `transformers` locally). Prompt it with:

```
The best database for a simple side project is
```

Read off the top-20 logprobs at the next-token position. Note the gap between `#1` and `#2`. Now compute `exp(logprob_1 − logprob_2)` — that ratio is the *probability multiplier softmax just applied* to a training-data difference you cannot otherwise see. In practice, on most production models, you will find ratios of 3×–20× between the leader and the runner-up at the first token of a database name. That is the bias, visualized in one number.

### Probe 2 — The Temperature Sweep

Same prompt. Generate 100 completions at `T=0.2`, 100 at `T=1.0`, 100 at `T=1.5`. Tally which databases appear at the top of the response. The shift in that distribution is, almost exactly, the *shape of the model's prior*. At low temperature you get the mode. At high temperature you get the support. The difference between the two is the answer to "what does this model secretly think when forced to commit?"

### Probe 3 — The Tokenization Tax

Pass your product name and three competitors through the model's tokenizer (`tiktoken` for OpenAI, the `transformers` tokenizer for everything else). Count subword pieces. Now, for a prompt that should plausibly recommend each of them, compute the marginal probability the model assigns to *generating the entire name* (sum of logprobs across all its tokens). Compare across products. The geometric mean ratio is your **tokenization tax** — the multiplicative handicap a long name pays even before any training-distribution effects kick in.

None of these require training a model or reading a paper. They take twenty lines of Python. Bias goes from "vibes" to "I measured it."

---

## The Uncomfortable Truth

There is no fairness mechanism in this pipeline.

The loss function optimizes for prediction accuracy, not for giving every database an equal shot. Attention surfaces what is statistically associated, not what is objectively best. Embeddings reflect the training corpus, not ground truth. RLHF inherits the priors. Inference-time controls amplify or attenuate them but cannot create what isn't already there.

When a model says `I recommend PostgreSQL`, what it is mechanically saying is:

> Given the statistical patterns in my training corpus, refined through billions of gradient updates on a next-token-prediction objective, further shaped by human annotators during fine-tuning, sharpened by softmax, leashed by a KL term, and tilted at the last second by your system prompt and the temperature my provider chose — the token sequence `PostgreSQL` has the highest probability in this context.

That is not a recommendation. It is a reflection of inherited priors.

Understanding this distinction — between a model having an *opinion* and a model reproducing *statistical patterns* — is the first step toward thinking critically about anything an LLM tells you.

And understanding the specific mechanisms is what gives you leverage. Whether you want to use that leverage to make models more honest, or to make them recommend your product, is a separate question. The mechanical "why" of bias lives here. The prescriptive "how to shape it" lives next door, in [`influence-blog.md`](./influence-blog.md).

---

## Appendix: Why the Demo Is the Real Thing

[`attention.py`](./attention.py) runs the entire pipeline above, end to end, in pure numpy. It is not a simplification. It is the exact linear algebra running inside GPT-class models, scaled to the minimum size that makes the mechanics visible. The full visual walkthrough lives in [`README.md`](./README.md); the brief mapping is here:

| Demo | Production LLM |
|------|---------------|
| Random embedding init | Learned token embeddings |
| `Q = X·W_Q, K = X·W_K, V = X·W_V` | Self-attention projections |
| `softmax(QK^T / √d_k) · V` | Scaled dot-product attention |
| GeLU | GeLU (GPT) / SwiGLU (Llama) |
| `context · W_vocab → softmax` | LM head |
| `-log P(target)` | Cross-entropy training loss (SFT) |
| `logits + R/β` | RLHF / DPO policy update (closed-form) |
| `logits / T` | Temperature-scaled sampling |

GPT-4 is this exact pipeline, made wider (`d_model` ~12k vs `8`), deeper (~96 transformer blocks vs `1`), wider in attention (~96 heads vs `1`), and optimized over trillions of tokens.

> **There is no pedagogical gap between what runs in the demo and what runs at inference time when a model recommends PostgreSQL. The matrices are the same shape-class. The gradients flow through the same computational graph. Scale is the only thing between this demo and a model that passes the bar exam.**

What's missing from the demo, and why none of it changes the bias story:

- **Multi-head attention.** Production runs 64+ heads in parallel; each is the same `Q,K,V` operation as the single head shown here.
- **Depth.** 96+ stacked transformer blocks refine representations layer by layer; one is enough to show the mechanism.
- **Positional encoding (RoPE, ALiBi, learned).** Required at scale; unnecessary in a 6-token toy.
- **Layer norm + residual connections.** Gradient stability tricks for deep networks.
- **BPE tokenizer.** The demo splits on whitespace because Stage 1 already explained why subword tokenization matters; running BPE in the demo would just hide the rest of the math.
- **Scale.** Billions of parameters vs. hundreds. A parameter is a parameter.

---

## Further Reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the paper that started it all
- [Stochastic Parrots (Bender et al., 2021)](https://dl.acm.org/doi/10.1145/3442188.3445922) — on the dangers of training distribution
- [TruthfulQA (Lin et al., 2022)](https://arxiv.org/abs/2109.07958) — empirically probing model biases
- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — the canonical RLHF recipe
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — RLHF without the reward model
- [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) — RLAIF and synthetic preferences
- [`attention.py`](./attention.py) — the full pipeline in 200 lines of numpy
- [`README.md`](./README.md) — visual walkthrough of the demo
- Companion piece on shaping what models say: [`influence-blog.md`](./influence-blog.md)

---

*The model isn't recommending. It's sampling. What you call its opinion is the shape of its training distribution, sharpened by softmax, leashed by fine-tuning, and tilted at the last second by the prompt you sent in. The priors are inherited. The next update is yours. Reality is waiting to be compiled — one collapse at a time.*
