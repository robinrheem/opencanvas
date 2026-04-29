# OpenCanvas

OSS reproduction of **CANVAS: Continuity-Aware Narratives via Visual Agentic Storyboarding** ([arxiv 2604.13452](https://arxiv.org/abs/2604.13452)).

CANVAS is a training-free, multi-agent framework that turns a list of shot descriptions into a coherent storyboard sequence. It enforces continuity through a four-agent pipeline backed by a persistent visual state memory: a global planner, an anchor retriever, an image generator, and a QA-based selector. OpenCanvas reimplements the published algorithms end-to-end against fully open-source backbones.

## Table of contents
- [What this is](#what-this-is)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Story input format](#story-input-format)
- [Output layout](#output-layout)
- [Configuration](#configuration)
- [Pipeline internals](#pipeline-internals)
- [Paper fidelity](#paper-fidelity)
- [Project layout](#project-layout)
- [Development](#development)
- [Hardware notes](#hardware-notes)

## What this is

A self-contained, async-first Python implementation of the CANVAS pipeline. Given a `Story` (title + a list of natural-language shot descriptions, optionally enriched with character/location/prop declarations), OpenCanvas:

1. **Plans** the story by calling four sub-agents in parallel — location clustering (Table 20), character timelines (Table 22), prop timelines (Table 23), and per-shot continuation decisions (Table 24) — followed by per-shot background reasoning (Table 21).
2. **Generates** K candidate frames per shot using a multi-image edit model conditioned on retrieved anchor images (previous frame, character anchors, location anchor, prop anchors).
3. **Selects** the best candidate via VLM Likert scoring on four axes (Table 26).
4. **Updates memory** by extracting visibility bboxes per entity (Tables 27–29), segmenting the subject out of the chosen frame, and storing it on a neutral plate so downstream conditioning carries identity but not background drift.

All four phases (planning, generation, selection, visibility extraction) leverage `asyncio.gather` so concurrent LLM/VLM calls hit the OpenAI-compat endpoint in parallel.

## Architecture

```
Story (JSON)
    │
    ▼
plan() ──── 4 sub-planners + Table 21 backgrounds  (one asyncio.gather)
    │
    ▼
For each shot t:
    │
    ├── retrieve(shot, plan, memory)            ── Algorithm 2: anchor selection
    │       (canonical-vs-recent character rule, prop fallback chain)
    │
    ├── _cached_generate(...)                   ── Diskcache resume layer
    │       diffusers.Flux2KleinPipeline
    │       → K candidate PNGs
    │
    ├── select(candidates, shot, memory)        ── Algorithm 3: K concurrent VLM judges
    │       Likert: shot_alignment, character_consistency,
    │               background_continuity, prop_state_correctness
    │       → chosen frame (argmax overall_score)
    │
    ├── extract_visibility(shot, chosen)        ── Algorithm 4 phase 1
    │       3 concurrent VLM calls (Tables 27/28/29)
    │       → bboxes per visible character/prop, location-visible flag
    │
    └── _refresh_anchors(...)                   ── Algorithm 4 phase 2
            character / prop: bbox-crop + BiRefNet segment + neutral-gray plate
            location: inverse-segment subjects out of frame, neutral-fill silhouettes
            → memory anchors saved to disk
```

`Memory` (`opencanvas/memory.py`) is a dataclass of typed dicts persisted as `manifest.json`:
- `characters[(cid, state)] -> Path` — recent appearance anchors
- `characters_canonical[(cid, state)] -> Path` — author-supplied or pre-seeded canonical anchors
- `locations[lid] -> Path`
- `props[(pid, state)] -> Path`
- `frames[shot_index] -> Path`

`Memory.character(cid, state, prev_state=None)` implements the Algorithm 2 §4 canonical-vs-recent precedence rule (canonical first if appearance changed, recent first otherwise). `Memory.prop(pid, state)` walks the state → `default` → any-prior-state fallback chain.

## Quickstart

**Requirements**
- Python 3.12
- A CUDA-capable GPU (B200, H100, or anything with ≥40 GB free VRAM for FLUX.2 [klein] 4B + ~1 GB for BiRefNet)
- An OpenAI-compatible HTTP endpoint serving a multimodal LLM (Gemma 4 31B is the default; any tool-call-capable VLM works with minor prompt tuning)
- `uv` for dependency management

**Install**

```bash
uv sync
```

If your torch wheel is built against CUDA 13, pin `torchvision` to match:

```bash
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130
```

**Serve the LLM**

OpenCanvas talks to any OpenAI-compat endpoint. The two we test against:

```bash
# Option A — Ollama (simplest)
ollama pull gemma4:31b
ollama serve                                 # http://localhost:11434/v1

# Option B — vLLM (highest throughput; recommended on B200/H100)
vllm serve google/gemma-4-31b-it \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --served-model-name gemma4:31b \
  --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt '{"image": 4}' \
  --port 8999 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

> `--enable-auto-tool-choice --tool-call-parser hermes` is required because pydantic-ai uses tool calling for typed outputs by default. Alternatively, OpenCanvas wraps every typed call in `pydantic_ai.NativeOutput`, so vLLM's `response_format=json_schema` constrained-decoding path is used end-to-end and tool calling is bypassed entirely.

**Run**

```bash
export OPENCANVAS_BASE_URL=http://localhost:8999/v1   # vLLM port; Ollama is :11434
export OPENCANVAS_API_KEY=EMPTY                       # any non-empty string works

# Plan only (no image gen)
opencanvas plan examples/mini_story.json

# Full pipeline
opencanvas run examples/mini_story.json --k 4 -o ./out
```

To pin diffusers to a specific GPU different from the LLM server:

```bash
CUDA_VISIBLE_DEVICES=1 opencanvas run examples/mini_story.json --device cuda -o ./out
```

## Story input format

```json
{
  "title": "Ada and the Warehouse Journal",
  "shots": [
    "Ada, a 30-year-old engineer in a gray raincoat, pushes open the heavy metal door of an abandoned warehouse at dusk. Wide shot, low light.",
    "Interior: Ada walks between towering industrial shelves. Dust motes drift in shafts of amber evening light. Medium shot from behind.",
    "Close-up of Ada's hand lifting a weathered leather journal from a steel crate labeled 1987.",
    "Ada sits on the warehouse floor, raincoat now dust-streaked, flipping through the journal. Medium close-up.",
    "Exterior again: Ada, journal tucked under her arm, steps back into the rain outside the warehouse. Same wide framing as shot 1 but darker."
  ],
  "characters": [
    {
      "id": "char-ada",
      "name": "Ada",
      "description": "30-year-old engineer, short dark hair, gray raincoat.",
      "appearance_states": ["default", "dust-streaked"],
      "reference_image": "/path/to/ada_canonical.png"
    }
  ],
  "locations": [
    {"id": "loc-warehouse-ext", "name": "Warehouse exterior", "description": "..."},
    {"id": "loc-warehouse-int", "name": "Warehouse interior", "description": "..."}
  ],
  "props": [
    {
      "id": "prop-journal",
      "name": "Leather journal",
      "description": "Weathered 1987-era leather-bound journal.",
      "states": ["intact", "burned"]
    }
  ]
}
```

Only `title` and `shots` are required. Declared `characters`, `locations`, and `props` give the planner stable IDs to anchor against. `appearance_states` and `prop.states` serve as a vocabulary hint to keep the planner from inventing variant state names (`burned` vs `sort_of_burned`). `reference_image` is copied into `characters_canonical` at run start and used as the canonical anchor whenever the appearance state changes.

## Output layout

A run writes everything to `--out` (default `./out`). Structure:

```
out/
  plan.json                          # full Plan (Tables 20+22+23+24+21 outputs)
  results.json                       # per-shot chosen frame + Likert scores
  candidates/
    shot_0000/cand_0.png  …  cand_K-1.png
    shot_0001/...
    ...
  crops/
    shot_0000/
      char__char-ada__default.png    # bbox crop + segment + neutral plate
      loc__warehouse_exterior.png    # inverse-segment (subjects masked out)
      prop__prop-journal__intact.png
    ...
  memory/
    manifest.json
    characters/
    characters_canonical/
    locations/
    props/
    frames/
      shot_0000.png  …                # the chosen frames in order
```

Re-running with the same `--out` resumes: `Memory.load_or_empty` rehydrates the manifest, and `_cached_generate` short-circuits any shot whose `(shot, anchors, seed, k)` cache key already has all candidate files on disk.

## Configuration

All settings come from `pydantic_settings.BaseSettings` with prefix `OPENCANVAS_`:

| Setting | Env var | Default | Purpose |
|---|---|---|---|
| `base_url` | `OPENCANVAS_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compat endpoint |
| `api_key` | `OPENCANVAS_API_KEY` | `ollama` | Any non-empty token; ignored by Ollama / vLLM |
| `model` | `OPENCANVAS_MODEL` | `gemma4:31b` | Served model name (must match `--served-model-name`) |
| `image_model` | `OPENCANVAS_IMAGE_MODEL` | `black-forest-labs/FLUX.2-klein-4B` | HF repo id for diffusers |
| `device` | `OPENCANVAS_DEVICE` | `cuda` | Torch device for diffusers + per-candidate generators |
| `k_candidates` | `OPENCANVAS_K_CANDIDATES` | `4` | K from Algorithm 1 |
| `num_inference_steps` | `OPENCANVAS_NUM_INFERENCE_STEPS` | `4` | FLUX.2 [klein] is step-distilled |
| `guidance_scale` | `OPENCANVAS_GUIDANCE_SCALE` | `1.0` | FLUX.2 [klein] is guidance-distilled (CFG ignored) |
| `seed` | `OPENCANVAS_SEED` | `42` | Base seed |
| `seed_stride` | `OPENCANVAS_SEED_STRIDE` | `1000` | Per-shot seed offset = `seed + index * stride` |
| `cache_dir` | `OPENCANVAS_CACHE_DIR` | `./cache` | Diskcache directory |
| `out_dir` | `OPENCANVAS_OUT_DIR` | `./out` | Default output directory |
| `segment_model` | `OPENCANVAS_SEGMENT_MODEL` | `birefnet-general` | rembg backbone for subject segmentation |
| `segment_bg_color` | `OPENCANVAS_SEGMENT_BG_COLOR` | `(128, 128, 128)` | Neutral plate color for anchor crops |
| `enable_segmentation` | `OPENCANVAS_ENABLE_SEGMENTATION` | `true` | Set false to use bbox crops without bg removal |

CLI flags `--out`, `--k`, `--model`, `--seed`, `--device` override env vars per invocation.

## Pipeline internals

### Planner (`plan` in `agents.py`)

Five sub-agents, all dispatched concurrently via `asyncio.gather`:

| Sub-agent | Paper table | Output schema |
|---|---|---|
| `_cluster_locations` | Table 20 | `LocationClustering` (per-shot location_id + continuation_mode + name lookup) |
| `_plan_character` (per character) | Table 22 | `CharacterTimeline` (appearance state per shot) |
| `_plan_prop` (per prop) | Table 23 | `PropTimeline` (state + carrier per shot) |
| `_decide_continuation` (per shot t > 0) | Table 24 | `ContinuationDecision` (precise per-shot mode) |
| `_plan_background` (per shot) | Table 21 | `BackgroundPlan` (must_appear, must_not_appear, carried_props) |

Continuation mode in `Shot.continuation_mode` comes from Table 24 (per-shot fine-grained); Table 20's bulk-pass mode is captured separately in `LocationClustering.shot_continuity_mode` for inspection or fallback.

### Anchor retrieval (`retrieve`)

Per Algorithm 2:

- `previous_frame_continuation` → `previous_frame` from memory + character anchors + prop anchors
- `location_reappearance` → `location_ref` from memory + character anchors + prop anchors
- `fresh_location` → character anchors + prop anchors only

Character anchors honor the canonical-vs-recent rule: when the appearance state differs from the previous shot, canonical wins; otherwise recent wins.

Prop anchors walk a three-step fallback: exact `(pid, state)` → `(pid, "default")` → most recent state of `pid` (Algorithm 2 §6 "if prop has appeared previously in the story").

### Generation (`generate` + `_cached_generate`)

`diffusers.Flux2KleinPipeline` is invoked with the ordered anchor images (previous frame first, then characters, then location, then props) plus a Table 25 prompt that describes the shot, lists the anchor roles, and appends the Table 21 background constraints. K candidates are produced with seeds `seed`, `seed+1`, …, `seed+K-1`. Each generation is cached on disk by SHA-256 over `(shot, ordered_anchors, seed, K)` so reruns short-circuit.

### Selection (`select`)

K judge calls run concurrently. Each candidate is scored on four 1–5 axes per Table 26: `shot_alignment`, `character_consistency`, `background_continuity`, `prop_state_correctness`. The argmax over `(sum / 4)` wins.

### Visibility extraction (`extract_visibility`)

Three concurrent VLM checks gated by Algorithm 4:

- Table 27 — characters visible in the frame, with normalized bounding boxes
- Table 28 — location is recognizable (boolean)
- Table 29 — props visible, with bounding boxes

Empty-expected sets short-circuit without an LLM call.

### Anchor refresh (`_refresh_anchors`)

For each visible character or prop with a bbox, OpenCanvas crops the chosen frame, runs BiRefNet via rembg to segment the subject, and composites it onto a neutral mid-gray plate. The result is stored as the per-entity anchor. This is the key mitigation for **background drift**: multi-image edit models like FLUX.2 [klein] 4B condition on every pixel of every reference, so a character cropped while standing in a warehouse pulls warehouse pixels into the next shot. Segmenting the subject onto a neutral plate prevents that.

For the location anchor, OpenCanvas does the inverse: take the chosen frame, segment out every visible character/prop silhouette, and replace those pixels with the same neutral plate. Background geometry between subjects is preserved intact, satisfying Table 28's "avoid including large foreground characters" guideline.

## Discrepancies with the paper

OpenCanvas implements every paper component inside the generation scope. This section enumerates every place where the implementation diverges from the published paper, why, and what the user can do about it.

### Out of scope (intentional)

| Component | Status | Reason |
|---|---|---|
| ContinuityEval scoring framework (Tables 6, 7, 9, 10, 11) | Not implemented | Generation-pipeline reproduction only |
| HardContinuityBench dataset construction | Not implemented | Same |
| Baseline reimplementations (AutoStudio, Story-Iter, Story2Board, Gemini-CT) | Not implemented | Same |
| Result tables 14, 15, 17, 18, 19 | Not reproduced | No baselines to compare against |

### Backbone substitutions (intentional, OSS choice)

| Role | Paper | OpenCanvas |
|---|---|---|
| Image generation | Gemini-3-pro-image (Google API) | FLUX.2 [klein] 4B (Apache 2.0) |
| LLM planner | Gemini-2.5-Flash | Gemma 4 31B Dense (multimodal) |
| VLM judge + visibility | Gemini-2.5-Flash | Gemma 4 31B Dense (same model) |

Effect: results will not numerically match the paper's reported scores. Method and architecture are identical; the rendering and judging models differ. Swap in any other multimodal LLM via `OPENCANVAS_MODEL`; swap diffusers backbones via `OPENCANVAS_IMAGE_MODEL`.

### Hyperparameters (paper unspecified)

| Knob | Paper | OpenCanvas default |
|---|---|---|
| `K` (candidates per shot from Algorithm 1) | Not specified | `4` |
| Diffusion `num_inference_steps` | Not specified | `4` (FLUX.2 [klein] step-distilled) |
| `guidance_scale` | Not specified | `1.0` (FLUX.2 [klein] guidance-distilled) |

All three are configurable via `Settings` / env vars / CLI flags.

### Anchor extraction mechanism (real interpretive choice)

Paper Tables 27 / 29 say "extract a clean visual anchor image" with output JSON like `{"name": "Ethan", "anchor": "character_anchor_image"}` — a **placeholder string** instead of actual image data or coordinates. Algorithm 4 step 2 references "Table ??" (a broken cross-reference in the published PDF). The mechanism is genuinely under-specified.

Three plausible readings:
- **(a)** VLM returns bboxes; orchestrator crops → discrete anchor images per entity.
- **(b)** VLM returns descriptions; a separate edit model regenerates clean isolated anchors.
- **(c)** The chosen frame itself is the anchor; the VLM only gates whether each entity counts as "extracted".

OpenCanvas implements **(a)**. Reasons:
- Paper says "clean visual anchors" (plural, distinct per entity) — argues against (c)'s frame-aliasing.
- Gemma 4 (and Gemini-2.5-Flash) both have native bbox / pointing output — (a) requires no extra model.
- (b) would add an image-gen pass per shot to extraction; the architecture diagram doesn't show this.

If interpretation (a) is wrong, the bboxes from Tables 27/29 would be present but unused in the paper's pipeline.

### Location anchor (Table 28 interpretation)

Table 28's guideline: *"Avoid including large foreground characters when possible."* No explicit mechanism is given.

OpenCanvas inverse-segments: take the chosen frame, segment out every visible character/prop silhouette via BiRefNet, replace those pixels with a neutral mid-gray plate. Background geometry between subjects is preserved intact. Possible alternatives the paper might have used:
- VLM returns a "background bbox" → crop a foreground-free region.
- A separate inpainting model fills subject silhouettes with plausible background.
- The chosen frame is reused as-is (and the guideline is just an aspiration).

OpenCanvas's inverse-segmentation is a defensible compromise but is not paper-prescribed.

### Background-drift mitigation (additive)

Multi-image edit models (FLUX.2 [klein] 4B, IP-Adapter, ReferenceNet) condition on every pixel of every reference. A character cropped while standing in a warehouse pulls warehouse pixels into the next shot regardless of the new shot's text prompt.

OpenCanvas mitigates by compositing each segmented subject onto a neutral mid-gray plate before storing as the memory anchor. The paper does not mention this — Gemini-3-pro-image apparently handles subject-vs-background separation more robustly out of the box. Set `OPENCANVAS_ENABLE_SEGMENTATION=false` to revert to bbox-only crops.

### Algorithm 2 §6 prop fallback

Paper: *"if the prop has appeared previously in the story, retrieve the stored prop anchor"* — under-specified about which state's anchor when the requested state has none.

OpenCanvas: `Memory.prop(pid, state)` walks `state` → `default` state → most recent state of `pid`. The third fallback corresponds to "previously appeared" but is looser than a strict per-state lookup. Tighter readings of the paper would skip the third fallback or require an exact match.

### Continuation mode redundancy (paper has both)

Both Table 20 (location clustering) and Table 24 (continuation decision) emit a `continuity_mode` per shot. Paper has both intentionally: Table 20 is bulk pass during clustering; Table 24 is per-shot fine-grained reasoning.

OpenCanvas captures both in the `Plan` (`LocationClustering.shot_continuity_mode` for Table 20; `Shot.continuation_mode` for Table 24) but uses **only Table 24** for `Shot.continuation_mode`. Algorithm 2 §3 step 3a in the paper describes a precise per-shot continuation reasoning step, which matches Table 24. Table 20's bulk mode is persisted for inspection / cross-validation but not consumed downstream.

### Prompt format adaptations

Paper-faithful: Tables 20, 22, 23, 24, 25, 26 are kept as close to verbatim as possible.

Mechanical adaptations:
- **Input section templating** — Paper Input sections are abstract bullets ("Frame image", "Shot description s_t"); OpenCanvas templates them with `{shot_description}`-style placeholders for Python `.format()` substitution. Strings the LLM sees are concretely different but semantics are preserved.
- **Length hints** — Sub-planner user prompts append `Return appearance_by_shot as a list of length {N}`. Not in paper. Compensates for OSS LLMs occasionally returning wrong-length lists (covered by `_pad` fallback if it still happens).
- **JSON examples removed** — Paper prompts include worked JSON examples; OpenCanvas relies on pydantic-ai's typed-output enforcement (see below) so the example is structurally redundant. The LLM does see the schema (field names + descriptions) via the response_format JSON schema, just not the paper's worked example.

### Tables 27, 28, 29 prompt rewrites

Paper Tables 27/28/29 say "extract anchor image" with placeholder string outputs. OpenCanvas asks for `visible: bool` plus `bbox: BBox | None` instead, because bbox is what the chosen interpretation (a) needs and because VLMs can't emit image data directly. If you adopt interpretation (b) or (c), these prompts would revert closer to verbatim.

### Pydantic-ai `NativeOutput` wrapping

Paper presumably parses JSON from natural-language responses — the prompts include `Return STRICT JSON in the following format` instructions and worked examples.

OpenCanvas wraps every typed call in `pydantic_ai.NativeOutput(SchemaClass)`. This routes through OpenAI's `response_format={"type": "json_schema", ...}` — vLLM and most modern OpenAI-compat backends apply xgrammar-based **constrained decoding** so the model literally cannot emit JSON outside the schema. Stronger guarantee than instruction-following; the model never sees the paper's example, only the schema's field names and descriptions.

Alternative: drop `NativeOutput` to use tool calling (vLLM needs `--tool-call-parser hermes`) or `PromptedOutput` to use prompted-JSON parsing.

### State-vocabulary hint (option B opt-in)

Paper Tables 22 and 23 do not constrain LLM-emitted state names — they let the planner invent identifiers from shot descriptions.

OpenCanvas: when the input `Story` declares `Character.appearance_states` or `Prop.states` with non-default values, `_states_hint` injects `Prefer one of these state names when applicable: [...]` into the user prompt. Reduces hallucinated variants like `"sort_of_burned"` vs `"burned"` on OSS LLMs. Empty / single-`default` declarations skip the hint, so paper-faithful behavior is preserved when the author does not opt in.

### Filename quirks

Memory filenames embed state names: `char__char-ada__gray raincoat.png`. Spaces in state names are preserved verbatim because the planner is free to invent multi-word states. Files load fine via Python; non-Linux tooling may struggle.

### Summary

| Discrepancy | Type | User can override? |
|---|---|---|
| Backbone substitutions | Intentional, OSS | Yes — `OPENCANVAS_MODEL`, `OPENCANVAS_IMAGE_MODEL` |
| K, inference steps, true_cfg_scale | Paper-unspecified | Yes — env / CLI |
| Anchor extraction = bbox + crop | Interpretive | No (would need code change) |
| Location anchor inverse-segment | Interpretive | `OPENCANVAS_ENABLE_SEGMENTATION=false` |
| Subject-segment-onto-plate (drift mitigation) | Additive | `OPENCANVAS_ENABLE_SEGMENTATION=false` |
| Prop fallback to any prior state | Loose interpretation | No (would need code change) |
| Table 24 wins over Table 20 for continuation_mode | Implementation choice | No |
| `.format()` placeholders in prompts | Mechanical | No |
| Length hints in sub-planner prompts | OSS-LLM compensation | No |
| JSON examples removed from prompts | Pydantic-ai enforcement | No |
| Tables 27/28/29 rewritten to ask for bboxes | Downstream of anchor-extraction interpretation | No |
| `NativeOutput` constrained decoding | Stronger than paper's instruction-following | Code change |
| State-vocabulary hint when author opts in | Conditional deviation | Don't declare states → no hint |

## Project layout

```
opencanvas/
  schemas.py     pydantic models, StrEnums, generation cache key
  config.py      pydantic-settings Settings + CACHE_TAG_GENERATE
  memory.py      Visual State Memory dataclass + manifest persistence
  prompts.py     Verbatim Tables 20–29 prompt templates
  agents.py      4 sub-planners + retrieve + generate + select + extract_visibility
  pipeline.py    Algorithm 1 orchestrator (run / run_sync) + diskcache wrapping
  cli.py         typer CLI (run + plan commands)

tests/           36 unit + integration tests, all run on CPU with stubs
examples/        mini_story.json — a 5-shot story exercising recurring locations
```

Total: ~1500 source lines, ~700 test lines.

## Development

```bash
# Install dev deps + pre-commit
uv sync
uv run pre-commit install

# Run tests (no GPU / no LLM required — uses pydantic-ai TestModel + fake diffusers pipeline)
uv run pytest

# Lint / format
uv run pre-commit run -a
```

Tests use `pydantic_ai.models.test.TestModel` to fake the LLM, a stubbed `torch.Generator` to avoid CUDA, and a tiny in-process `FakeImagePipeline` to substitute for diffusers. The full pipeline is exercised end-to-end in `tests/test_pipeline.py` without touching any external service.

## Hardware notes

| Hardware | LLM serving | Diffusers + segmentation | Notes |
|---|---|---|---|
| 1× B200 (180 GB) | vLLM Gemma 4 bf16 (~80 GB) | Co-resident (~30 GB) | Full bf16, 128k context, no quant required |
| 1× H100 (80 GB) | vLLM Gemma 4 FP8 (~33 GB) | Co-resident (~30 GB) | FP8 is Hopper-native; minimal quality loss |
| 2× H100 (80 GB each) | GPU 0: vLLM bf16 | GPU 1: diffusers + rembg | Cleanest split, full bf16 throughout |
| 1× RTX 4090 (24 GB) | Ollama Gemma 4 (CPU offload) | Diffusers tight | Possible; expect slow image gen |

**Two-GPU pattern** (recommended on multi-GPU boxes):

```bash
# Shell 1 — vLLM on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-31b-it \
  --gpu-memory-utilization 0.95 --port 8999 \
  --enable-auto-tool-choice --tool-call-parser hermes ...

# Shell 2 — opencanvas + diffusers + rembg on GPU 1
CUDA_VISIBLE_DEVICES=1 opencanvas run story.json --device cuda -o ./out
```

`CUDA_VISIBLE_DEVICES=N` + `--device cuda` is cleaner than `--device cuda:N` since opencanvas only sees one GPU.

## Citation

If you use OpenCanvas in research or build on top of it, cite the original CANVAS paper:

```bibtex
@article{mondal2026canvas,
  title   = {CANVAS: Continuity-Aware Narratives via Visual Agentic Storyboarding},
  author  = {Mondal, Ishani and Song, Yiwen and Parmar, Mihir and Goyal, Palash and Boyd-Graber, Jordan and Pfister, Tomas and Song, Yale},
  journal = {arXiv preprint arXiv:2604.13452},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
