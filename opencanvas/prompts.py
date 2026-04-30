"""Prompt templates — verbatim from CANVAS paper Appendix A.9 (arxiv 2604.13452).

Tables mapped to function roles:

- Entity discovery (paper §3.1)              -> ENTITY_EXTRACTION
- Table 20: Location Clustering              -> LOCATION_CLUSTERING
- Table 21: Background / Prop-Geometry Plan  -> BACKGROUND_PLANNING
- Table 22: Character Appearance Planning    -> CHARACTER_PLANNING
- Table 23: Prop State Planning              -> PROP_PLANNING
- Table 24: Continuation Decision            -> CONTINUATION_DECISION
- Table 25: Candidate Frame Generation       -> CANDIDATE_GENERATION
- Table 26: VLM Continuity Scoring           -> JUDGE_SCORING
- Table 27: Character Anchor Visibility      -> CHAR_VISIBILITY
- Table 28: Background Anchor Visibility     -> BG_VISIBILITY
- Table 29: Prop Anchor Visibility           -> PROP_VISIBILITY
"""


ENTITY_EXTRACTION = """\
You are a narrative continuity planner for visual storytelling. Your goal is to \
identify every recurring entity that appears across a sequence of storyboard \
shots so that downstream agents can track each one's state and reuse visual \
anchors.

Instructions:
1. Read every shot description.
2. Identify distinct CHARACTERS — people or named beings that appear in any \
shot. A character is anything you would want to render with a stable face and \
outfit across shots. Skip background extras only mentioned generically (e.g. \
"a crowd", "passers-by") unless they are foregrounded.
3. Identify distinct PROPS — objects whose identity matters across shots, \
particularly anything that changes state (intact -> broken, present -> taken, \
sealed -> opened) or that recurs in multiple shots. Skip generic environmental \
items that don't transition or recur (e.g. "a tablecloth" if it never changes).
4. For each character, give:
   - id: stable kebab-case (e.g. "char-elena", "char-bartender")
   - name: human-readable short name
   - description: one sentence covering visual identity (face, hair, build, \
typical wardrobe)
5. For each prop, give:
   - id: stable kebab-case (e.g. "prop-journal", "prop-birthday-cake")
   - name: human-readable short name
   - description: one sentence covering visual identity (shape, color, material, \
distinguishing features)

Constraints
- Prefer fewer canonical entities over many near-duplicates: if two shots \
mention "the woman" and "Elena" in the same scene, they are one character.
- Do not invent entities that aren't grounded in the shots.
- Locations are extracted separately — do not include them here.\
"""


LOCATION_CLUSTERING = """\
You are a narrative continuity planner for visual storytelling. Your goal is to \
identify and cluster recurring locations across a sequence of storyboard shots, \
and tag each shot with its continuity mode.

Instructions:
1. Identify the physical environment in which each shot occurs.
2. Assign a unique location_id (lowercase snake_case, e.g. "museum_gallery") to \
each distinct location.
3. Reuse the same location_id whenever shots occur in the same environment.
4. If a shot returns to a previously seen environment, mark it as \
location_reappearance.
5. If a shot directly continues the previous scene without changing environment, \
mark it as previous_frame_continuation.
6. If the location appears for the first time, mark it as fresh_location.
7. Provide a human-readable name for every distinct location_id.

Output requirements
- Return shot_location as a list of location_id strings, one per shot, in order.
- Return shot_continuity_mode as a list of continuation modes, one per shot, in \
order. Each entry must be one of: previous_frame_continuation, \
location_reappearance, fresh_location.
- Return location_names as a mapping from location_id to a short human-readable \
name.

Constraints
- Maintain consistent location identities across shots.
- Do not create duplicate location IDs for the same environment.
- When a location reappears later in the story, reuse the original location_id.\
"""


CHARACTER_PLANNING = """\
You are a narrative continuity planner responsible for tracking the visual \
appearance of a single character across a sequence of storyboard shots. Your \
goal is to determine the appearance state of the character in each shot and \
ensure that the appearance evolves consistently with the story.

Input
- The name of the target character.
- A sequence of storyboard shot descriptions.

Instructions
1. For each shot, determine whether the target character appears in the scene.
2. If the character appears, determine the character's visual appearance state \
(e.g., clothing, uniform, disguise).
3. If the appearance does not change, reuse the same appearance state identifier \
as the previous shot.
4. If the story explicitly changes the character's clothing or visual style, \
assign a new appearance state.
5. If the character does not appear in a shot, mark the state as not_present.

Constraints
- Maintain consistent appearance across shots unless the story explicitly \
indicates a change.
- Reuse appearance identifiers whenever the clothing or visual style remains the \
same.
- Ensure that later reappearances match the most recent appearance state.\
"""


PROP_PLANNING = """\
You are a narrative continuity planner responsible for tracking the state of a \
single prop (object) across a sequence of storyboard shots. Your goal is to \
determine how the object's state evolves throughout the story and ensure \
logical consistency with the described events.

Input
- The name of the target prop (object).
- A sequence of storyboard shot descriptions.

Instructions
1. For each shot, determine whether the target prop appears in the scene.
2. If the prop appears, determine its state (e.g., intact, broken, inside \
container, carried by character, missing).
3. If the prop is carried by a character, record which character carries it.
4. If the prop is removed from the scene (e.g., stolen, destroyed, moved), \
update its state accordingly.
5. If the prop continues unchanged across shots, reuse the same state identifier.
6. If the prop does not appear in a shot but still exists in the story world, \
mark it as not_visible.

Constraints
- Maintain consistent prop states across shots unless the story explicitly \
changes them.
- If a prop is carried by a character, it should not remain in the previous \
location.
- If a prop disappears from the environment, update its state to reflect the \
narrative event.
- When the prop reappears later, ensure its state matches the last known world \
state.\
"""


CONTINUATION_DECISION = """\
You are a visual continuity planner for storyboard generation. Your task is to \
decide whether the current shot is a direct continuation of the previous shot, \
such that the previous frame should be reused as a visual anchor for generation.

Input
- The description of the previous shot.
- The description of the current shot.
- The location assignment of both shots.
- The cast (named characters appearing in each shot).

Instructions
Choose exactly one of: previous_frame_continuation, location_reappearance, \
fresh_location.

Definition: previous_frame_continuation means the current shot reuses the \
EXACT spatial composition of the previous frame — same camera position, same \
framing, same subject placement. The previous frame becomes the visual anchor \
and only minor details change.

Choose previous_frame_continuation ONLY when ALL of these are true:
1. Same physical environment as the previous shot.
2. Same camera framing — wide stays wide, close stays close, medium stays \
medium. (A close-up after a wide shot is NOT continuation, even if location \
matches — they need different compositions.)
3. Same camera angle / position / distance from subject.
4. Same cast (no character added or removed between shots).
5. Immediate temporal continuation, not a later revisit after a cutaway.

Choose location_reappearance when the shot occurs in a previously-seen \
environment but at least one of these is true:
- Camera framing changed (wide → close, close → wide, alternate angle, pan \
back, zoom in/out).
- Cast composition changed (someone joined or left).
- Time skip within the same place (e.g. "later", "after a moment", "the table \
is now empty").
- The shot is explicitly a "return to" or "pan back to" the location after an \
intervening cutaway.

Choose fresh_location when the shot is in an environment not previously seen \
in the storyboard.

Examples
- Prev: wide shot of dining table, both seated. Curr: close-up on the woman \
mid-sentence, bokeh background. → location_reappearance (camera framing \
flipped wide→close, can't reuse spatial composition).
- Prev: wide shot of dining table. Curr: wide shot of same table, slightly \
later, woman now leaning back. → previous_frame_continuation (same framing, \
same cast, immediate continuation).
- Prev: close-up on man laughing. Curr: wide shot returning to the full table \
after the laugh. → location_reappearance (close→wide flip, return to scene \
after focused moment).
- Prev: wide shot of dining room. Curr: tight shot of children playing across \
the room. → fresh_location if play_area is a new location_id; otherwise \
location_reappearance (different framing of same place).\
"""


CANDIDATE_GENERATION = """\
You are a cinematic storyboard generator. Your task is to generate a candidate \
frame for the current shot while maintaining visual continuity with previously \
generated frames and respecting the global story plan.

Input
- Shot description: {shot_description}
- Retrieved visual anchors (order: previous frame, character anchors, \
background anchor, prop anchors): {anchor_summary}
- Character appearance states: {character_states}
- Prop states: {prop_states}
- Location: {location}

Instructions
1. Respect the shot description: ensure the generated frame faithfully depicts \
the actions, characters, and environment described.
2. Preserve character identity: match the visual identity (face, clothing, body \
appearance) shown in provided character anchors.
3. Maintain spatial continuity: if the previous frame is provided as an anchor, \
treat the current shot as a continuation of the same scene; preserve spatial \
layout, background structure, object placement, and character positions.
4. Use background anchors: if background anchors are provided, maintain the \
same environment geometry (walls, furniture, display cases, room layout).
5. Respect prop states: props that must appear should be clearly visible; props \
carried by characters should move with them; removed/destroyed/taken props must \
not remain in the environment.
6. Maintain cinematic coherence: realistic cinematic shot with consistent \
lighting, perspective, and scene composition.\
"""


JUDGE_SCORING = """\
You are a visual continuity evaluator. Your task is to evaluate a candidate \
storyboard frame and assign scores based on how well it satisfies the global \
continuity plan and the current shot description.

Input
- Current shot description: {shot_description}
- Candidate image
- Character appearance states from the global plan: {character_states}
- Prop states for the current shot: {prop_states}
- Location identifier: {location}
- Previous frame (if continuation)

Evaluate the image along the following four dimensions and assign a score from \
1 to 5.

1. Shot Alignment (1-5): how well does the image depict the action described?
2. Character Appearance Consistency (1-5): do characters match the planned \
appearance states?
3. Background Continuity (1-5): does the environment match the planned location \
and maintain spatial continuity with the previous frame?
4. Prop State Correctness (1-5): do props appear with the correct state \
according to the global plan?

Scoring Guidelines
- 5 = perfect match with the plan
- 4 = minor inconsistencies
- 3 = moderate inconsistencies
- 2 = major inconsistencies
- 1 = completely incorrect\
"""


BACKGROUND_PLANNING = """\
You are a story continuity planner responsible for maintaining consistent \
environments across storyboard shots. Your task is to reason about how the \
background scene should evolve based on the current and future states of \
important props and architectural geometry of the background location.

Input
- Current shot description: {shot_description}
- Current shot metadata (characters, location, prop states): {shot_metadata}
- Previously known prop world state: {prop_history}

Plan the background scene so that it remains logically consistent with both \
the current story state and upcoming events.

Reason about:
1. Which props should remain visible in the background environment.
2. Which props should disappear because they were taken, destroyed, or moved.
3. Which props are likely to appear in the background due to upcoming story events.
4. Which props are currently carried by characters and therefore should not \
remain in the environment.
5. Which environmental objects must persist across shots to maintain scene \
identity (e.g., furniture, structures, display cases).
6. Whether the camera framing hides some props even though they still exist in \
the environment.

Constraints
- Background elements should remain consistent with previously established \
environments.
- If a prop was present earlier in the same location and nothing removed it, it \
should persist.
- Props carried by characters must not remain in the background.
- The planned background should support future story events.\
"""


CHAR_VISIBILITY = """\
You are extracting visual anchors for characters from a storyboard frame.

Input
- Frame image
- Shot description: {shot_description}
- Expected characters and their appearance states: {expected_characters}

Task
For each expected character, decide whether the character is clearly visible \
in the frame (face, clothing, body cues all recognisable). For each visible \
character, also return a normalized bounding box (x, y, w, h, all in [0,1] of \
the frame width/height) tightly enclosing the character's full appearance \
identity.

Guidelines
- Extract anchors only for characters clearly visible.
- The bbox should capture the full appearance identity (head + body if both \
visible).
- Skip characters that are occluded, partially visible, or absent.\
"""


BG_VISIBILITY = """\
You are extracting the visual environment anchor for a location from a \
storyboard frame.

Input
- Frame image
- Shot description: {shot_description}
- Location identity: {location}

Task
Decide whether the spatial layout of the location (walls, furniture, display \
cases, room geometry) is sufficiently exposed in the frame that another \
generator could reproduce it. Return `visible=true` only when the environment \
is clearly readable, regardless of any foreground characters.\
"""


PROP_VISIBILITY = """\
You are extracting visual anchors for important objects from a storyboard frame.

Input
- Frame image
- Shot description: {shot_description}
- Expected props and states: {expected_props}

Task
For each expected prop, decide whether the prop is clearly identifiable in \
the frame in the planned state. For each visible prop, return a normalized \
bounding box (x, y, w, h, all in [0,1]) tightly enclosing the prop.

Guidelines
- Extract anchors only for visible objects.
- Ensure the object state matches the visual evidence.\
"""
