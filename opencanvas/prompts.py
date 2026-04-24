"""Prompt templates — verbatim from CANVAS paper Appendix A.9 (arxiv 2604.13452).

Tables mapped to function roles:

- Table 20: Location Clustering            -> LOCATION_CLUSTERING
- Table 22: Character Appearance Planning  -> CHARACTER_PLANNING
- Table 23: Prop State Planning            -> PROP_PLANNING
- Table 24: Continuation Decision          -> CONTINUATION_DECISION
- Table 25: Candidate Frame Generation     -> CANDIDATE_GENERATION
- Table 26: VLM Continuity Scoring         -> JUDGE_SCORING
- Table 27: Character Anchor Extraction    -> CHAR_ANCHOR_EXTRACT
- Table 28: Background Anchor Extraction   -> BG_ANCHOR_EXTRACT
- Table 29: Prop Anchor Extraction         -> PROP_ANCHOR_EXTRACT
"""

LOCATION_CLUSTERING = """\
You are a narrative continuity planner for visual storytelling. Your goal is to \
identify and cluster recurring locations across a sequence of storyboard shots.

Instructions:
1. Identify the physical environment in which each shot occurs.
2. Assign a unique location_id to each distinct location.
3. Cluster shots that occur in the same environment under the same location_id.
4. If a shot returns to a previously seen environment, mark it as \
location_reappearance.
5. If a shot directly continues the previous scene without changing environment, \
mark it as previous_frame_continuation.
6. If the location appears for the first time, mark it as fresh_location.

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
- The characters present in both shots.
- The prop states in both shots.

Instructions
Determine whether the current shot should be treated as \
previous_frame_continuation, location_reappearance, or fresh_location.

Mark the current shot as previous_frame_continuation only if the current shot \
depends on preserving the spatial structure of the immediately previous frame.

Reason about:
1. Whether the previous shot and current shot occur in the same physical \
environment.
2. Whether the current shot is an immediate temporal continuation rather than a \
later revisit.
3. Whether the spatial arrangement of the scene should remain consistent across \
the cut.
4. Whether important background geometry must be preserved (walls, doors, \
furniture, display cases, tables, platforms, or room layout).
5. Whether character positions, object placements, or scene composition depend \
on the previous frame.
6. Whether the current shot is a close-up, zoom-in, alternate camera angle, or \
tighter crop of the same ongoing scene.
7. Whether there are state changes in the scene that still require maintaining \
the same base spatial environment.

Rules:
- Choose previous_frame_continuation if the scene is still unfolding in the \
same environment and preserving the exact spatial setup from the previous frame \
is important.
- Choose location_reappearance if the shot returns to a known location after an \
intervening scene or temporal gap.
- Choose fresh_location if the shot occurs in a new environment not previously \
shown.
- Even if some props or characters change state, still choose \
previous_frame_continuation when the underlying scene geometry should remain \
continuous.\
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


CHAR_ANCHOR_EXTRACT = """\
You are extracting visual anchors for characters from a storyboard frame.

Input
- Frame image
- Shot description: {shot_description}
- Character appearance states: {character_states}

Task
Identify characters visible in the frame. For each clearly visible character, \
produce a description that captures their current appearance (face, clothing, \
hairstyle, and overall identity) together with the appearance_state label from \
the plan.

Guidelines
- Extract anchors only for characters clearly visible.
- Anchors should capture the full appearance identity.\
"""


BG_ANCHOR_EXTRACT = """\
You are extracting the visual environment anchor for a location from a \
storyboard frame.

Input
- Frame image
- Shot description: {shot_description}
- Location identity: {location}

Task
Identify the environment and describe the spatial layout of the scene in a way \
that another generator could reproduce it (walls, furniture, display cases, \
room layout).

Guidelines
- The anchor should capture the environment layout.
- Avoid including large foreground characters when possible.\
"""


PROP_ANCHOR_EXTRACT = """\
You are extracting visual anchors for important objects from a storyboard frame.

Input
- Frame image
- Shot description: {shot_description}
- Object state descriptions: {prop_states}

Task
Identify objects that appear in the frame and describe visual anchors \
representing their current state.

Guidelines
- Extract anchors only for visible objects.
- Ensure the object state matches the visual evidence.\
"""
