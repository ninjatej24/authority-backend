"""Deterministic scenario intelligence profiles for Authority Milestone 9."""

from __future__ import annotations

from dataclasses import dataclass


DIMENSIONS = ("command", "clarity", "composure", "presence", "persuasion", "structure")
METRICS = (
    "pace",
    "fillers",
    "pause_ownership",
    "terminal_endings",
    "dynamic_emphasis",
    "projection",
    "structure",
    "specificity",
    "rambling",
    "certainty_language",
    "hedging",
    "opening",
    "closing",
    "confidence_drops",
    "monotony",
)
TRAITS = (
    "confident",
    "composed",
    "credible",
    "trustworthy",
    "warm",
    "commanding",
    "persuasive",
    "energetic",
    "calm",
    "rushed",
    "flat",
    "hesitant",
    "leadership_ready",
    "interview_ready",
    "executive_presence",
    "structured_thinker",
    "clear_communicator",
)

WEIGHT_VERSION = "scenario_weights_v1"
BASE_DIMENSION_WEIGHTS = {
    "command": 0.22,
    "clarity": 0.20,
    "composure": 0.17,
    "presence": 0.15,
    "persuasion": 0.14,
    "structure": 0.12,
}
NEUTRAL_METRIC_WEIGHTS = {metric: 1.0 for metric in METRICS}
NEUTRAL_TRAIT_WEIGHTS = {trait: 1.0 for trait in TRAITS}
NEUTRAL_SCALING = {
    "filler_penalty": 1.0,
    "rambling_penalty": 1.0,
    "monotony_penalty": 1.0,
    "rising_ending_penalty": 1.0,
    "audio_quality_penalty": 1.0,
    "short_speech_penalty": 1.0,
    "low_confidence_penalty": 1.0,
    "mid_recording_collapse_penalty": 1.0,
}
NEUTRAL_BONUS_SCALING = {
    "opening_strength": 1.0,
    "ending_strength": 1.0,
    "consistency_bonus": 1.0,
}


@dataclass(frozen=True)
class ScenarioProfile:
    scenario_id: str
    description: str
    primary_dimensions: tuple[str, ...]
    secondary_dimensions: tuple[str, ...]
    least_important_dimensions: tuple[str, ...]
    dimension_weights: dict[str, float]
    metric_weights: dict[str, float]
    trait_weights: dict[str, float]
    penalty_multipliers: dict[str, float]
    bonus_multipliers: dict[str, float]
    coaching_priorities: tuple[str, ...]
    report_emphasis: tuple[str, ...]
    share_card_behaviour: str
    expected_speaking_style: tuple[str, ...]
    confidence_modifiers: dict[str, float]
    audio_tolerance: str


def _weights(**overrides: float) -> dict[str, float]:
    weights = dict(BASE_DIMENSION_WEIGHTS)
    weights.update(overrides)
    total = sum(weights.values())
    return {key: round(value / total, 4) for key, value in weights.items()}


def _metric(**overrides: float) -> dict[str, float]:
    values = dict(NEUTRAL_METRIC_WEIGHTS)
    values.update(overrides)
    return values


def _traits(**overrides: float) -> dict[str, float]:
    values = dict(NEUTRAL_TRAIT_WEIGHTS)
    values.update(overrides)
    return values


def _penalties(**overrides: float) -> dict[str, float]:
    values = dict(NEUTRAL_SCALING)
    values.update(overrides)
    return values


def _bonuses(**overrides: float) -> dict[str, float]:
    values = dict(NEUTRAL_BONUS_SCALING)
    values.update(overrides)
    return values


SCENARIO_PROFILES: dict[str, ScenarioProfile] = {
    "benchmark": ScenarioProfile(
        "benchmark",
        "Neutral first-impression benchmark with balanced Authority weighting.",
        ("command", "clarity", "composure"),
        ("presence", "persuasion", "structure"),
        (),
        dict(BASE_DIMENSION_WEIGHTS),
        dict(NEUTRAL_METRIC_WEIGHTS),
        dict(NEUTRAL_TRAIT_WEIGHTS),
        dict(NEUTRAL_SCALING),
        dict(NEUTRAL_BONUS_SCALING),
        ("pause_ownership", "declarative_finality", "structure_compression"),
        ("first_impression", "professional_read", "leadership_read"),
        "neutral_public_summary",
        ("clear", "controlled", "credible"),
        {"confidence_delta": 0.0},
        "standard",
    ),
    "interview": ScenarioProfile(
        "interview",
        "Interview answers reward clarity, structure, composure, and concise evidence.",
        ("clarity", "structure", "composure"),
        ("command", "persuasion"),
        ("presence",),
        _weights(command=0.16, clarity=0.25, composure=0.21, presence=0.09, persuasion=0.11, structure=0.18),
        _metric(rambling=1.25, opening=1.2, structure=1.25, specificity=1.15, fillers=1.15, pace=1.1, dynamic_emphasis=0.9),
        _traits(credible=1.25, trustworthy=1.15, structured_thinker=1.3, clear_communicator=1.25, interview_ready=1.35, persuasive=0.9),
        _penalties(rambling_penalty=1.25, filler_penalty=1.15, low_confidence_penalty=1.1),
        _bonuses(opening_strength=1.15, consistency_bonus=1.1),
        ("rambling_reduction", "opening_strength", "structure_compression", "specificity"),
        ("interview_read", "professional_read", "trust_read"),
        "emphasise_hireable_strengths",
        ("answer-first", "structured", "specific", "composed"),
        {"confidence_delta": -0.01},
        "standard",
    ),
    "leadership": ScenarioProfile(
        "leadership",
        "Leadership communication rewards command, composure, presence, and clean endings.",
        ("command", "composure", "presence"),
        ("structure", "persuasion"),
        ("clarity",),
        _weights(command=0.28, clarity=0.15, composure=0.22, presence=0.18, persuasion=0.10, structure=0.07),
        _metric(terminal_endings=1.3, pause_ownership=1.25, confidence_drops=1.2, projection=1.15, dynamic_emphasis=1.1, structure=0.9),
        _traits(commanding=1.35, composed=1.25, credible=1.15, leadership_ready=1.35, executive_presence=1.25, warm=0.95),
        _penalties(rising_ending_penalty=1.25, mid_recording_collapse_penalty=1.2, monotony_penalty=1.1),
        _bonuses(ending_strength=1.2, consistency_bonus=1.15),
        ("command", "pause_ownership", "declarative_finality", "confidence_under_pressure"),
        ("leadership_read", "social_status_read", "first_impression"),
        "emphasise_command_without_private_labels",
        ("decisive", "settled", "floor-owning"),
        {"confidence_delta": -0.01},
        "strict",
    ),
    "sales": ScenarioProfile(
        "sales",
        "Sales communication rewards listener pull, warmth, energy contrast, and clear stakes.",
        ("persuasion", "presence", "clarity"),
        ("composure", "command"),
        ("structure",),
        _weights(command=0.15, clarity=0.17, composure=0.13, presence=0.22, persuasion=0.25, structure=0.08),
        _metric(dynamic_emphasis=1.3, projection=1.25, certainty_language=1.2, specificity=1.15, monotony=1.2, structure=0.9),
        _traits(persuasive=1.35, warm=1.25, energetic=1.25, trustworthy=1.15, clear_communicator=1.05, structured_thinker=0.9),
        _penalties(monotony_penalty=1.25, rising_ending_penalty=1.05, rambling_penalty=1.05),
        _bonuses(opening_strength=1.1, ending_strength=1.1),
        ("dynamic_emphasis", "projection", "certainty_language", "specificity"),
        ("persuasion_read", "trust_read", "emotional_read"),
        "public_safe_persuasion_identity",
        ("engaging", "specific", "energetic", "trust-building"),
        {"confidence_delta": -0.01},
        "standard",
    ),
    "founder_pitch": ScenarioProfile(
        "founder_pitch",
        "Founder pitches reward a controlled path, command, conviction, and memorable proof.",
        ("structure", "command", "persuasion"),
        ("presence", "clarity"),
        ("composure",),
        _weights(command=0.23, clarity=0.15, composure=0.10, presence=0.15, persuasion=0.20, structure=0.17),
        _metric(opening=1.25, closing=1.2, structure=1.25, specificity=1.2, certainty_language=1.2, dynamic_emphasis=1.1),
        _traits(commanding=1.25, persuasive=1.25, credible=1.2, structured_thinker=1.25, energetic=1.1),
        _penalties(rambling_penalty=1.2, rising_ending_penalty=1.15, filler_penalty=1.1),
        _bonuses(opening_strength=1.25, ending_strength=1.2),
        ("structure_compression", "opening_strength", "command", "certainty_language"),
        ("first_impression", "persuasion_read", "professional_read"),
        "emphasise_pitch_readiness",
        ("compressed", "decisive", "specific", "memorable"),
        {"confidence_delta": -0.01},
        "standard",
    ),
    "presentation": ScenarioProfile(
        "presentation",
        "Presentations reward presence, structure, clarity, and memorable emphasis.",
        ("presence", "structure", "clarity"),
        ("persuasion", "command"),
        ("composure",),
        _weights(command=0.14, clarity=0.19, composure=0.12, presence=0.24, persuasion=0.15, structure=0.16),
        _metric(dynamic_emphasis=1.25, projection=1.2, structure=1.15, opening=1.15, closing=1.15, monotony=1.2),
        _traits(energetic=1.2, persuasive=1.15, clear_communicator=1.2, structured_thinker=1.15, flat=1.15),
        _penalties(monotony_penalty=1.2, rambling_penalty=1.1),
        _bonuses(opening_strength=1.15, ending_strength=1.15),
        ("dynamic_emphasis", "projection", "structure_compression", "closing_strength"),
        ("first_impression", "emotional_read", "persuasion_read"),
        "emphasise_memorable_presence",
        ("clear", "varied", "memorable", "easy to follow"),
        {"confidence_delta": -0.01},
        "standard",
    ),
    "meeting": ScenarioProfile(
        "meeting",
        "Meetings reward balanced clarity, composure, concise structure, and enough command.",
        ("clarity", "composure", "command"),
        ("structure", "presence", "persuasion"),
        (),
        _weights(command=0.21, clarity=0.22, composure=0.20, presence=0.12, persuasion=0.10, structure=0.15),
        _metric(pace=1.1, fillers=1.1, pause_ownership=1.1, structure=1.1, rambling=1.15, dynamic_emphasis=0.95),
        _traits(composed=1.2, credible=1.15, trustworthy=1.15, clear_communicator=1.2, commanding=1.05),
        _penalties(rambling_penalty=1.15, filler_penalty=1.1),
        _bonuses(consistency_bonus=1.1),
        ("pace_regulation", "pause_ownership", "structure_compression", "filler_reduction"),
        ("professional_read", "trust_read", "leadership_read"),
        "balanced_work_safe_summary",
        ("concise", "composed", "clear", "useful"),
        {"confidence_delta": 0.0},
        "standard",
    ),
    "podcast": ScenarioProfile(
        "podcast",
        "Podcast speaking rewards warmth, presence, vocal variety, and sustained clarity.",
        ("presence", "clarity", "persuasion"),
        ("composure", "structure"),
        ("command",),
        _weights(command=0.10, clarity=0.21, composure=0.15, presence=0.25, persuasion=0.17, structure=0.12),
        _metric(dynamic_emphasis=1.25, projection=1.1, monotony=1.25, pace=1.1, fillers=1.05, terminal_endings=0.9),
        _traits(warm=1.3, energetic=1.2, trustworthy=1.15, persuasive=1.1, flat=1.2, commanding=0.9),
        _penalties(monotony_penalty=1.25, rising_ending_penalty=0.9, rambling_penalty=1.05),
        _bonuses(consistency_bonus=1.15),
        ("dynamic_emphasis", "presence", "pace_regulation", "filler_reduction"),
        ("emotional_read", "trust_read", "first_impression"),
        "public_safe_warmth_presence",
        ("warm", "varied", "sustained", "easy to keep listening to"),
        {"confidence_delta": 0.0},
        "tolerant",
    ),
}


def get_scenario_profile(scenario: str | None) -> ScenarioProfile:
    scenario_id = (scenario or "benchmark").strip().lower().replace("-", "_")
    return SCENARIO_PROFILES.get(scenario_id, SCENARIO_PROFILES["benchmark"])


def validate_scenario_profile(profile: ScenarioProfile) -> None:
    total = round(sum(profile.dimension_weights.values()), 4)
    if abs(total - 1.0) > 0.002:
        raise ValueError(f"Scenario dimension weights must sum to 1.0, got {total}")
    if any(value < 0 for value in profile.dimension_weights.values()):
        raise ValueError("Scenario dimension weights cannot be negative")
    if any(not 0.5 <= value <= 1.5 for value in profile.penalty_multipliers.values()):
        raise ValueError("Scenario penalty scaling must remain bounded")
    if any(not 0.5 <= value <= 1.5 for value in profile.bonus_multipliers.values()):
        raise ValueError("Scenario bonus scaling must remain bounded")


def apply_scenario_weights(base_weights: dict[str, float], scenario: str | None) -> dict[str, float]:
    profile = get_scenario_profile(scenario)
    validate_scenario_profile(profile)
    if profile.scenario_id == "benchmark":
        return dict(base_weights)
    weights = {dimension: profile.dimension_weights.get(dimension, base_weights[dimension]) for dimension in DIMENSIONS}
    total = sum(weights.values())
    return {dimension: round(weight / total, 4) for dimension, weight in weights.items()}


def calculate_dimension_relevance(dimension: str, scenario: str | None) -> float:
    profile = get_scenario_profile(scenario)
    baseline = BASE_DIMENSION_WEIGHTS.get(dimension, 0.0)
    value = profile.dimension_weights.get(dimension, baseline)
    return round(value / baseline, 3) if baseline else 1.0


def calculate_metric_relevance(metric: str, scenario: str | None) -> float:
    profile = get_scenario_profile(scenario)
    return round(profile.metric_weights.get(metric, 1.0), 3)


def calculate_trait_relevance(trait: str, scenario: str | None) -> float:
    profile = get_scenario_profile(scenario)
    return round(profile.trait_weights.get(trait, 1.0), 3)


def calculate_penalty_scaling(penalty_id: str, scenario: str | None) -> float:
    profile = get_scenario_profile(scenario)
    return round(profile.penalty_multipliers.get(penalty_id, 1.0), 3)


def calculate_bonus_scaling(bonus_id: str, scenario: str | None) -> float:
    profile = get_scenario_profile(scenario)
    return round(profile.bonus_multipliers.get(bonus_id, 1.0), 3)


def major_weight_changes(scenario: str | None) -> list[str]:
    profile = get_scenario_profile(scenario)
    changes: list[str] = []
    for dimension in DIMENSIONS:
        delta = profile.dimension_weights[dimension] - BASE_DIMENSION_WEIGHTS[dimension]
        if abs(delta) >= 0.025:
            direction = "upweighted" if delta > 0 else "downweighted"
            changes.append(f"{dimension}:{direction}:{round(delta, 3)}")
    return changes
