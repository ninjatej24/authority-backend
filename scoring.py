def compute_authority_score(voice_metrics, cognitive_metrics):

    pitch_variation = voice_metrics["pitch_variation"]
    energy = voice_metrics["energy_mean"]

    clarity = cognitive_metrics["clarity"]
    persuasion = cognitive_metrics["persuasion"]
    articulation = cognitive_metrics["articulation"]
    idea_density = cognitive_metrics["idea_density"]
    structure = cognitive_metrics["structure"]

    # Normalize pitch variation (ideal ~40-80)
    pitch_score = min(max((pitch_variation / 80) * 100, 0), 100)

    # Normalize energy (~60 ideal speaking energy)
    energy_score = min(max((energy / 60) * 100, 0), 100)

    delivery_score = (pitch_score + energy_score) / 2

    cognitive_score = (
        clarity +
        persuasion +
        articulation +
        idea_density +
        structure
    ) / 5

    authority_score = (0.4 * delivery_score) + (0.6 * cognitive_score)

    return round(authority_score, 2)