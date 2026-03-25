def compute_authority_score(voice_metrics, cognitive_metrics, delivery_metrics):

    # =========================
    # 🧠 COGNITIVE SCORES
    # =========================
    clarity = cognitive_metrics["clarity"]["score"]
    persuasion = cognitive_metrics["persuasion"]["score"]
    coherence = cognitive_metrics["coherence"]["score"]
    idea_strength = cognitive_metrics["idea_strength"]["score"]
    conciseness = cognitive_metrics["conciseness"]["score"]

    failure = cognitive_metrics.get("failure", False)

    cognitive_score = (
        clarity +
        persuasion +
        coherence +
        idea_strength +
        conciseness
    ) / 5

    # =========================
    # 🎤 DELIVERY (BASIC)
    # =========================
    wpm = delivery_metrics["words_per_minute"]
    filler_density = delivery_metrics["filler_density"]

    pitch_variation = voice_metrics["pitch_variation"]
    energy = voice_metrics["energy_mean"]

    pitch_score = min(max((pitch_variation / 80) * 100, 0), 100)
    energy_score = min(max((energy / 60) * 100, 0), 100)

    base_delivery = (pitch_score + energy_score) / 2

    # =========================
    # 🧠 ADVANCED DELIVERY
    # =========================
    silence_ratio = voice_metrics.get("silence_ratio", 0)
    avg_pause = voice_metrics.get("avg_pause_duration", 0)
    energy_variation = voice_metrics.get("energy_variation", 0)

    # --- Pause quality (ideal ~0.5–1.2s) ---
    if 0.4 <= avg_pause <= 1.2:
        pause_score = 85
    elif avg_pause < 0.25:
        pause_score = 40 # rushed, no pauses
    elif avg_pause > 1.5:
        pause_score = 50 # hesitant
    else:
        pause_score = 70

    # --- Silence control ---
    if silence_ratio < 0.05:
        silence_score = 40 # rushing
    elif silence_ratio > 0.35:
        silence_score = 50 # too hesitant
    else:
        silence_score = 80

    # --- Dynamic range (monotone detection) ---
    if energy_variation < 5:
        dynamic_score = 40 # monotone
    elif energy_variation < 10:
        dynamic_score = 65
    else:
        dynamic_score = 85

    advanced_delivery = (pause_score + silence_score + dynamic_score) / 3

    # =========================
    # 🔗 COMBINED DELIVERY
    # =========================
    delivery_score = (0.6 * base_delivery) + (0.4 * advanced_delivery)

    # =========================
    # 🔗 BASE SCORE
    # =========================
    base_score = (0.55 * cognitive_score) + (0.45 * delivery_score)

    # =========================
    # 🔥 PENALTIES
    # =========================
    penalty = 0

    # filler penalty
    if filler_density > 0.10:
        penalty += 20
    elif filler_density > 0.05:
        penalty += 12
    elif filler_density > 0.02:
        penalty += 6

    # speaking rate penalty
    if wpm > 185:
        penalty += 12
    elif wpm > 165:
        penalty += 6
    elif wpm < 95:
        penalty += 8

    # weak idea
    if idea_strength < 50:
        penalty += 10

    # poor coherence
    if coherence < 50:
        penalty += 8

    # monotone extra penalty
    if energy_variation < 5:
        penalty += 8

    # no pauses (rushed speech)
    if silence_ratio < 0.05:
        penalty += 8

    # too many pauses (hesitation)
    if silence_ratio > 0.35:
        penalty += 6

    # =========================
    # 🚨 FAILURE OVERRIDE
    # =========================
    if failure:
        return round(min(base_score, 55), 2)

    # =========================
    # 🎯 FINAL SCORE
    # =========================
    final_score = base_score - penalty

    final_score = max(30, min(95, final_score))

    return round(final_score, 2)