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
    # 🎤 BASIC DELIVERY
    # =========================
    wpm = delivery_metrics["words_per_minute"]
    filler_density = delivery_metrics["filler_density"]

    pitch_variation = voice_metrics["pitch_variation"]
    energy_mean = voice_metrics["energy_mean"]

    # pitch variation: too low = flat / monotone, too high = unstable
    if pitch_variation < 22:
        pitch_score = 35
    elif pitch_variation < 30:
        pitch_score = 50
    elif pitch_variation < 40:
        pitch_score = 65
    elif pitch_variation <= 65:
        pitch_score = 82
    elif pitch_variation <= 85:
        pitch_score = 72
    else:
        pitch_score = 55

    # energy mean: low = weak / timid, too high can sound forced
    if energy_mean < 35:
        energy_score = 40
    elif energy_mean < 42:
        energy_score = 55
    elif energy_mean <= 58:
        energy_score = 78
    elif energy_mean <= 65:
        energy_score = 68
    else:
        energy_score = 55

    basic_delivery = (pitch_score + energy_score) / 2

    # =========================
    # 🧠 ADVANCED DELIVERY
    # =========================
    silence_ratio = voice_metrics.get("silence_ratio", 0)
    avg_pause = voice_metrics.get("avg_pause_duration", 0)
    pause_frequency = voice_metrics.get("pause_frequency", 0)
    energy_variation = voice_metrics.get("energy_variation", 0)

    # --- Pause quality ---
    # ideal pauses feel controlled, not absent and not hesitant
    if 0.35 <= avg_pause <= 0.9:
        pause_score = 82
    elif 0.25 <= avg_pause < 0.35:
        pause_score = 65
    elif 0.9 < avg_pause <= 1.25:
        pause_score = 65
    elif avg_pause < 0.18:
        pause_score = 30
    elif avg_pause > 1.25:
        pause_score = 35
    else:
        pause_score = 50

    # --- Silence control ---
    # too little silence = rushed, too much = hesitant
    if 0.10 <= silence_ratio <= 0.26:
        silence_score = 84
    elif 0.07 <= silence_ratio < 0.10:
        silence_score = 65
    elif 0.26 < silence_ratio <= 0.32:
        silence_score = 62
    elif silence_ratio < 0.07:
        silence_score = 28
    elif silence_ratio > 0.32:
        silence_score = 35
    else:
        silence_score = 50

    # --- Pause frequency ---
    # too many pauses can signal broken rhythm / uncertainty
    if pause_frequency < 0.18:
        pause_frequency_score = 45
    elif 0.18 <= pause_frequency <= 0.38:
        pause_frequency_score = 82
    elif 0.38 < pause_frequency <= 0.55:
        pause_frequency_score = 58
    else:
        pause_frequency_score = 35

    # --- Dynamic range / monotone detection ---
    if energy_variation < 4:
        dynamic_score = 25
    elif energy_variation < 7:
        dynamic_score = 45
    elif energy_variation < 11:
        dynamic_score = 62
    elif energy_variation <= 22:
        dynamic_score = 82
    else:
        dynamic_score = 72

    advanced_delivery = (
        pause_score +
        silence_score +
        pause_frequency_score +
        dynamic_score
    ) / 4

    # =========================
    # 🔗 COMBINED DELIVERY
    # =========================
    # make delivery matter more than before
    delivery_score = (0.45 * basic_delivery) + (0.55 * advanced_delivery)

    # =========================
    # 🔗 BASE SCORE
    # =========================
    # slightly increase delivery importance
    base_score = (0.48 * cognitive_score) + (0.52 * delivery_score)

    # =========================
    # 🔥 PENALTIES
    # =========================
    penalty = 0

    # filler penalty
    if filler_density > 0.10:
        penalty += 22
    elif filler_density > 0.05:
        penalty += 14
    elif filler_density > 0.02:
        penalty += 7

    # speaking rate penalty
    if wpm > 185:
        penalty += 14
    elif wpm > 170:
        penalty += 8
    elif wpm < 90:
        penalty += 10
    elif wpm < 105:
        penalty += 6

    # weak idea / content penalties
    if idea_strength < 45:
        penalty += 12
    elif idea_strength < 55:
        penalty += 6

    if coherence < 45:
        penalty += 10
    elif coherence < 55:
        penalty += 5

    if conciseness < 45:
        penalty += 8

    # delivery penalties
    if pitch_variation < 22:
        penalty += 12
    elif pitch_variation < 30:
        penalty += 7

    if energy_variation < 4:
        penalty += 14
    elif energy_variation < 7:
        penalty += 8

    if silence_ratio < 0.07:
        penalty += 12
    elif silence_ratio > 0.32:
        penalty += 10

    if avg_pause < 0.18:
        penalty += 10
    elif avg_pause > 1.25:
        penalty += 8

    if pause_frequency > 0.55:
        penalty += 10
    elif pause_frequency > 0.38:
        penalty += 5

    # =========================
    # 🚨 FAILURE OVERRIDE
    # =========================
    if failure:
        return round(min(base_score, 50), 2)

    # =========================
    # 🚨 WEAK DELIVERY CAPS
    # =========================
    # if delivery is clearly weak, transcript quality cannot rescue it too much
    delivery_red_flags = 0

    if pitch_variation < 30:
        delivery_red_flags += 1
    if energy_variation < 7:
        delivery_red_flags += 1
    if silence_ratio < 0.07 or silence_ratio > 0.32:
        delivery_red_flags += 1
    if avg_pause < 0.18 or avg_pause > 1.25:
        delivery_red_flags += 1
    if pause_frequency > 0.55:
        delivery_red_flags += 1
    if wpm > 185 or wpm < 90:
        delivery_red_flags += 1

    final_score = base_score - penalty

    if delivery_score < 45:
        final_score = min(final_score, 52)
    elif delivery_score < 55:
        final_score = min(final_score, 60)
    elif delivery_red_flags >= 3:
        final_score = min(final_score, 58)
    elif delivery_red_flags >= 2:
        final_score = min(final_score, 64)

    # clamp score
    final_score = max(25, min(95, final_score))

    return round(final_score, 2)