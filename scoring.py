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
    # 🎤 DELIVERY INPUTS
    # =========================
    wpm = delivery_metrics["words_per_minute"]
    filler_density = delivery_metrics["filler_density"]

    pitch_variation = voice_metrics.get("pitch_variation", 0)
    energy_mean = voice_metrics.get("energy_mean", 0)
    silence_ratio = voice_metrics.get("silence_ratio", 0)
    avg_pause = voice_metrics.get("avg_pause_duration", 0)
    pause_frequency = voice_metrics.get("pause_frequency", 0)
    energy_variation = voice_metrics.get("energy_variation", 0)

    # =========================
    # 🎤 BASIC DELIVERY SCORING
    # =========================
    # Controlled variation is good.
    # Very low = flat. Very high = unstable / messy.
    if 35 <= pitch_variation <= 70:
        pitch_score = 85
    elif 20 <= pitch_variation < 35:
        pitch_score = 65
    elif 70 < pitch_variation <= 90:
        pitch_score = 60
    else:
        pitch_score = 40

    # Energy mean: too low = weak, too high = forced
    if energy_mean < 35:
        energy_score = 40
    elif energy_mean < 42:
        energy_score = 55
    elif energy_mean <= 58:
        energy_score = 80
    elif energy_mean <= 65:
        energy_score = 68
    else:
        energy_score = 55

    basic_delivery = (pitch_score + energy_score) / 2

    # =========================
    # 🧠 ADVANCED DELIVERY
    # =========================
    # Pause quality
    if 0.35 <= avg_pause <= 0.9:
        pause_score = 82
    elif 0.25 <= avg_pause < 0.35:
        pause_score = 65
    elif 0.9 < avg_pause <= 1.2:
        pause_score = 62
    elif avg_pause < 0.18:
        pause_score = 28
    else:
        pause_score = 35

    # Silence control
    if 0.10 <= silence_ratio <= 0.26:
        silence_score = 84
    elif 0.07 <= silence_ratio < 0.10:
        silence_score = 65
    elif 0.26 < silence_ratio <= 0.32:
        silence_score = 58
    elif silence_ratio < 0.07:
        silence_score = 28
    else:
        silence_score = 35

    # Pause frequency / rhythm
    if 0.18 <= pause_frequency <= 0.38:
        pause_frequency_score = 82
    elif 0.38 < pause_frequency <= 0.52:
        pause_frequency_score = 55
    elif pause_frequency < 0.18:
        pause_frequency_score = 45
    else:
        pause_frequency_score = 32

    # Dynamic range / monotone detection
    if energy_variation < 4:
        dynamic_score = 22
    elif energy_variation < 7:
        dynamic_score = 42
    elif energy_variation < 11:
        dynamic_score = 60
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
    delivery_score = (0.42 * basic_delivery) + (0.58 * advanced_delivery)

    # =========================
    # 🔗 BASE SCORE
    # =========================
    # Delivery now matters slightly MORE than content
    base_score = (0.45 * cognitive_score) + (0.55 * delivery_score)

    # =========================
    # 🔥 PENALTIES
    # =========================
    penalty = 0

    # Filler penalty
    if filler_density > 0.10:
        penalty += 22
    elif filler_density > 0.05:
        penalty += 14
    elif filler_density > 0.02:
        penalty += 7

    # Speaking rate penalty
    if wpm > 185:
        penalty += 14
    elif wpm > 170:
        penalty += 8
    elif wpm < 90:
        penalty += 10
    elif wpm < 105:
        penalty += 6

    # Weak content penalties
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

    # =========================
    # ⚠️ VOICE STABILITY PENALTY
    # =========================
    stability_penalty = 0

    if pitch_variation > 90:
        stability_penalty += 12
    elif pitch_variation > 75:
        stability_penalty += 6

    # Chaotic pitch + hesitation is especially bad
    if pitch_variation > 70 and silence_ratio > 0.25:
        stability_penalty += 10

    # Very flat pitch is also bad
    if pitch_variation < 22:
        stability_penalty += 10
    elif pitch_variation < 30:
        stability_penalty += 5

    # =========================
    # 🚨 HESITATION PENALTY
    # =========================
    hesitation_penalty = 0

    if silence_ratio > 0.30:
        hesitation_penalty += 12
    elif silence_ratio > 0.26:
        hesitation_penalty += 6

    if avg_pause > 1.2:
        hesitation_penalty += 10
    elif avg_pause > 0.9:
        hesitation_penalty += 5

    if pause_frequency > 0.60:
        hesitation_penalty += 10
    elif pause_frequency > 0.45:
        hesitation_penalty += 5

    # Rushing penalties
    if silence_ratio < 0.07:
        hesitation_penalty += 10
    if avg_pause < 0.18:
        hesitation_penalty += 8

    # Monotone penalties
    if energy_variation < 4:
        hesitation_penalty += 14
    elif energy_variation < 7:
        hesitation_penalty += 8

    penalty += stability_penalty
    penalty += hesitation_penalty

    # =========================
    # 🚨 FAILURE OVERRIDE
    # =========================
    if failure:
        return round(min(base_score, 50), 2)

    # =========================
    # 🎯 FINAL SCORE
    # =========================
    final_score = base_score - penalty

    # =========================
    # 🚨 WEAK DELIVERY CAPS
    # =========================
    delivery_red_flags = 0

    if pitch_variation < 30 or pitch_variation > 90:
        delivery_red_flags += 1
    if energy_variation < 7:
        delivery_red_flags += 1
    if silence_ratio < 0.07 or silence_ratio > 0.30:
        delivery_red_flags += 1
    if avg_pause < 0.18 or avg_pause > 1.2:
        delivery_red_flags += 1
    if pause_frequency > 0.60:
        delivery_red_flags += 1
    if wpm > 185 or wpm < 90:
        delivery_red_flags += 1

    if delivery_score < 42:
        final_score = min(final_score, 48)
    elif delivery_score < 50:
        final_score = min(final_score, 55)
    elif delivery_score < 58:
        final_score = min(final_score, 62)
    elif delivery_red_flags >= 3:
        final_score = min(final_score, 55)
    elif delivery_red_flags >= 2:
        final_score = min(final_score, 61)

    # Clamp final score
    final_score = max(25, min(95, final_score))

    return round(final_score, 2)