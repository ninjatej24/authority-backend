def clamp(value, low, high):
    return max(low, min(high, value))


def compute_authority_score(voice_metrics, cognitive_metrics, delivery_metrics):
    # =========================
    # 🧠 CONTENT / TRANSCRIPT
    # =========================
    clarity = cognitive_metrics["clarity"]["score"]
    persuasion = cognitive_metrics["persuasion"]["score"]
    coherence = cognitive_metrics["coherence"]["score"]
    idea_strength = cognitive_metrics["idea_strength"]["score"]
    conciseness = cognitive_metrics["conciseness"]["score"]
    failure = cognitive_metrics.get("failure", False)

    content_score = (
        clarity +
        persuasion +
        coherence +
        idea_strength +
        conciseness
    ) / 5

    # =========================
    # 🎤 RAW DELIVERY INPUTS
    # =========================
    wpm = delivery_metrics.get("words_per_minute", 0)
    filler_density = delivery_metrics.get("filler_density", 0)

    pitch_variation = voice_metrics.get("pitch_variation", 0)
    energy_mean = voice_metrics.get("energy_mean", 0)
    energy_variation = voice_metrics.get("energy_variation", 0)
    silence_ratio = voice_metrics.get("silence_ratio", 0)
    avg_pause = voice_metrics.get("avg_pause_duration", 0)
    pause_frequency = voice_metrics.get("pause_frequency", 0)

    # =========================
    # 🎯 DELIVERY SUBSCORES
    # =========================

    # 1. Pace score
    # Ideal conversational authority zone is moderate, not too slow, not rushed.
    if 115 <= wpm <= 155:
        pace_score = 85
    elif 105 <= wpm < 115 or 155 < wpm <= 170:
        pace_score = 72
    elif 95 <= wpm < 105 or 170 < wpm <= 180:
        pace_score = 55
    elif 85 <= wpm < 95 or 180 < wpm <= 195:
        pace_score = 38
    else:
        pace_score = 25

    # 2. Pause control score
    # Controlled pauses are good; lots of hesitating or no pauses at all are bad.
    if 0.30 <= avg_pause <= 0.75:
        pause_control_score = 85
    elif 0.22 <= avg_pause < 0.30 or 0.75 < avg_pause <= 1.0:
        pause_control_score = 68
    elif 0.16 <= avg_pause < 0.22 or 1.0 < avg_pause <= 1.3:
        pause_control_score = 48
    else:
        pause_control_score = 28

    # 3. Rhythm score
    # Too many pauses often signals broken flow / hesitation.
    if 0.18 <= pause_frequency <= 0.35:
        rhythm_score = 85
    elif 0.35 < pause_frequency <= 0.45:
        rhythm_score = 65
    elif 0.45 < pause_frequency <= 0.55:
        rhythm_score = 45
    elif pause_frequency < 0.18:
        rhythm_score = 50
    else:
        rhythm_score = 28

    # 4. Vocal control score
    # Moderate pitch variation is best. Too flat = dull. Too chaotic = unstable.
    if 32 <= pitch_variation <= 60:
        vocal_control_score = 85
    elif 24 <= pitch_variation < 32 or 60 < pitch_variation <= 72:
        vocal_control_score = 65
    elif 18 <= pitch_variation < 24 or 72 < pitch_variation <= 85:
        vocal_control_score = 45
    else:
        vocal_control_score = 25

    # 5. Energy control score
    # Need enough energy to sound engaged, but not forced.
    energy_base = 0
    if 45 <= energy_mean <= 60:
        energy_base = 80
    elif 40 <= energy_mean < 45 or 60 < energy_mean <= 66:
        energy_base = 65
    elif 35 <= energy_mean < 40 or 66 < energy_mean <= 72:
        energy_base = 48
    else:
        energy_base = 30

    # Energy variation helps detect monotone delivery
    if 10 <= energy_variation <= 22:
        energy_variation_bonus = 5
    elif 7 <= energy_variation < 10:
        energy_variation_bonus = 0
    elif 4 <= energy_variation < 7:
        energy_variation_bonus = -12
    else:
        energy_variation_bonus = -20

    energy_control_score = clamp(energy_base + energy_variation_bonus, 20, 90)

    # 6. Silence control score
    # Too much silence = hesitant; too little = rushed.
    if 0.10 <= silence_ratio <= 0.24:
        silence_control_score = 85
    elif 0.24 < silence_ratio <= 0.30 or 0.07 <= silence_ratio < 0.10:
        silence_control_score = 65
    elif 0.30 < silence_ratio <= 0.36 or 0.05 <= silence_ratio < 0.07:
        silence_control_score = 45
    else:
        silence_control_score = 25

    # =========================
    # 🔗 DELIVERY SCORE
    # =========================
    delivery_score = (
        pace_score +
        pause_control_score +
        rhythm_score +
        vocal_control_score +
        energy_control_score +
        silence_control_score
    ) / 6

    # =========================
    # 🔗 BASE SCORE
    # =========================
    # Communication authority depends slightly more on delivery than transcript.
    base_score = (0.42 * content_score) + (0.58 * delivery_score)

    # =========================
    # 🔥 PENALTIES
    # =========================
    penalty = 0

    # Filler penalty
    if filler_density > 0.10:
        penalty += 20
    elif filler_density > 0.05:
        penalty += 12
    elif filler_density > 0.02:
        penalty += 6

    # Weak transcript penalties
    if idea_strength < 45:
        penalty += 10
    elif idea_strength < 55:
        penalty += 5

    if coherence < 45:
        penalty += 8
    elif coherence < 55:
        penalty += 4

    if conciseness < 45:
        penalty += 6

    # Delivery-specific penalties
    if wpm < 90:
        penalty += 10
    elif wpm < 100:
        penalty += 6
    elif wpm > 185:
        penalty += 10
    elif wpm > 170:
        penalty += 5

    if pause_frequency > 0.50:
        penalty += 8
    elif pause_frequency > 0.42:
        penalty += 4

    if avg_pause > 1.1:
        penalty += 8
    elif avg_pause > 0.9:
        penalty += 4

    if silence_ratio > 0.30:
        penalty += 8
    elif silence_ratio < 0.07:
        penalty += 6

    if pitch_variation < 22:
        penalty += 8
    elif pitch_variation > 80:
        penalty += 8

    if energy_variation < 6:
        penalty += 10
    elif energy_variation < 8:
        penalty += 5

    # Combined hesitation / instability penalties
    if pause_frequency > 0.45 and silence_ratio > 0.24:
        penalty += 8

    if pitch_variation > 72 and silence_ratio > 0.24:
        penalty += 8

    # =========================
    # 🚨 FAILURE OVERRIDE
    # =========================
    if failure:
        return round(min(base_score, 45), 2)

    # =========================
    # 🎯 FINAL SCORE PRE-CAP
    # =========================
    final_score = base_score - penalty

    # =========================
    # 🚨 DELIVERY CAPS
    # =========================
    # This is the key rule: bad delivery cannot be rescued too much by decent words.
    if delivery_score < 35:
        final_score = min(final_score, 42)
    elif delivery_score < 45:
        final_score = min(final_score, 50)
    elif delivery_score < 55:
        final_score = min(final_score, 58)
    elif delivery_score < 62:
        final_score = min(final_score, 64)

    # Additional cap for obvious hesitation patterns
    delivery_red_flags = 0

    if wpm < 95 or wpm > 185:
        delivery_red_flags += 1
    if pause_frequency > 0.50:
        delivery_red_flags += 1
    if silence_ratio > 0.30:
        delivery_red_flags += 1
    if avg_pause > 1.1:
        delivery_red_flags += 1
    if pitch_variation < 22 or pitch_variation > 80:
        delivery_red_flags += 1
    if energy_variation < 6:
        delivery_red_flags += 1

    if delivery_red_flags >= 4:
        final_score = min(final_score, 46)
    elif delivery_red_flags >= 3:
        final_score = min(final_score, 52)
    elif delivery_red_flags >= 2:
        final_score = min(final_score, 60)

    # =========================
    # ✅ CLAMP
    # =========================
    final_score = clamp(final_score, 25, 95)

    return round(final_score, 2)