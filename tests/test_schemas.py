"""Schema validation tests for authority.v2."""

from schemas import AuthorityV2Response


def _minimal_valid_payload() -> dict:
    return {
        "schema_version": "authority.v2",
        "analysis_id": "00000000-0000-0000-0000-000000000001",
        "created_at": "2026-06-30T12:00:00Z",
        "request": {
            "scenario": "benchmark",
            "prompt_id": "authority_benchmark_v1",
            "language": "en",
            "duration_ms": 54230,
            "device_context": "ios_app",
            "user_id": None,
        },
        "audio_quality": {
            "usable": True,
            "snr_estimate_db": 21.4,
            "clipping_detected": False,
            "background_noise_level": "low",
            "single_speaker_likelihood": 0.98,
            "quality_warnings": [],
        },
        "transcript": {
            "full_text": "I believe we should move forward with the plan.",
            "speaker_language_confidence": 0.93,
            "asr_model": "whisper-1",
            "overall_asr_confidence": 0.88,
            "words": [
                {
                    "text": "I",
                    "start_ms": 1200,
                    "end_ms": 1380,
                    "confidence": 0.99,
                    "is_filler": False,
                }
            ],
            "segments": [
                {
                    "segment_id": "seg_1",
                    "start_ms": 1200,
                    "end_ms": 6400,
                    "text": "I believe we should move forward with the plan.",
                    "role": "opening",
                }
            ],
        },
        "scores": {
            "authority_score": 72,
            "authority_percentile_estimate": 0.61,
            "score_confidence": 0.79,
            "dimension_scores": {
                "command": 74,
                "clarity": 70,
                "composure": 68,
                "presence": 76,
                "persuasion": 67,
                "structure": 71,
            },
            "derived_axes": {
                "trust_warmth": 58,
                "dominance_status": 73,
                "nervousness": 42,
                "interview_readiness": 70,
                "leadership_readiness": 69,
            },
            "score_components": {
                "weighted_base": 71.4,
                "bonuses": {
                    "opening_strength": 1.5,
                    "ending_strength": 0.0,
                    "consistency_bonus": 0.8,
                },
                "penalties": {
                    "filler_penalty": 2.0,
                    "rambling_penalty": 0.0,
                    "monotony_penalty": 1.5,
                    "rising_ending_penalty": 0.8,
                    "audio_quality_penalty": 0.0,
                },
            },
        },
        "metrics": {
            "raw_acoustic": {
                "words_per_minute": 146,
                "syllables_per_second": 4.7,
                "pause_frequency_per_min": 8.4,
                "avg_pause_ms": 420,
                "longest_pause_ms": 1410,
                "mid_phrase_pause_rate": None,
                "f0_median_hz": 132.1,
                "f0_range_semitones": 7.8,
                "f0_variability_semitones": 2.1,
                "terminal_rise_ratio": None,
                "loudness_mean_db_relative": -20.1,
                "loudness_variation_db": 5.2,
                "hnr": None,
                "jitter_local": None,
                "shimmer_local": None,
            },
            "linguistic": {
                "filler_words_per_min": 3.1,
                "hedges_per_100_words": 1.8,
                "certainty_markers_per_100_words": 2.6,
                "passive_voice_ratio": None,
                "apology_markers": 0,
                "self_doubt_markers": 1,
                "repetition_rate": 0.04,
                "specificity_score": 0.63,
                "concreteness_score": 0.58,
                "rambling_score": None,
                "opening_strength_score": 0.74,
                "closing_strength_score": 0.48,
                "structure_score": 0.71,
            },
            "derived": {
                "monotony_index": 0.29,
                "hesitation_cluster_score": 0.33,
                "dynamic_emphasis_score": 0.66,
                "speech_continuity_score": 0.72,
                "confidence_drop_count": 1,
            },
        },
        "perception_profile": {
            "headline": "You sound capable and credible.",
            "how_you_currently_come_across": "Listeners likely see you as capable.",
            "biggest_strength": {
                "title": "Clear structure",
                "explanation": "You establish your point early.",
            },
            "biggest_drag": {
                "title": "Partial approval-seeking",
                "explanation": "Some endings sound less final.",
            },
            "listener_assumptions": ["You know what you want to say"],
            "reads": {
                "emotional": "mostly calm",
                "professional": "competent",
                "social_status": "respectable",
                "interview": "hireable",
                "leadership": "trusted for contribution",
            },
        },
        "evidence": [
            {
                "id": "ev_1",
                "trait": "command",
                "direction": "positive",
                "headline": "You owned the opening",
                "why_it_matters": "Early decisiveness anchors first impressions.",
                "signals": ["clear thesis"],
            }
        ],
        "moments": [
            {
                "moment_id": "m1",
                "type": "strongest_moment",
                "start_ms": 6200,
                "end_ms": 9800,
                "severity": "highlight",
                "headline": "Most commanding section",
                "summary": "Pace and ending aligned here.",
                "dimension_impact": {"command": 0.18},
                "preview_visible_free": True,
            }
        ],
        "recommendations": {
            "highest_leverage_issue": "declarative finality",
            "fastest_improvement_tip": "Finish key lines with cleaner endings.",
            "coaching_summary": "Tighten delivery finality for the fastest lift.",
        },
        "drills": [
            {
                "drill_id": "drop_the_landing_v1",
                "title": "Drop the landing",
                "goal": "reduce rising declarative endings",
                "instructions": ["Read 8 short statements"],
                "duration_min": 4,
                "difficulty": "beginner",
                "target_metrics": ["terminal_rise_ratio", "command"],
            }
        ],
        "progress": {
            "comparison_available": False,
            "baseline_analysis_id": None,
            "delta_authority_score": None,
            "dimension_deltas": None,
        },
        "paywall": {
            "free_preview": {
                "show_score": True,
                "show_headline": True,
                "show_strength": True,
                "show_drag": True,
                "show_fast_tip": True,
                "show_single_visible_moment": True,
            },
            "locked_modules": ["full_transcript"],
        },
        "uncertainty": {
            "overall_confidence_label": "medium_high",
            "suppressed_traits": [],
            "reasons": [],
        },
        "safety": {
            "responsible_framing": "These results describe likely listener impressions.",
            "limitations": ["Short single-sample recording"],
        },
    }


def test_authority_v2_schema_accepts_spec_example():
    model = AuthorityV2Response.model_validate(_minimal_valid_payload())
    assert model.schema_version == "authority.v2"
    assert model.scores.authority_score == 72
    assert len(model.scores.dimension_scores.model_dump()) == 6
    assert model.psychological_inference.micro_behaviours == []
    assert model.report.mirror is None


def test_authority_v2_round_trip():
    payload = _minimal_valid_payload()
    model = AuthorityV2Response.model_validate(payload)
    dumped = model.model_dump()
    assert dumped["schema_version"] == "authority.v2"
    AuthorityV2Response.model_validate(dumped)
