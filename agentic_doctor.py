def agentic_doctor_response(disease, confidence, info):
    # severity score: 1–10
    severity_score = max(1, min(10, int(round(confidence * 10))))

    if severity_score >= 8:
        severity_level = "HIGH"
    elif severity_score >= 5:
        severity_level = "MODERATE"
    else:
        severity_level = "LOW"

    home_care_map = {
        "acne": "Wash face gently, avoid squeezing pimples, use mild skincare products.",
        "eczema": "Moisturize frequently, avoid harsh soaps and allergens.",
        "psoriasis": "Use moisturizers, avoid triggers, get moderate sunlight.",
        "ringworm": "Keep area dry, avoid sharing towels, apply antifungal cream.",
        "scabies": "Wash clothes and bedding in hot water, treat close contacts."
    }

    return {
        "severity_score": severity_score,
        "severity_level": severity_level,
        "explanation": f"Image features are consistent with {disease}. This is not a confirmed diagnosis.",
        "home_care": home_care_map.get(disease, "Maintain hygiene and monitor symptoms."),
        "warning": info["consult_doctor_when"]
    }
