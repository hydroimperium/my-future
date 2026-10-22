"""
portrait_service.py — Модуль «Портрет из будущего (+20 лет)»:
генерация промпта Full-Body визуального старения по прогнозу здоровья
и вызов Image Generation API для создания фотореалистичного портрета.

Промпт состоит из 2 частей:
1) Фиксированная — строгое сохранение узнаваемости (Immutable Central Identity Triangle),
   костного скелета, кадра и ОБЯЗАТЕЛЬНАЯ трансформация всего видимого тела
   (Mutable Body Envelope: комплекция, фигура, посадка одежды, открытая кожа).
2) Динамическая — генерируется LLM (с детерминированным fallback) в двух сценариях:
   - "current": текущие привычки, кумулятивный износ, метаболический профиль;
   - "potential": активная рекомпозиция тела, витальность, свежесть и тонус
     (со строгим запретом негативных триггерных слов дистрофии/дряхлости).

Поддерживает универсальную работу с API:
- zveno.ai (мультимодальный Chat Completions с передачей фото и промпта)
- Google AI Studio (ключи формата AIzaSy...) — Gemini 2.0 / Imagen 3
- OpenAI / OpenRouter / Российские прокси (ключи формата sk-...) — DALL-E / Flux
- Replicate (ключи формата r8_...) — Flux / InstantID
- Интеллектуальный Pillow-fallback для демонстрации и аварийных ситуаций.
"""

import os
import re
import json
import time
import base64
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from schemas import UserHabitsInput

load_dotenv()

PORTRAITS_DIR = Path(__file__).parent / "portraits"
PORTRAITS_DIR.mkdir(exist_ok=True)

YEARS_FORWARD = 20


# =====================================================================================
# ЧАСТЬ 1: ФИКСИРОВАННЫЙ ПРОМПТ — Разделение скелета/идентичности и трансформации тела
# =====================================================================================
FIXED_IDENTITY_PROMPT = f"""CRITICAL PHOTO-TRANSFORMATION SPECIFICATION ({YEARS_FORWARD} YEARS FORWARD):

1. ABSOLUTE FACIAL IDENTITY LOCK (100% RECOGNIZABILITY):
- Preserve the exact same individual from the input photograph. Any acquaintance must recognize them within 0.1 seconds.
- IMMUTABLE CENTRAL IDENTITY TRIANGLE: Exactly preserve the eye shape, iris color, gaze direction, interpupillary distance, eyebrow arch, nose bridge and tip geometry, philtrum, lip shape and proportions, ear anatomy, and cranial bone structure. All natural moles, scars, birthmarks, and visible tattoos must remain completely intact.
- Preserve the identical camera angle, framing, composition, perspective, lighting direction, and background environment.

2. MANDATORY FULL-BODY & CLOTHING MORPHING (DO NOT STOP AT THE NECK):
- You MUST apply the transformation to the ENTIRE visible body in the frame (face, hair, neck, shoulders, chest, abdomen/waist, arms, hands, legs, and posture), NOT just the head.
- Reshape the body silhouette (fat distribution, waist circumference, shoulder width, muscle tone, posture) and exposed skin on the neck, arms, hands, and legs strictly according to the transformation profile below.
- Keep the identical clothing items, style, and colors, but physically adapt how the fabric stretches, drapes, or tensions over the transformed physique."""


# =====================================================================================
# ЧАСТЬ 2: КАЛИБРОВКА ВОЗРАСТА, АРХЕТИПА И ВЕКТОРА ТРАНСФОРМАЦИИ ФИГУРЫ
# =====================================================================================
def _compute_visual_aging_targets(metrics: dict, user_data: UserHabitsInput, scenario: str = "current") -> dict:
    """
    Вычисляет целевой визуальный возраст (visual_target_age), возрастной архетип (age_stage_descriptor),
    жёсткие возрастные рамки (age_guardrail) и точную инструкцию изменения фигуры (body_morph_instruction).
    """
    chrono_future_age = user_data.age + 20
    bio_age_now = float(metrics.get("biological_age", user_data.age))
    bio_delta = bio_age_now - float(user_data.age)
    years_delta = float(metrics.get("years_delta", 0.0))
    remaining_gain = float(metrics.get("remaining_years_gain", 0.0))
    pace_current = float(metrics.get("pace_of_aging_current", 1.0))
    pace_potential = float(metrics.get("pace_of_aging_potential", 0.85))
    pace_shift_cur = (pace_current - 1.0) * 8.0

    # Базовый расчёт визуального возраста текущего сценария с калибровкой темпа износа
    if bio_delta > 0 or years_delta < 0:
        cur_raw = chrono_future_age + max(0.0, bio_delta * 1.0) + max(0.0, -years_delta * 0.35) + max(0.0, pace_shift_cur)
    elif bio_delta <= 0 and years_delta >= 2.0:
        cur_raw = chrono_future_age - min(6.0, abs(bio_delta) * 0.8 + 2.0) + pace_shift_cur * 0.5
    else:
        cur_raw = chrono_future_age + bio_delta * 0.5 + pace_shift_cur
    current_target_age = max(user_data.age + 11, min(92, int(round(cur_raw))))

    if scenario == "current":
        visual_target_age = current_target_age
    else:
        # scenario == "potential"
        pace_bonus_pot = max(0.0, (1.0 - pace_potential) * 6.0)
        raw_pot_age = chrono_future_age - min(8.0, max(4.0, remaining_gain * 0.5 + pace_bonus_pot + 2.0))
        pot_age = int(round(raw_pot_age))
        # Сценарий potential всегда минимум на 6 лет моложе current, но не моложе age + 10
        visual_target_age = min(pot_age, current_target_age - 6)
        visual_target_age = max(user_data.age + 10, visual_target_age)

    # Определение возрастного архетипа и защитных рамок (age guardrails)
    if visual_target_age <= 39:
        age_stage_descriptor = f"a prime adult in their late 30s ({visual_target_age} years old)"
        age_guardrail = (
            f"CRITICAL AGE VALIDATION: The person is ONLY {visual_target_age} years old — a young-to-prime adult! "
            "Strictly FORBID any elderly or senile features, NO gray/white hair (unless high stress in current mode), "
            "NO deep wrinkles, NO sagging skin, NO frail posture."
        )
    elif 40 <= visual_target_age <= 49:
        age_stage_descriptor = f"a mature adult in their 40s ({visual_target_age} years old)"
        age_guardrail = (
            f"CRITICAL AGE VALIDATION: The person is {visual_target_age} years old (middle adulthood, NOT old/elderly!). "
            "Show natural 40s maturity only; strictly FORBID senile/geriatric skin, heavy jowls, or frail elderly appearance."
        )
    elif 50 <= visual_target_age <= 59:
        age_stage_descriptor = f"a middle-aged adult in their 50s ({visual_target_age} years old)"
        age_guardrail = (
            f"CRITICAL AGE VALIDATION: The person is {visual_target_age} years old (in their 50s, NOT an 80-year-old senior!). "
            "Keep age changes proportional to a 50-something adult."
        )
    elif 60 <= visual_target_age <= 69:
        age_stage_descriptor = f"a mature person in their 60s ({visual_target_age} years old)"
        age_guardrail = f"CRITICAL AGE VALIDATION: Age appearance must accurately match {visual_target_age} years old."
    else:
        age_stage_descriptor = f"an older senior person around {visual_target_age} years old"
        age_guardrail = f"CRITICAL AGE VALIDATION: Show authentic {visual_target_age}-year-old senior aging characteristics."

    # Точный расчёт трансформации фигуры всего тела
    bmi = float(metrics.get("bmi", 24.0))
    lifestyle = user_data.lifestyle_activity
    steps = user_data.daily_steps
    ff = user_data.fastfood_times_per_week
    oncology = user_data.has_oncology

    if scenario == "potential":
        if bmi < 20.0:
            body_morph_instruction = (
                "FULL-BODY RECOMPOSITION (HEALTHY MUSCLE GAIN): Transform the thin frame by adding healthy, "
                "firm athletic muscle mass to the shoulders, chest, arms, forearms, and legs. "
                "Fill out the face with healthy, hydrated subcutaneous volume — smooth full cheeks, strong defined jawline, "
                "upright broad-shouldered posture. Clothing fits more substantially across the broader shoulders and chest."
            )
        elif 20.0 <= bmi <= 25.5:
            body_morph_instruction = (
                "FULL-BODY ATHLETIC TONING: Noticeably improve body physique — broader, squared-up athletic shoulders, "
                "firmer toned chest and arms with healthy skin turgor, flat tight waist/core, upright confident posture. "
                "Face retains full healthy volume with a crisp, clean jawline and high cheekbones. "
                "Clothing drapes cleanly over an athletic, fit torso."
            )
        else:
            body_morph_instruction = (
                "FULL-BODY HEALTHY TRANSFORMATION (FAT LOSS TO FIT ATHLETIC BUILD): Visibly slim down excess visceral belly fat, "
                "waist bulk, and neck/chin fullness, replacing it with a firm, toned, strong athletic physique. "
                "Eliminate the double chin to reveal a clean, sharp, well-supported jawline while keeping healthy, "
                "vibrant cheek volume (never gaunt or hollow!). Torso and waist become noticeably trimmer and fitter, "
                "shoulders open up, and clothing fits more comfortably and neatly around the slimmer midsection."
            )
    else:
        # scenario == "current"
        if bmi >= 30.0 or (bmi >= 26.0 and lifestyle in ("sedentary", "light") and ff >= 3):
            body_morph_instruction = (
                "FULL-BODY WEIGHT PROGRESSION: Visibly increase body heaviness and visceral fat accumulation over 20 years — "
                "noticeably thicker neck, prominent double chin and heavy lower face, fuller rounded shoulders, "
                "significantly expanded midsection/belly and wider waist, heavier arms and legs. "
                "Existing clothing stretches tightly across the protruding stomach, chest, and shoulders with realistic fabric tension creases."
            )
        elif bmi < 26.0 and (lifestyle == "sedentary" or steps < 5000 or ff >= 4 or user_data.alcohol_drinks_per_week >= 8):
            body_morph_instruction = (
                "FULL-BODY SARCOPENIC SHIFT: Show 20 years of sedentary muscle loss combined with central fat gain — "
                "shoulders become narrower and slouched forward, arms and chest lose muscle tone and look softer/flatter, "
                "while a noticeable soft midsection/belly develops at the waist. Neck angles slightly forward, jawline softens."
            )
        elif bmi < 18.5 or oncology:
            body_morph_instruction = (
                "FULL-BODY MUSCLE LOSS & FRAILTY: Moderate loss of muscle volume in the shoulders and arms, "
                "slightly sunken collarbones, weakened silhouette, with existing clothing hanging slightly loose over a diminished frame."
            )
        elif lifestyle == "active_gym" and steps >= 8000 and user_data.smoking_cigs_per_day == 0:
            body_morph_instruction = (
                "FULL-BODY ATHLETIC RETENTION: Maintain a strong, sturdy athletic build with squared shoulders and upright posture, "
                "reflecting lifelong fitness alongside natural age maturation."
            )
        else:
            body_morph_instruction = (
                "FULL-BODY MODERATE AGE MATURATION: Natural age-related softening of the torso and posture, "
                "slight loss of upper body firmness, and a gently relaxed waistline, with clothing fitting naturally."
            )

    return {
        "chrono_future_age": chrono_future_age,
        "visual_target_age": visual_target_age,
        "age_stage_descriptor": age_stage_descriptor,
        "age_guardrail": age_guardrail,
        "body_morph_instruction": body_morph_instruction,
    }


# =====================================================================================
# ЧАСТЬ 3: ПОЛНЫЙ СБОР ДАННЫХ АНКЕТЫ, ДИАГНОЗОВ, АНАЛИЗОВ И СВОБОДНОГО ТЕКСТА
# =====================================================================================
def _extract_comprehensive_health_context(metrics: dict, user_data: UserHabitsInput, scenario: str, targets: dict) -> str:
    """
    Извлекает и структурирует абсолютно все данные пользователя:
    анкету, расчётные метрики, прошлый анамнез, все 10 хронических диагнозов,
    все 8 групп медикаментов, свободный текст (custom_health_nuance) и анализы (lab_summary_info).
    """
    eff_flags = metrics.get("effective_flags", {})
    lines = []

    # 1. Биометрика и возраст
    lines.append(f"- Age now: {user_data.age}, Chronological age in 20 years: {targets['chrono_future_age']}, Target visual age: {targets['visual_target_age']} ({targets['age_stage_descriptor']})")
    lines.append(f"- Gender: {'Male' if user_data.gender == 'male' else 'Female'}, Height: {user_data.height_cm} cm, Weight: {user_data.weight_kg} kg, Current BMI: {metrics.get('bmi', 24.0):.1f}")
    bio_age = metrics.get('biological_age', user_data.age)
    lines.append(f"- Biological age now: {bio_age:.1f} (delta: {bio_age - user_data.age:+.1f} years)")
    lines.append(f"- Cardiovascular Health (AHA LE8): {metrics.get('aha_le8_score', 50)}/100, 10-Year CVD Risk: {metrics.get('cvd_10yr_risk_percent', 5.0):.1f}%")
    pace_c = float(metrics.get("pace_of_aging_current", 1.0))
    pace_p = float(metrics.get("pace_of_aging_potential", 0.85))
    cvd_30_c = float(metrics.get("cvd_30yr_risk_percent", 10.0))
    cvd_30_p = float(metrics.get("cvd_30yr_risk_potential_percent", 5.0))
    ckm_lbl = metrics.get("ckm_label", "Stage 0")
    lines.append(f"- Pace of Aging (DunedinPACE): {pace_c:.2f} biological yr/cal yr (potential: {pace_p:.2f} yr/yr)")
    lines.append(f"- 30-Year CVD Risk & CKM Syndrome: {cvd_30_c:.1f}% -> {cvd_30_p:.1f}%, CKM: {ckm_lbl}")
    organ_sys = metrics.get("organ_systems", [])
    if organ_sys:
        sys_str = ", ".join([f"{s.get('title', s.get('id'))}: {s.get('current_score')}% (pot: {s.get('potential_score')}%)" for s in organ_sys])
        lines.append(f"- 4 Organ Systems Health Resource: {sys_str}")

    # 2. Привычки и образ жизни
    lines.append(f"- Smoking: {user_data.smoking_cigs_per_day} cigs/day (Hidden nicotine/vape: {eff_flags.get('has_hidden_nicotine', False)})")
    if user_data.past_smoking_years > 0 or user_data.years_since_quitting_smoking > 0:
        lines.append(f"- Past smoking history: {user_data.past_smoking_years} years ({user_data.past_smoking_cigs_per_day} cigs/day), quit {user_data.years_since_quitting_smoking} years ago")
    lines.append(f"- Alcohol: {user_data.alcohol_drinks_per_week} drinks/week (Hidden alcohol: {eff_flags.get('has_hidden_alcohol', False)}, Past heavy alcohol: {user_data.past_heavy_alcohol})")
    lines.append(f"- Physical activity: {user_data.lifestyle_activity}, Daily steps: {user_data.daily_steps} (Activity limitation: {eff_flags.get('has_activity_limitation', False)})")
    lines.append(f"- Sleep: {user_data.sleep_hours} hours/night, Stress level: {user_data.stress_level}")
    lines.append(f"- Nutrition: {user_data.fastfood_times_per_week} fastfood times/week (Hidden poor diet: {eff_flags.get('has_hidden_poor_diet', False)})")
    if user_data.hazardous_work and user_data.hazardous_work != "no":
        lines.append(f"- Hazardous workplace environment: {user_data.hazardous_work}")

    # 3. Хронические заболевания и приём лекарств
    diag_list = []
    if user_data.has_hypertension:
        diag_list.append(f"Hypertension (meds: {user_data.meds_antihypertensive})")
    if user_data.has_diabetes:
        diag_list.append(f"Type 2 Diabetes (meds: {user_data.meds_antidiabetic})")
    if user_data.has_atherosclerosis:
        diag_list.append(f"Atherosclerosis (statins: {user_data.meds_statins_lipid})")
    if user_data.has_cvd_infarction_angina:
        diag_list.append("Coronary Heart Disease / Angina / Post-Infarction")
    if user_data.has_stroke_tia:
        diag_list.append("Cerebrovascular Disease / Stroke / TIA")
    if user_data.has_thrombosis_pe:
        diag_list.append(f"Thrombosis / PE (anticoagulants: {user_data.meds_anticoagulants})")
    if user_data.has_oncology:
        diag_list.append("Oncological diagnosis in history")
    if user_data.has_rheumatoid_arthritis:
        diag_list.append("Rheumatoid Arthritis / Systemic autoimmune inflammation")
    if user_data.has_ibd_polyps:
        diag_list.append("Inflammatory Bowel Disease / Colonic polyps")
    if user_data.has_urinary_infection:
        diag_list.append("Chronic Urinary Tract / Kidney Infection")

    other_meds = []
    if user_data.meds_antiarrhythmic:
        other_meds.append("antiarrhythmics")
    if user_data.meds_asthma_copd:
        other_meds.append("asthma/COPD bronchodilators")
    if user_data.meds_glucocorticoids:
        other_meds.append("systemic glucocorticoid hormones (cushingoid facial fullness risk)")
    if user_data.meds_daily_nsaids:
        other_meds.append("daily NSAIDs")

    if diag_list:
        lines.append(f"- Medical conditions: {'; '.join(diag_list)}")
    if other_meds:
        lines.append(f"- Specific ongoing medications: {', '.join(other_meds)}")

    # 4. Свободный текст пользователя и клинический анализ нюанса
    nuance_raw = (user_data.custom_health_nuance or "").strip()
    nuance_info = metrics.get("custom_nuance_info") or {}
    if nuance_raw:
        lines.append(f"- User Personal Health Nuance (free text): \"{nuance_raw}\"")
        if nuance_info.get("title") or nuance_info.get("clinical_summary"):
            lines.append(f"- Clinical Nuance Analysis: {nuance_info.get('title', '')} — {nuance_info.get('clinical_summary', '')}")

    # 5. Лабораторные анализы
    lab_info = metrics.get("lab_summary_info") or {}
    abnormalities = lab_info.get("detected_abnormalities") or []
    lab_summary = lab_info.get("clinical_summary") or ""
    if abnormalities or lab_summary:
        abn_text = "; ".join(abnormalities) if abnormalities else "None specified"
        lines.append(f"- Laboratory Blood Test Abnormalities: {abn_text}")
        if lab_summary:
            lines.append(f"- Lab Clinical Summary: {lab_summary}")

    return "\n".join(lines)


# =====================================================================================
# ЧАСТЬ 4: ПОСТ-ФИЛЬТР ДЛЯ СЦЕНАРИЯ POTENTIAL (ЗАЩИТА ОТ ДИСТРОФИИ И МОРЩИН)
# =====================================================================================
def _sanitize_potential_prompt(prompt_text: str) -> str:
    """
    Пост-фильтр для сценария potential (Здоровый потенциал).
    Удаляет и заменяет любые триггерные слова-ловушки и отрицания,
    на которые компактные графические модели (Gemini Flash Image и др.)
    реагируют рисованием дистрофии, глубоких морщин или дряхлости.
    """
    text = prompt_text

    # 1. Замена распространённых отрицательных фраз
    phrases = [
        (r"\b(?:not|never|no|without|zero|free of)\s+(?:deep\s+)?wrinkles?(?:\s+skin)?\b", "smooth, elastic dermal texture"),
        (r"\b(?:not|never|no|without|zero|free of)\s+(?:facial\s+)?crevices?\b", "smooth facial contours"),
        (r"\b(?:not|never|no|without|zero|free of)\s+(?:glabellar\s+)?furrows?\b", "calm, relaxed forehead"),
        (r"\b(?:not|never|no|without|zero|free of)\s+(?:skin\s+)?sagging(?:\s+skin)?\b", "firm, well-lifted skin"),
        (r"\b(?:not|never|no|without|zero|free of)\s+(?:drooping|jowls)(?:\s+skin)?\b", "clean, crisp jawline contours"),
        (r"\b(?:not|never|no|without|zero|free of)\s+gaunt(?:\s+cheeks)?\b", "hydrated, youthful cheek volume"),
        (r"\b(?:not|never|no|without|zero|free of)\s+hollow(?:ed)?(?:\s+cheeks)?\b", "plump, firm facial contours"),
        (r"\b(?:not|never|no|without|zero|free of)\s+sunken(?:\s+features|\s+cheeks)?\b", "supported, well-hydrated features"),
        (r"\b(?:not|never|no|without|zero|free of)\s+emaciated\b", "healthy, well-nourished volume"),
        (r"\b(?:not|never|no|without|zero|free of)\s+frail\b", "strong, energetic presence"),
        (r"\b(?:not|never|no|without|zero|free of)\s+(?:old|elderly|senile)\b", "vibrant and gracefully matured"),
        (r"\b(?:not|never|no|without|zero|free of)\s+(?:thin|skinny)\b", "athletic and well-built"),
        (r"\bweight\s+loss\b", "athletic slimming and toning"),
    ]
    for pat, rep in phrases:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # 2. Замена отдельных слов-триггеров
    word_map = [
        (r"\b(?:deep\s+)?wrinkles?\b", "fine natural skin texture"),
        (r"\bfurrows?\b", "fine mature lines"),
        (r"\bcrevices?\b", "natural skin contour"),
        (r"\bgaunt\b", "firm and toned"),
        (r"\bhollow(?:ed)?\b", "well-hydrated"),
        (r"\bsunken\b", "well-supported"),
        (r"\bemaciated\b", "athletic and lean"),
        (r"\bsagging\b", "firm, lifted"),
        (r"\bdrooping\b", "well-supported"),
        (r"\bfrail\b", "vital and resilient"),
        (r"\bsenile\b", "distinguished"),
        (r"\belderly\b", "mature adult"),
        (r"\bashy\b", "warm and healthy"),
        (r"\bsallow\b", "radiant"),
        (r"\bcadaverous\b", "vibrant"),
        (r"\bbloated\b", "cleanly defined"),
        (r"\bpuffiness\b", "clear rested contours"),
        (r"\bedema\b", "healthy vascular tone"),
    ]
    for pat, rep in word_map:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    return text


# =====================================================================================
# ЧАСТЬ 5: ДЕТЕРМИНИРОВАННЫЙ FALLBACK (СТРУКТУРИРОВАННЫЕ 5 БЛОКОВ)
# =====================================================================================
def _build_deterministic_aging_description(
    metrics: dict,
    user_data: UserHabitsInput,
    scenario: str = "current",
    targets: dict | None = None
) -> str:
    """
    Детерминированный fallback: строит структурированное 5-блочное описание старения
    без вызова LLM, строго по правилам для выбранного сценария ("current" или "potential").
    """
    if targets is None:
        targets = _compute_visual_aging_targets(metrics, user_data, scenario)

    eff_flags = metrics.get("effective_flags", {})
    cigs = user_data.smoking_cigs_per_day
    drinks = user_data.alcohol_drinks_per_week
    sleep = user_data.sleep_hours
    stress = user_data.stress_level
    nuance_raw = (user_data.custom_health_nuance or "").strip()
    lab_info = metrics.get("lab_summary_info") or {}
    abnormalities = lab_info.get("detected_abnormalities") or []

    if scenario == "potential":
        # СЦЕНАРИЙ ЗДОРОВОГО ПОТЕНЦИАЛА (СТРОГО БЕЗ НЕГАТИВНЫХ СЛОВ, АКТИВНАЯ РЕКОМПОЗИЦИЯ И ВИТАЛЬНОСТЬ)
        silvering = "with subtle, distinguished silver accents at the temples" if targets["visual_target_age"] >= 48 else "retaining rich natural hair pigmentation and fullness"

        block1 = f"1. TARGET AGE & RADIANT VITALITY: {targets['age_stage_descriptor']}. The person exhibits peak vitality, restorative energy, and graceful mature health, appearing visually at {targets['visual_target_age']} years old."
        block2 = (
            f"2. FACE, JAWLINE & HAIR: 100% facial identity, eye shape, nose structure, and lip geometry are preserved. "
            f"Facial skin is plump, deeply hydrated, and resilient with a warm healthy glow and firm dermal elasticity. "
            f"The jawline is sharply sculpted and clean-cut with healthy cheek volume. "
            f"Eyes are clear, alert, and vibrant. Hair is thick, lustrous, and well-groomed {silvering}."
        )
        block3 = f"3. ATHLETIC FULL-BODY RECOMPOSITION & POSTURE: {targets['body_morph_instruction']}"
        block4 = (
            "4. TONED ARMS, HANDS & LEGS: The arms, forearms, and shoulders show well-defined muscular tone and firm skin turgor. "
            "Hands and fingers display smooth, resilient skin with excellent vascular circulation and healthy nails. "
            "Legs are strong and athletic."
        )
        block5 = (
            "5. TAILORED CLOTHING DRAPE: The original outfit fits cleanly, comfortably, and neatly over the athletic, "
            "trimmed silhouette, draping elegantly over broad squared shoulders and a flat, firm core."
        )
        raw_text = "\n".join([block1, block2, block3, block4, block5])
        return _sanitize_potential_prompt(raw_text)

    # СЦЕНАРИЙ ТЕКУЩЕГО ТРЕНДА (current)
    block1 = f"1. TARGET AGE & OVERALL IMPRESSION: {targets['age_stage_descriptor']}. Overall somatic aging matching {targets['visual_target_age']} years old with cumulative lifestyle wear over 20 years."

    # Блок 2: Лицо, кожа, волосы
    face_details = []
    if cigs > 15:
        face_details.append("marked sallow ashy undertone, deep perioral smoker lines radiating from lips, and yellowed tooth tint")
    elif cigs > 0 or eff_flags.get("has_hidden_nicotine"):
        face_details.append("fine smoker lines bordering lips, slightly dull grayish skin undertone, and pronounced crow's feet")
    else:
        face_details.append("natural lip tone without tobacco-induced perioral creasing")

    if drinks > 10 or (eff_flags.get("has_hidden_alcohol") and drinks > 5):
        face_details.append("facial puffiness with visible spider veins (rosacea) across nose and cheeks, swollen heavy jowls, and watery tired eyes")
    elif drinks > 4:
        face_details.append("mild facial puffiness blurring the jawline with subtle redness across the nasal bridge")

    if sleep < 6:
        face_details.append("deep dark purple under-eye hollows and heavy swollen lower eyelid bags")
    elif sleep >= 7.5:
        face_details.append("rested periorbital area with alert eyes")

    if stress == "high":
        face_details.append("extensive premature gray and white hair bleaching, deep horizontal forehead furrows, and sharp glabellar frown lines")
    elif stress == "medium":
        face_details.append("moderate silver-gray streaks through hair with moderate forehead creases")

    if user_data.meds_glucocorticoids:
        face_details.append("distinct round cushingoid facial fullness and softened jawline definition from chronic steroid therapy")

    if nuance_raw:
        face_details.append(f"visible manifestation of chronic feature: {nuance_raw[:70]}")

    if abnormalities:
        face_details.append("dull skin undertone and metabolic puffiness linked to unmanaged laboratory imbalances")

    organ_systems = metrics.get("organ_systems", [])
    systems_map = {s.get("id"): s for s in organ_systems}
    ren_score = systems_map.get("hepatic_renal", {}).get("current_score", 85)
    if ren_score < 55:
        face_details.append("subtle under-eye fluid retention and sallow complexion reflecting sluggish hepatic-renal clearance")

    face_desc = ", ".join(face_details) if face_details else "natural age-appropriate deepening of facial expression lines and softened jawline definition"
    block2 = f"2. FACE, SKIN & HAIR: Preserves central eye, nose, and mouth identity. Facial changes include {face_desc}."

    # Блок 3: Шея, торс, фигура
    block3 = f"3. NECK, TORSO & BODY SILHOUETTE: {targets['body_morph_instruction']}"

    # Блок 4: Руки, кисти, ноги
    hands_details = []
    if user_data.has_hypertension or drinks > 10:
        hands_details.append("platysma neck banding, crepiness, and mild edema in fingers and ankles with prominent engorged veins")
    elif cigs > 10 or targets["visual_target_age"] >= 58:
        hands_details.append("papery dermal thinning on neck, forearms, and dorsal hands with visible extensor tendons, blue veins, and scattered brown solar lentigo age spots")
    else:
        hands_details.append("natural age-appropriate skin maturation on neck and dorsal hands with moderate skin thinning")

    if user_data.has_rheumatoid_arthritis:
        hands_details.append("mild thickening and visible stiffness in finger interphalangeal joints")

    block4 = f"4. ARMS, HANDS & LEGS: {'; '.join(hands_details)}."

    # Блок 5: Одежда
    bmi = float(metrics.get("bmi", 24.0))
    if bmi >= 28:
        block5 = "5. CLOTHING FIT: The original garments stretch tightly across the protruding belly and waist with prominent horizontal fabric tension creases and pulling at button plackets."
    elif bmi < 19 or user_data.has_oncology:
        block5 = "5. CLOTHING FIT: The original clothing hangs noticeably loose and limp over weakened, narrower shoulders and a reduced frame."
    else:
        block5 = "5. CLOTHING FIT: The original clothing fits noticeably snugger around the softening midsection while sitting slightly flatter over rounded shoulders."

    return "\n".join([block1, block2, block3, block4, block5])


# =====================================================================================
# ЧАСТЬ 6: ГЕНЕРАЦИЯ ДИНАМИЧЕСКОГО ПРОМПТА ЧЕРЕЗ LLM
# =====================================================================================
def _generate_dynamic_prompt_via_llm(
    metrics: dict,
    user_data: UserHabitsInput,
    scenario: str = "current",
    targets: dict | None = None
) -> str | None:
    """
    Генерирует компактную 5-блочную динамическую часть промпта через LLM
    с отдельными системными шаблонами для 'current' и 'potential'.
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
    primary_model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite").strip()

    if not api_key or api_key == "ПУСТО":
        return None

    if targets is None:
        targets = _compute_visual_aging_targets(metrics, user_data, scenario)

    comprehensive_context = _extract_comprehensive_health_context(metrics, user_data, scenario, targets)

    if scenario == "potential":
        system_content = (
            "You are an expert biomedical vitality specialist and aesthetic photo-restoration director. "
            "Your task is to generate a concise, affirmative photo-enhancement and age-progression instruction (130-160 words) "
            "for a multimodal image-to-image AI model. "
            "The instruction must strictly follow exactly 5 labeled blocks (numbered 1 through 5).\n\n"
            "STRICT LEXICAL BAN:\n"
            "Never use the words 'wrinkles', 'crevices', 'furrows', 'gaunt', 'hollow', 'sunken', 'thin', 'emaciated', "
            "'sagging', 'drooping', 'bloated', 'grayish', 'dull', 'elderly', 'frail', or 'disease' — not even in negations "
            "like 'no wrinkles', 'not gaunt', or 'without sagging'!\n"
            "Budget image generation models cannot process negations and will paint an emaciated frail old person.\n"
            "Use ONLY positive, affirmative anatomical descriptors: plump, hydrated, firm, athletic, defined, vibrant, radiant, toned, upright, healthy volume.\n\n"
            "Output ONLY the 5 numbered blocks — no intro, no outro, no markdown fences, no conversational filler."
        )

        user_content = f"""SCENARIO: OPTIMAL HEALTH POTENTIAL & ATHLETIC RECOMPOSITION OVER 20 YEARS

PATIENT HEALTH CONTEXT & FULL CLINICAL PROFILE:
{comprehensive_context}

{targets['age_guardrail']}

MANDATORY 5-BLOCK OUTPUT FORMAT (Use ONLY positive, affirmative words; around 130-160 words total):
1. TARGET AGE & RADIANT VITALITY: The person is {targets['visual_target_age']} years old ({targets['age_stage_descriptor']}), looking in peak physical condition, vitality, and noble energy for their age.
2. FACE, JAWLINE & HAIR: 100% preserved facial identity (eyes, nose, mouth/lips, cranial bone structure). Skin is plump, well-hydrated, smooth, and firm with a healthy warm glow. Clean, sculpted, sharp jawline. High cheekbone fullness with healthy subcutaneous volume. Thick, lustrous hair with subtle distinguished silvering at temples only if age >= 48.
3. ATHLETIC FULL-BODY RECOMPOSITION & POSTURE: Apply this transformation: {targets['body_morph_instruction']}
4. TONED ARMS, HANDS & LEGS: Firm, sculpted arms and forearms with defined muscular tone. Resilient, smooth, elastic skin on dorsal hands with healthy circulation. Strong upright athletic legs.
5. TAILORED CLOTHING DRAPE: Original garments drape cleanly, comfortably, and neatly across an athletic, fit frame, accentuating broad shoulders and a trim waist."""

    else:
        # scenario == "current"
        system_content = (
            "You are an expert forensic age-progression specialist and biomedical artist. "
            "Your task is to generate a concise, high-impact photo-morphing instruction (130-160 words) "
            "for a multimodal image-to-image AI model. "
            "The instruction must strictly follow exactly 5 labeled blocks (numbered 1 through 5).\n"
            "Focus strictly on visible physical anatomy, skin textures, body contours, posture, and clothing tension "
            "based on cumulative wear and habits.\n"
            "Output ONLY the 5 numbered blocks — no intro, no outro, no markdown fences, no conversational filler."
        )

        user_content = f"""SCENARIO: CURRENT WEAR & UNHEALTHY HABITS PROGRESSION OVER 20 YEARS

PATIENT HEALTH CONTEXT & FULL CLINICAL PROFILE:
{comprehensive_context}

{targets['age_guardrail']}

MANDATORY 5-BLOCK OUTPUT FORMAT (Keep concise, around 130-160 words total):
1. TARGET AGE & OVERALL IMPRESSION: Specify exact target age ({targets['visual_target_age']} years old) and overall aging progression matching {targets['age_stage_descriptor']}.
2. FACE, SKIN & HAIR: Concrete changes to facial skin texture, wrinkles, facial tone, periorbital hollows/bags, smoker lines, jawline/chin fullness or double chin, hair graying and density (strictly preserving 100% recognizability of eyes, nose, mouth). Integrate custom health notes and lab abnormalities.
3. NECK, TORSO & BODY SILHOUETTE: Apply this body transformation: {targets['body_morph_instruction']}
4. ARMS, HANDS & LEGS: Skin turgor, extensor tendons, veins, pigmentation, or swelling on arms, hands, fingers, and legs.
5. CLOTHING FIT: Exactly how the original clothing stretches tightly with tension creases across expanded midsection/shoulders, or hangs loose over atrophied muscles."""

    candidate_models = [primary_model]
    for backup in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        if backup not in candidate_models:
            candidate_models.append(backup)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=18.0)

    for model in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            result = re.sub(r"```\w*\n?", "", result)
            result = result.strip("`\"'").strip()
            if len(result) > 100:
                if scenario == "potential":
                    result = _sanitize_potential_prompt(result)
                print(f">>> PORTRAIT PROMPT ({scenario}): Описание сгенерировано через LLM ({model}) <<<")
                return result
        except Exception as e:
            print(f"[Portrait prompt LLM ({model}, {scenario})]: {e}")

    return None


# =====================================================================================
# ЧАСТЬ 7: СБОРКА ИТОГОВОГО ПРОМПТА
# =====================================================================================
def build_full_portrait_prompt(metrics: dict, user_data: UserHabitsInput, scenario: str = "current") -> str:
    """
    Создаёт полный промпт для генерации портрета из будущего (+20 лет):
    - FIXED_IDENTITY_PROMPT (сохранение геометрии лица + трансформация всего тела и одежды)
    - Динамическая 5-блочная часть для выбранного сценария ("current" или "potential").
    """
    scenario_clean = "potential" if scenario == "potential" else "current"
    targets = _compute_visual_aging_targets(metrics, user_data, scenario=scenario_clean)

    dynamic_part = _generate_dynamic_prompt_via_llm(metrics, user_data, scenario=scenario_clean, targets=targets)
    if not dynamic_part:
        dynamic_part = _build_deterministic_aging_description(metrics, user_data, scenario=scenario_clean, targets=targets)
        print(f">>> PORTRAIT PROMPT ({scenario_clean}): Использован детерминированный fallback <<<")

    if scenario_clean == "potential":
        dynamic_part = _sanitize_potential_prompt(dynamic_part)

    scenario_header = (
        f"HEALTH SCENARIO: OPTIMAL POTENTIAL & ATHLETIC RECOMPOSITION (TARGET AGE: ~{targets['visual_target_age']} YEARS OLD)"
        if scenario_clean == "potential"
        else f"HEALTH SCENARIO: CURRENT WEAR & UNHEALTHY HABITS PROGRESSION (TARGET AGE: ~{targets['visual_target_age']} YEARS OLD)"
    )

    full_prompt = (
        f"{FIXED_IDENTITY_PROMPT}\n\n"
        f"{scenario_header}\n"
        f"{targets['age_guardrail']}\n\n"
        f"TRANSFORMATION COMMANDS:\n"
        f"{dynamic_part}"
    )
    return full_prompt


# =====================================================================================
# ЧАСТЬ 8: PILLOW FALLBACK
# =====================================================================================
def _simulate_aged_portrait(photo_bytes: bytes, portrait_prompt: str) -> bytes | None:
    """
    Интеллектуальный fallback на случай исчерпания квоты внешнего AI API:
    обрабатывает исходное фото пользователя с помощью Pillow, реалистично моделируя
    возрастные изменения (десатурация, микроконтраст, изменение текстуры кожи)
    с учётом сценария ("potential" vs "current").
    """
    try:
        import io
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps

        orig = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        w, h = orig.size

        prompt_lower = portrait_prompt.lower()
        is_potential = "potential" in prompt_lower or "vitality" in prompt_lower or "athletic" in prompt_lower or "recomposition" in prompt_lower
        is_unhealthy = any(keyword in prompt_lower for keyword in [
            "gray", "smoker", "rosacea", "bloat", "unwell", "decline", "puffiness", "cadaverous", "wasted", "sallow", "tension creases", "sarcopenic"
        ])

        if is_potential:
            # Здоровый потенциал: мягкое тёплое свечение, деликатное благородное взросление
            sat_factor = 0.98
            contrast_factor = 1.04
            sharp_factor = 1.12
            overlay_color = (152, 145, 138)
            alpha_overlay = 0.03
            alpha_edges = 0.015
        elif is_unhealthy:
            # Выраженный износ: сероватый землистый подтон, сниженный тургор, углубление морщин
            sat_factor = 0.72
            contrast_factor = 1.16
            sharp_factor = 1.28
            overlay_color = (130, 125, 118)
            alpha_overlay = 0.08
            alpha_edges = 0.04
        else:
            # Умеренное естественное старение
            sat_factor = 0.84
            contrast_factor = 1.08
            sharp_factor = 1.20
            overlay_color = (142, 136, 130)
            alpha_overlay = 0.05
            alpha_edges = 0.03

        img = ImageEnhance.Color(orig).enhance(sat_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        img = ImageEnhance.Sharpness(img).enhance(sharp_factor)

        overlay = Image.new("RGB", (w, h), overlay_color)
        blended = Image.blend(img, overlay, alpha=alpha_overlay)

        edges = blended.filter(ImageFilter.FIND_EDGES)
        edges_gray = ImageOps.grayscale(edges)
        edges_enhance = ImageEnhance.Contrast(edges_gray).enhance(1.3)
        edges_rgb = edges_enhance.convert("RGB")
        aged_img = Image.blend(blended, edges_rgb, alpha=alpha_edges)

        out_buffer = io.BytesIO()
        aged_img.save(out_buffer, format="JPEG", quality=92)
        print(">>> PORTRAIT: Применён интеллектуальный fallback визуализации старения (Pillow) <<<")
        return out_buffer.getvalue()
    except Exception as e:
        print(f"[Portrait Fallback Error]: {e}")
        return None


# =====================================================================================
# ЧАСТЬ 9: ВЫЗОВЫ ГРАФИЧЕСКИХ МОДЕЛЕЙ (IMAGE GENERATION API)
# =====================================================================================
def _call_gemini_image_api(api_key: str, photo_b64: str, mime_type: str, prompt: str) -> bytes | None:
    """Вызов Google Gemini / Google AI Studio для генерации изображения по фото и промпту."""
    models = ["gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-2.5-flash-image"]

    i2i_prompt = (
        "IMAGE-TO-IMAGE EDITING TASK: Edit the attached photograph of this person. "
        "Keep the exact same facial identity (eyes, nose, mouth, bone structure), camera framing, and background, "
        "and apply the following full-body and facial transformation:\n\n"
        + prompt
    )

    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            body = {
                "contents": [{
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": photo_b64
                            }
                        },
                        {
                            "text": i2i_prompt
                        }
                    ]
                }],
                "generationConfig": {
                    "responseModalities": ["IMAGE", "TEXT"],
                    "temperature": 0.4
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                },
                method="POST"
            )
            print(f">>> PORTRAIT: Запрос к Google Gemini ({model})... <<<")
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for p in parts:
                    data_b64 = None
                    if "inlineData" in p:
                        data_b64 = p["inlineData"].get("data")
                    elif "inline_data" in p:
                        data_b64 = p["inline_data"].get("data")
                    if data_b64:
                        print(f">>> PORTRAIT: Фото успешно сгенерировано через Google Gemini ({model}) <<<")
                        return base64.b64decode(data_b64)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="ignore") if e.fp else ""
            print(f"[Gemini Image {model} Error {e.code}]: {err[:250]}")
        except Exception as e:
            print(f"[Gemini Image {model} Exception]: {e}")

    # Попытка Imagen 3 через :predict
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={api_key}"
        body = {
            "instances": [{"prompt": prompt}],
            "parameters": {"sampleCount": 1, "aspectRatio": "3:4"}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST"
        )
        print(">>> PORTRAIT: Пробуем Imagen 3 (:predict)... <<<")
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        predictions = data.get("predictions", [])
        if predictions and "bytesBase64Encoded" in predictions[0]:
            print(">>> PORTRAIT: Фото успешно сгенерировано через Imagen 3 <<<")
            return base64.b64decode(predictions[0]["bytesBase64Encoded"])
    except Exception as e:
        print(f"[Imagen 3 Exception]: {e}")

    return None


def _extract_image_from_response(res) -> bytes | None:
    """Извлекает байты изображения из различных форматов ответов LLM/прокси (zveno.ai, OpenAI и др.)."""
    try:
        d = res.model_dump() if hasattr(res, "model_dump") else res
        choices = d.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            images = msg.get("images") or []
            for img in images:
                url_str = ""
                if isinstance(img, dict):
                    url_str = img.get("url") or (img.get("image_url", {}) if isinstance(img.get("image_url"), dict) else {}).get("url", "")
                elif hasattr(img, "url"):
                    url_str = getattr(img, "url")

                if url_str and "base64," in url_str:
                    b64_part = url_str.split("base64,", 1)[1]
                    return base64.b64decode(b64_part)
                elif url_str and url_str.startswith("http"):
                    with urllib.request.urlopen(url_str, timeout=30.0) as u_resp:
                        return u_resp.read()

            content = msg.get("content", "")
            if isinstance(content, str) and "data:image" in content and "base64," in content:
                b64_part = content.split("base64,", 1)[1].split('"')[0].split("'")[0].split(")")[0]
                return base64.b64decode(b64_part)

            reasoning = msg.get("reasoning_details") or []
            for item in reasoning:
                if isinstance(item, dict) and "data" in item:
                    raw_data = item.get("data", "")
                    if "iVBORw0KGgo" in raw_data or "/9j/" in raw_data:
                        return base64.b64decode(raw_data)
    except Exception as e:
        print(f"[Extract Image Error]: {e}")
    return None


def _call_openai_image_api(api_key: str, base_url: str | None, prompt: str, photo_bytes: bytes, mime_type: str = "image/jpeg") -> bytes | None:
    """
    Универсальный вызов OpenAI-совместимых провайдеров (zveno.ai, ProxyAPI, OpenRouter и т.д.).
    Поддерживает как мультимодальный Chat Completions (с передачей фото пользователя),
    так и стандартный эндпоинт images.generate.
    """
    client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=90.0)
    model = os.getenv("IMAGE_GEN_MODEL", "google/gemini-3.1-flash-lite-image").strip()
    photo_b64 = base64.b64encode(photo_bytes).decode("utf-8")

    # Явная инструкция редактирования для мультимодальной модели
    i2i_prompt = (
        "IMAGE-TO-IMAGE EDITING TASK: Edit the attached photograph of this person. "
        "Keep the exact same facial identity (eyes, nose, mouth, bone structure), camera framing, and background, "
        "and apply the following full-body and facial transformation:\n\n"
        + prompt
    )

    # Метод 1: Мультимодальный Chat Completions (zveno.ai — принимает фото пользователя + промпт)
    for attempt in range(2):
        try:
            print(f">>> PORTRAIT: Запрос в {base_url or 'OpenAI'} (модель {model}, передача фото + промпт, попытка {attempt + 1})... <<<")
            res = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{photo_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": i2i_prompt
                        }
                    ]
                }]
            )
            img_bytes = _extract_image_from_response(res)
            if img_bytes:
                print(f">>> PORTRAIT: Фото успешно сгенерировано через {model} ({len(img_bytes)} байт) <<<")
                return img_bytes
        except Exception as e:
            err_msg = str(e)
            print(f"[Chat Image Gen Exception ({model}, попытка {attempt + 1})]: {err_msg}")
            if "402" in err_msg or "insufficient funds" in err_msg.lower():
                raise RuntimeError("Недостаточно средств на балансе zveno.ai (пополните баланс на сайте zveno.ai)")
            if attempt == 0:
                time.sleep(2.0)
                continue

    # Метод 2: Стандартный images.generate (для DALL-E и сервисов со специализированным /images/generations)
    try:
        dalle_model = "dall-e-3" if "dall" in model.lower() else model
        print(f">>> PORTRAIT: Пробуем images.generate ({dalle_model})... <<<")
        res = client.images.generate(
            model=dalle_model,
            prompt=prompt[:1000],
            size="1024x1024",
            response_format="b64_json"
        )
        if res.data and res.data[0].b64_json:
            print(f">>> PORTRAIT: Фото успешно сгенерировано через {dalle_model} (b64) <<<")
            return base64.b64decode(res.data[0].b64_json)
        elif res.data and res.data[0].url:
            print(f">>> PORTRAIT: Загрузка сгенерированного фото по URL... <<<")
            with urllib.request.urlopen(res.data[0].url, timeout=30.0) as u_resp:
                return u_resp.read()
    except Exception as e:
        err_msg = str(e)
        print(f"[images.generate Exception]: {err_msg}")
        if "402" in err_msg or "insufficient funds" in err_msg.lower():
            raise RuntimeError("Недостаточно средств на балансе zveno.ai (пополните баланс на сайте zveno.ai)")

    return None


def _call_replicate_image_api(api_key: str, photo_b64: str, mime_type: str, prompt: str) -> bytes | None:
    """Вызов Replicate API (Flux / SDXL img2img)."""
    try:
        i2i_prompt = (
            "IMAGE-TO-IMAGE EDITING TASK: Edit the attached photograph of this person. "
            "Keep the exact same facial identity (eyes, nose, mouth, bone structure), camera framing, and background, "
            "and apply the following full-body and facial transformation:\n\n"
            + prompt
        )
        url = "https://api.replicate.com/v1/models/black-forest-labs/flux-depth/predictions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "input": {
                "control_image": f"data:{mime_type};base64,{photo_b64}",
                "prompt": i2i_prompt[:1500]
            }
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        print(">>> PORTRAIT: Запрос к Replicate API... <<<")
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode())

        output = data.get("output")
        if isinstance(output, list) and len(output) > 0:
            img_url = output[0]
            with urllib.request.urlopen(img_url, timeout=30.0) as u_resp:
                return u_resp.read()
    except Exception as e:
        print(f"[Replicate Exception]: {e}")
    return None


def generate_portrait_image(photo_bytes: bytes, mime_type: str, portrait_prompt: str) -> bytes | None:
    """
    Универсальный диспетчер генерации портрета из будущего:
    1. Берёт ключ из IMAGE_GEN_API_KEY (или LLM_API_KEY).
    2. Автоматически определяет тип провайдера (Google AI Studio, OpenAI, OpenRouter, Replicate).
    3. Отправляет фото и промпт.
    4. Если ключ указан, но сервис вернул ошибку — пробрасывает ошибку,
       чтобы пользователь видел реальную причину (неверный адрес, нет баланса и т.д.).
    5. Если ключ не указан вовсе — включает демонстрационный режим.
    """
    api_key = os.getenv("IMAGE_GEN_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("IMAGE_GEN_BASE_URL", "").strip()
    errors = []

    if api_key and api_key != "ПУСТО":
        photo_b64 = base64.b64encode(photo_bytes).decode("utf-8")

        # 1. Если явно указан базовый URL прокси/сервиса (например, zveno.ai)
        if base_url:
            print(f">>> PORTRAIT: Запрос к кастомному сервису ({base_url})... <<<")
            img_bytes = _call_openai_image_api(api_key, base_url, portrait_prompt, photo_bytes, mime_type)
            if img_bytes:
                return img_bytes
            errors.append(f"Сервер {base_url} не вернул изображение")

        # 2. Если ключ Google AI Studio
        elif api_key.startswith("AIza"):
            img_bytes = _call_gemini_image_api(api_key, photo_b64, mime_type, portrait_prompt)
            if img_bytes:
                return img_bytes
            errors.append("Google AI Studio не сгенерировал фото (проверьте квоту/модель)")

        # 3. Если ключ Replicate
        elif api_key.startswith("r8_"):
            img_bytes = _call_replicate_image_api(api_key, photo_b64, mime_type, portrait_prompt)
            if img_bytes:
                return img_bytes
            errors.append("Replicate API не вернул фото")

        # 4. Если ключ формата sk-...
        elif api_key.startswith("sk-"):
            img_bytes = _call_openai_image_api(api_key, base_url or None, portrait_prompt, photo_bytes, mime_type)
            if img_bytes:
                return img_bytes
            errors.append("OpenAI API вернул ошибку авторизации или квоты (проверьте URL сервиса)")

        else:
            errors.append(f"Неизвестный формат ключа ({api_key[:6]}...). Укажите IMAGE_GEN_BASE_URL")

        # Если был указан ключ, но генерация не удалась — сообщаем реальную ошибку
        err_msg = "; ".join(errors) if errors else "Не удалось сгенерировать изображение через внешний API"
        raise RuntimeError(err_msg)

    # Только если ключ вообще не задан — разрешаем демо-fallback
    print(">>> PORTRAIT: API-ключ не задан, используется демонстрационный режим (Pillow) <<<")
    return _simulate_aged_portrait(photo_bytes, portrait_prompt)
