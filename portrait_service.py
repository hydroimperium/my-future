"""
portrait_service.py — Модуль «Портрет из будущего»:
генерация промпта визуального старения по прогнозу здоровья
и вызов Image Generation API для создания фотореалистичного портрета.

Промпт состоит из 2 частей:
1) Фиксированная — строгое сохранение идентичности, черт лица, кадра, экспозиции.
2) Динамическая — генерируется LLM по детальным правилам маппинга
   каждого аспекта здоровья в визуальные эффекты старения.

Поддерживает универсальную работу с API:
- Google AI Studio (ключи формата AIzaSy...) — Gemini 2.0 / Imagen 3
- OpenAI / OpenRouter / Российские прокси (ключи формата sk-...) — DALL-E / Flux / SD
- Replicate (ключи формата r8_...) — Flux / InstantID
- Интеллектуальный Pillow-fallback на случай отсутствия квоты или сетевых ошибок.
"""

import os
import re
import json
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
# ЧАСТЬ 1: ФИКСИРОВАННЫЙ ПРОМПТ — абсолютное сохранение идентичности человека
# =====================================================================================
FIXED_IDENTITY_PROMPT = f"""Transform this photograph to realistically show the EXACT SAME person aged by {YEARS_FORWARD} years into their future. This is a predictive health-based aging visualization.

CRITICAL IDENTITY PRESERVATION REQUIREMENTS (violating ANY of these is unacceptable):
1. EXACTLY preserve the person's facial bone structure, nose shape and size, eye shape and color, ear shape, lip shape and proportions, face shape, jawline geometry, forehead proportions, and chin shape. These MUST remain identical and recognizable.
2. Preserve ALL unique identifying features: moles, birthmarks, dimples, scars, facial asymmetries, tattoos if visible.
3. Maintain the EXACT same pose, body position, camera angle, framing, composition, perspective, and distance from camera.
4. Preserve the same lighting direction, quality, intensity, and overall color temperature of the scene.
5. Keep the same background and environment — it should look like the same location photographed years later.
6. Keep clothing style and colors similar (slightly aged or worn if health is poor; well-maintained if healthy).
7. The person MUST be immediately and unmistakably recognizable — any observer who knows this person must identify them without hesitation.
8. The result MUST be fully PHOTOREALISTIC — completely indistinguishable from an actual photograph taken {YEARS_FORWARD} years later. No illustrations, paintings, cartoons, or artistic stylization.
9. Do NOT change gender, ethnicity, fundamental body frame type, or core bone structure.
10. Apply natural biological aging modulated by the specific health profile described below: a person with healthy habits ages gracefully and looks vibrant; a person with destructive habits shows dramatic visible deterioration."""


# =====================================================================================
# ЧАСТЬ 2: ПРАВИЛА МАППИНГА ФАКТОРОВ ЗДОРОВЬЯ → ВИЗУАЛЬНЫХ ЭФФЕКТОВ СТАРЕНИЯ
# Передаются LLM для генерации связной динамической части промпта
# =====================================================================================
AGING_RULES_FOR_LLM = """You are a forensic aging specialist and medical illustrator. Based on the health profile below, write a DETAILED visual description (in English) of exactly how this person's appearance will change over {years} years.

STRICT MAPPING RULES — apply ALL factors that match the patient's data:

1. SMOKING (cigarettes/day):
   - 0, no vape: clean healthy skin around mouth, natural pink lip color — a POSITIVE aging sign
   - 1-10/day: subtle perioral wrinkles (smoker's lines) radiating from lips, slightly yellowed tooth tint, mildly dull grayish skin tone
   - 11-20/day: deep wrinkles around mouth and upper lip, noticeably yellowed teeth, grayish-sallow skin undertone, thinned dehydrated lips, deeper nasolabial folds, premature crow's feet
   - >20/day: very deep perioral crevices, brownish-stained teeth, ashy-gray cadaverous skin, hollowed sunken cheeks, extremely thinned cracked lips
   - Vape/e-cigarettes/hookah: skin dehydration appearance, premature fine lines and dullness, slightly grayish undertone

2. ALCOHOL (standard drinks/week):
   - 0-2: clear non-puffy face, healthy even skin tone, bright clear eyes — POSITIVE
   - 3-7: subtle facial puffiness especially around jaw, very slight redness on nose
   - 8-14: noticeable facial bloating and puffiness, visible broken capillaries (spider veins/telangiectasia) on nose and cheeks, reddish nose, watery dull eyes, coarsened rough skin texture, developing jowls
   - >14: pronounced facial edema, dense network of broken capillaries (rosacea) covering nose and cheeks, severely puffy swollen eyelids, yellowish eye whites (jaundice), heavy sagging jowls, overall bloated unhealthy waterlogged appearance

3. BODY MASS INDEX (BMI):
   - <18.5: gaunt hollow face with sharply protruding cheekbones, sunken temples, thin bony neck, emaciated frame
   - 18.5-24.8: proportional well-shaped face, defined clean jawline, balanced healthy body — POSITIVE
   - 25-29.9: moderately fuller rounder face, developing double chin, slightly thicker neck, visibly heavier body proportions
   - 30-34.9: round puffy face, prominent double chin, thick neck, significantly heavier body, skin beginning to stretch
   - ≥35: very round face with multiple chin folds, massive thick neck, extremely heavy body with prominent skin folds and stretched distorted proportions

4. PHYSICAL ACTIVITY:
   - Sedentary + <3000 steps/day: soft undefined body contours, notably slouched rounded posture, visible muscle atrophy and wasting, soft drooping jawline, general physical decline and weakness
   - Light activity: mildly soft body, slightly rounded posture, minimal muscle definition
   - Moderate + 5000-8000 steps: reasonable muscle tone, decent upright posture, moderately fit appearance
   - Active gym + >8000 steps: well-maintained visible muscle tone and definition, upright confident posture, firm defined jawline, energetic vital youthful appearance — POSITIVE

5. SLEEP (hours/night):
   - <5h: extremely deep dark purple circles under eyes, heavily drooping swollen eyelids, dull lifeless grayish skin, pronounced puffy bags under eyes, exhausted hollow-eyed appearance
   - 5-6h: deep dark circles under eyes, drooping heavy eyelids, tired dull skin, visible bags under eyes
   - 6-6.9h: noticeable dark circles, slightly fatigued appearance, mild puffiness under eyes
   - 7-8.5h: clear rested appearance around eyes, bright alert eyes, fresh healthy skin — POSITIVE
   - >9.5h: mild facial puffiness and under-eye bags, slightly sluggish appearance

6. CHRONIC STRESS LEVEL:
   - High: dramatically grayed or completely white hair (far beyond age norm), deep horizontal forehead furrows, pronounced vertical frown lines between eyebrows (glabellar lines), visibly tense clenched jaw muscles, tight neck tendons, prematurely aged appearance well beyond actual calendar years
   - Medium: moderate gray hair mixed with natural color, moderate forehead lines, slight frown lines between brows
   - Low: natural age-appropriate hair with minimal graceful silvering, relaxed calm facial expression, noticeably fewer wrinkles than expected, peaceful appearance — POSITIVE

7. DIET / FAST FOOD (times per week):
   - 0-1: clear glowing complexion, healthy radiant skin texture, even smooth skin tone — POSITIVE
   - 2-5: slightly dull skin with minor imperfections, mild uneven tone
   - >5: dull grayish-yellowish complexion, possible acne/skin blemishes, puffy bloated appearance, oily shiny skin, poor coarsened skin texture, enlarged pores

8. HYPERTENSION:
   - Without medication: persistently flushed reddish face, visibly distended throbbing veins on temples and forehead, facial edema and puffiness
   - With medication: mild facial redness, slightly controlled but visible vascular prominence

9. DIABETES:
   - Present: darkened velvety patches on neck and skin folds (acanthosis nigricans), overall appearance suggesting metabolic syndrome, possible facial fullness from medication

10. CARDIOVASCULAR DISEASE (heart attack, stroke, angina):
    - Present: pale grayish complexion suggesting poor peripheral circulation, subtle bluish-cyanotic tint around lips and nail beds, overall frail fragile appearance, thin translucent-looking skin, wasted muscle mass

11. ONCOLOGY:
    - Present: markedly gaunt hollow features, deeply sunken eyes with dark orbital shadows, papery thin fragile skin, visibly weakened and unwell appearance, possible hair thinning from treatment

12. HAZARDOUS/HEAVY WORK:
    - Moderate: slightly roughened weathered skin texture, mild premature aging from occupational exposure
    - High/dangerous: rough weathered leathery deeply wrinkled skin, premature deep-set wrinkles especially around eyes and forehead, visible signs of chronic environmental exposure damage, coarsened calloused hands

OVERALL AGING INTENSITY CALIBRATION:
- Biological age: {bio_age} years vs chronological age: {chrono_age} years (delta: {bio_delta} years)
- If biological age EXCEEDS chronological → the person must appear to have aged MORE than {years} years (approximately {adjusted_years} years of visual aging, showing accelerated deterioration)
- If biological age is BELOW chronological → the person ages LESS visually (approximately {adjusted_years_good} years), looking remarkably preserved and vibrant
- Heart & vascular health index: {aha_score}/100 — scores below 40 demand sickly, worn-out, frail appearance; scores above 75 demand vibrant, healthy, energetic appearance despite aging

OUTPUT: Write a single coherent paragraph (6-10 sentences) in ENGLISH describing ALL the specific visible aging effects to apply to the photograph. Be vivid, precise, and focus ONLY on visual physical descriptions — skin texture, wrinkle patterns, hair color/quality, body composition, posture, facial features, eye appearance, skin color/tone. Do NOT include any medical terminology, health recommendations, explanations, or reasoning."""


def _build_deterministic_aging_description(metrics: dict, user_data: UserHabitsInput) -> str:
    """
    Детерминированный fallback: строит описание визуального старения без LLM,
    применяя каждое правило маппинга к данным пользователя.
    """
    effects = []
    eff_flags = metrics.get("effective_flags", {})

    # --- 1. Курение ---
    cigs = user_data.smoking_cigs_per_day
    has_hidden_nicotine = eff_flags.get("has_hidden_nicotine", False)
    if cigs > 20:
        effects.append("Very deep perioral crevices radiating from the mouth (severe smoker's lines), brownish-stained teeth, ashy-gray cadaverous skin with a sallow undertone, hollowed sunken cheeks, extremely thinned cracked lips")
    elif cigs > 10:
        effects.append("Deep wrinkles around the mouth and upper lip, noticeably yellowish teeth, grayish-sallow skin tone, thinned dehydrated lips, deeper nasolabial folds, premature crow's feet around the eyes")
    elif cigs > 0:
        effects.append("Subtle smoker's lines around the mouth, slightly yellowed teeth, mildly dull grayish skin tone")
    elif has_hidden_nicotine:
        effects.append("Skin dehydration with premature fine lines, slightly dull grayish complexion from chronic nicotine exposure via vaping or heated tobacco products")
    else:
        effects.append("Clean healthy skin around the mouth with natural pink lip color, no smoker's lines — a sign of preserved vascular health")

    # --- 2. Алкоголь ---
    drinks = user_data.alcohol_drinks_per_week
    has_hidden_alcohol = eff_flags.get("has_hidden_alcohol", False)
    if drinks > 20 or (has_hidden_alcohol and drinks > 10):
        effects.append("Pronounced facial edema and bloating, dense network of broken capillaries (rosacea) covering nose and cheeks, severely puffy swollen eyelids, yellowish eye whites, heavy sagging jowls, overall bloated unhealthy waterlogged appearance")
    elif drinks > 10:
        effects.append("Noticeable facial bloating and puffiness, visible broken capillaries (spider veins) on nose and cheeks, reddish nose area, watery dull eyes, coarsened rough skin texture, developing jowls")
    elif drinks > 5:
        effects.append("Subtle facial puffiness especially along the jawline, slight redness in the nose and cheek area")
    elif drinks <= 2 and not has_hidden_alcohol:
        effects.append("Clear non-puffy facial features, healthy even skin tone, bright clear eyes")

    # --- 3. ИМТ ---
    bmi = metrics.get("bmi", 24.0)
    if bmi >= 40:
        effects.append("Extremely round face with multiple deep chin folds, very thick massive neck, extremely heavy body with prominent skin folds and stretched distorted proportions")
    elif bmi >= 35:
        effects.append("Very round puffy face with multiple chin folds, thick neck, extremely heavy body proportions with visible skin folds")
    elif bmi >= 30:
        effects.append("Round puffy face with prominent double chin, noticeably thick neck, significantly heavier body proportions, skin beginning to stretch")
    elif bmi >= 25:
        effects.append("Moderately fuller rounder face with developing double chin, slightly thicker neck, visibly heavier body proportions")
    elif bmi < 16.5:
        effects.append("Extremely gaunt face with sharply protruding cheekbones, deeply sunken temples, skeletal thin neck, emaciated wasted body frame")
    elif bmi < 18.5:
        effects.append("Gaunt hollow face with visible cheekbones and slightly sunken temples, thin bony neck, lean frame")
    else:
        effects.append("Proportional well-shaped face with clean defined jawline, balanced healthy body proportions")

    # --- 4. Физическая активность ---
    lifestyle = user_data.lifestyle_activity
    steps = user_data.daily_steps
    has_limitation = eff_flags.get("has_activity_limitation", False)
    if has_limitation or (lifestyle == "sedentary" and steps < 3000):
        effects.append("Soft undefined body contours, notably slouched rounded posture, visible muscle atrophy and wasting, soft drooping jawline, general physical decline")
    elif lifestyle == "sedentary":
        effects.append("Soft body with poor muscle definition, slightly hunched forward posture, mild muscle wasting")
    elif lifestyle == "active_gym" and steps >= 8000:
        effects.append("Well-maintained visible muscle tone and definition, upright confident posture, firm defined jawline, energetic vital appearance")
    elif lifestyle == "moderate" or steps >= 6000:
        effects.append("Reasonable muscle tone, decent upright posture, moderately fit appearance")
    else:
        effects.append("Mildly soft body, slightly rounded posture, minimal visible muscle definition")

    # --- 5. Сон ---
    sleep = user_data.sleep_hours
    if sleep < 5:
        effects.append("Extremely deep dark purple circles under eyes, heavily drooping swollen eyelids, dull lifeless grayish skin, pronounced puffy bags under eyes, exhausted hollow-eyed appearance")
    elif sleep < 6:
        effects.append("Deep dark circles under eyes, drooping heavy eyelids, dull tired-looking skin, visible puffy bags under eyes")
    elif sleep < 7:
        effects.append("Noticeable dark circles under eyes, slightly fatigued appearance, mild puffiness under eyes")
    elif 7 <= sleep <= 8.5:
        effects.append("Clear rested appearance around eyes, bright alert eyes, fresh healthy-looking skin")
    elif sleep > 9.5:
        effects.append("Mild facial puffiness and under-eye bags, slightly sluggish puffy appearance")
    else:
        effects.append("Slightly puffy appearance around the eyes")

    # --- 6. Стресс ---
    if user_data.stress_level == "high":
        effects.append("Dramatically grayed or completely white hair far beyond age expectations, deep horizontal forehead furrows, pronounced vertical frown lines between eyebrows, visibly tense clenched jaw muscles, tight neck tendons, prematurely aged appearance well beyond calendar years")
    elif user_data.stress_level == "medium":
        effects.append("Moderate gray hair mixed with natural color, moderate horizontal forehead lines, slight frown lines between eyebrows")
    else:
        effects.append("Natural age-appropriate hair with minimal graceful silvering, relaxed calm facial expression, noticeably fewer wrinkles than expected for age")

    # --- 7. Питание ---
    ff = user_data.fastfood_times_per_week
    has_hidden_diet = eff_flags.get("has_hidden_poor_diet", False)
    if ff > 10 or (ff > 5 and has_hidden_diet):
        effects.append("Very dull grayish-yellowish complexion with visible skin blemishes and acne, puffy bloated appearance, oily shiny skin, coarsened poor skin texture with enlarged pores")
    elif ff > 5:
        effects.append("Dull grayish complexion with possible mild blemishes, slightly bloated puffy appearance, oily-looking skin")
    elif ff > 2:
        effects.append("Slightly dull skin with minor imperfections and mild uneven tone")
    elif ff <= 1 and not has_hidden_diet:
        effects.append("Clear glowing healthy complexion, radiant smooth skin texture, even natural skin tone")

    # --- 8. Хронические заболевания ---
    if user_data.has_hypertension and not user_data.meds_antihypertensive:
        effects.append("Persistently flushed reddish face, visibly distended veins on temples and forehead, facial puffiness and edema")
    elif user_data.has_hypertension:
        effects.append("Mild facial redness suggesting controlled but present blood pressure issues, slight vascular prominence")

    if user_data.has_diabetes:
        effects.append("Darkened velvety patches on neck and skin folds (acanthosis nigricans), overall appearance suggesting metabolic changes")

    if user_data.has_cvd_infarction_angina or user_data.has_stroke_tia:
        effects.append("Pale grayish complexion suggesting poor peripheral circulation, subtle bluish-cyanotic tint around lips, overall frail fragile appearance, thin translucent-looking skin")

    if user_data.has_oncology:
        effects.append("Markedly gaunt hollow features, deeply sunken eyes with dark orbital shadows, papery thin fragile skin, visibly weakened and unwell appearance")

    if user_data.has_thrombosis_pe:
        effects.append("Slight puffiness and congestion in skin appearance, mildly swollen extremities")

    # --- 9. Вредное производство ---
    if user_data.hazardous_work == "high":
        effects.append("Rough weathered leathery deeply wrinkled skin, premature deep-set wrinkles around eyes and forehead, visible signs of chronic environmental exposure damage, coarsened calloused hands")
    elif user_data.hazardous_work == "moderate":
        effects.append("Slightly roughened weathered skin texture, mild premature aging from occupational exposure")

    # --- 10. Общая калибровка интенсивности старения ---
    bio_age = metrics.get("biological_age", user_data.age)
    bio_delta = round(bio_age - user_data.age, 1)
    aha_score = metrics.get("aha_le8_score", 50)

    if bio_delta > 10:
        adjusted = YEARS_FORWARD + int(bio_delta * 0.8)
        effects.append(f"Overall, the person appears to have aged approximately {adjusted} years rather than {YEARS_FORWARD} — showing extreme accelerated deterioration, profound wear, and a sickly worn-out appearance far beyond their actual calendar age")
    elif bio_delta > 5:
        adjusted = YEARS_FORWARD + int(bio_delta * 0.6)
        effects.append(f"Noticeably accelerated aging — the person looks roughly {adjusted} years older with significantly more visible wear and deterioration than typical for {YEARS_FORWARD} years")
    elif bio_delta > 2:
        adjusted = YEARS_FORWARD + int(bio_delta * 0.4)
        effects.append(f"Moderately accelerated aging — the person looks approximately {adjusted} years older with more wear than typical")
    elif bio_delta < -5:
        adjusted = max(10, YEARS_FORWARD - int(abs(bio_delta) * 0.5))
        effects.append(f"Remarkably well-preserved — the person appears to have aged only about {adjusted} years, looking vibrant and strikingly youthful for the time elapsed")
    elif bio_delta < -2:
        adjusted = max(13, YEARS_FORWARD - int(abs(bio_delta) * 0.3))
        effects.append(f"Graceful slow aging — the person looks approximately {adjusted} years older, retaining a youthful vitality and energy")

    if aha_score < 25:
        effects.append("The overall impression is of severe physical decline — sickly, frail, visibly unwell, and appearing far older than chronological age would suggest")
    elif aha_score < 40:
        effects.append("General appearance suggests significant chronic health issues — tired, worn-out, and notably older than expected")
    elif aha_score > 85:
        effects.append("Despite the passage of years, the person radiates health, energy, and vitality — vibrant, strong, and aging beautifully with a natural healthy glow")
    elif aha_score > 70:
        effects.append("The person has aged well overall — healthy glow, good vitality, and maintained physical presence")

    return "SPECIFIC AGING AND HEALTH EFFECTS TO APPLY TO THE PHOTOGRAPH:\n" + ".\n".join(effects) + "."


def _generate_dynamic_prompt_via_llm(metrics: dict, user_data: UserHabitsInput) -> str | None:
    """
    Генерирует динамическую часть промпта через LLM по правилам маппинга.
    LLM создаёт связное, яркое визуальное описание на основе данных прогноза.
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
    primary_model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite").strip()

    if not api_key or api_key == "ПУСТО":
        return None

    bio_age = metrics.get("biological_age", user_data.age)
    bio_delta = round(bio_age - user_data.age, 1)
    aha_score = metrics.get("aha_le8_score", 50)

    # Рассчитываем скорректированную визуальную скорость старения
    if bio_delta > 5:
        adjusted_years = YEARS_FORWARD + int(bio_delta * 0.7)
        adjusted_years_good = YEARS_FORWARD
    elif bio_delta < -3:
        adjusted_years = YEARS_FORWARD
        adjusted_years_good = max(10, YEARS_FORWARD - int(abs(bio_delta) * 0.5))
    else:
        adjusted_years = YEARS_FORWARD + max(0, int(bio_delta * 0.3))
        adjusted_years_good = max(13, YEARS_FORWARD - max(0, int(abs(bio_delta) * 0.2)))

    eff_flags = metrics.get("effective_flags", {})
    lifestyle_desc = {
        "sedentary": "sedentary (sitting >8h/day, no exercise at all)",
        "light": "light activity (basic daily movement, no workouts)",
        "moderate": "moderate (exercise/fitness 1-2 times/week)",
        "active_gym": "active athlete (gym/sports 3+ times/week)"
    }.get(user_data.lifestyle_activity, user_data.lifestyle_activity)

    health_data = f"""
PATIENT HEALTH DATA FOR VISUAL AGING PREDICTION:
- Age: {user_data.age}, Gender: {"male" if user_data.gender == "male" else "female"}
- Cigarettes/day: {user_data.smoking_cigs_per_day}{" (ALSO uses vape/e-cigarettes/hookah — hidden nicotine!)" if eff_flags.get("has_hidden_nicotine") else ""}
- Alcohol standard drinks/week: {user_data.alcohol_drinks_per_week}{" (ALSO consumes alcohol despite reporting 0 — hidden drinking!)" if eff_flags.get("has_hidden_alcohol") else ""}
- BMI: {metrics.get("bmi", 24.0)} (height {user_data.height_cm}cm, weight {user_data.weight_kg}kg)
- Physical activity: {lifestyle_desc}, daily steps: {user_data.daily_steps}
- Sleep: {user_data.sleep_hours} hours/night
- Stress level: {user_data.stress_level}
- Fast food / sweets per week: {user_data.fastfood_times_per_week}{" (ALSO has hidden poor diet despite reporting 0!)" if eff_flags.get("has_hidden_poor_diet") else ""}
- Hazardous/heavy work: {user_data.hazardous_work}
- Hypertension: {"YES" + (" (taking medication)" if user_data.meds_antihypertensive else " (NO medication — uncontrolled!)") if user_data.has_hypertension else "no"}
- Diabetes: {"YES" + (" (taking medication)" if user_data.meds_antidiabetic else " (NO medication!)") if user_data.has_diabetes else "no"}
- Heart disease (MI/angina): {"YES" if user_data.has_cvd_infarction_angina else "no"}
- Stroke/TIA: {"YES" if user_data.has_stroke_tia else "no"}
- Atherosclerosis: {"YES" if user_data.has_atherosclerosis else "no"}
- Cancer: {"YES" if user_data.has_oncology else "no"}
- Thrombosis/PE: {"YES" if user_data.has_thrombosis_pe else "no"}
- Biological age: {bio_age} years (chronological: {user_data.age}) → Δ = {bio_delta:+.1f} years
- Heart & vascular health index: {aha_score}/100
- Predicted life expectancy: {metrics.get("predicted_expectancy")} years
- 10-year cardiovascular risk: {metrics.get("cvd_10yr_risk_percent")}%
"""

    rules_formatted = AGING_RULES_FOR_LLM.format(
        years=YEARS_FORWARD,
        bio_age=bio_age,
        chrono_age=user_data.age,
        bio_delta=f"{bio_delta:+.1f}",
        adjusted_years=adjusted_years,
        adjusted_years_good=adjusted_years_good,
        aha_score=aha_score
    )

    full_prompt = f"{rules_formatted}\n\n{health_data}"

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
                    {
                        "role": "system",
                        "content": (
                            "You are an expert forensic aging specialist and medical illustrator. "
                            "Generate a concise but vivid English visual description for AI image generation. "
                            "Output ONLY the visual description paragraph — no headers, no explanations, "
                            "no markdown formatting, no bullet points. Just one cohesive paragraph."
                        )
                    },
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            result = re.sub(r"```\w*\n?", "", result)
            result = result.strip("`\"'").strip()
            if len(result) > 80:
                print(f">>> PORTRAIT PROMPT: Динамическое описание сгенерировано через LLM ({model}) <<<")
                return result
        except Exception as e:
            print(f"[Portrait prompt LLM ({model})]: {e}")

    return None


def build_full_portrait_prompt(metrics: dict, user_data: UserHabitsInput) -> str:
    """
    Создаёт полный промпт для генерации портрета из будущего:
    фиксированная часть (идентичность) + динамическая часть (здоровье → визуал).
    Динамическая часть генерируется LLM с детерминированным fallback.
    """
    dynamic_part = _generate_dynamic_prompt_via_llm(metrics, user_data)
    if not dynamic_part:
        dynamic_part = _build_deterministic_aging_description(metrics, user_data)
        print(">>> PORTRAIT PROMPT: Использован детерминированный fallback <<<")

    full_prompt = f"{FIXED_IDENTITY_PROMPT}\n\nHEALTH-BASED AGING PROFILE FOR THIS SPECIFIC PERSON:\n{dynamic_part}"
    return full_prompt


def _simulate_aged_portrait(photo_bytes: bytes, portrait_prompt: str) -> bytes | None:
    """
    Интеллектуальный fallback на случай исчерпания квоты внешнего AI API:
    обрабатывает исходное фото пользователя с помощью Pillow, реалистично моделируя
    возрастные изменения (десатурация, микроконтраст, изменение текстуры кожи)
    на основе параметров здоровья. Позволяет тестировать UI, шторку-слайдер и UX.
    """
    try:
        import io
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps

        orig = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        w, h = orig.size

        # Анализируем промпт на неблагоприятные факторы
        prompt_lower = portrait_prompt.lower()
        is_unhealthy = any(keyword in prompt_lower for keyword in [
            "gray", "smoker", "rosacea", "bloat", "unwell", "decline", "puffiness", "cadaverous", "wasted"
        ])

        # 1. С возрастом и при стрессе снижается насыщенность и меняется тонус кожи
        sat_factor = 0.72 if is_unhealthy else 0.86
        contrast_factor = 1.14 if is_unhealthy else 1.06
        sharp_factor = 1.25

        img = ImageEnhance.Color(orig).enhance(sat_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        img = ImageEnhance.Sharpness(img).enhance(sharp_factor)

        # 2. Легкое тонирование (при нездоровом образе жизни — слегка землисто-сероватый оттенок)
        overlay_color = (135, 128, 120) if is_unhealthy else (145, 138, 132)
        overlay = Image.new("RGB", (w, h), overlay_color)
        blended = Image.blend(img, overlay, alpha=0.07)

        # 3. Деликатное подчеркивание микрорельефа (линии старения)
        edges = blended.filter(ImageFilter.FIND_EDGES)
        edges_gray = ImageOps.grayscale(edges)
        edges_enhance = ImageEnhance.Contrast(edges_gray).enhance(1.3)
        edges_rgb = edges_enhance.convert("RGB")
        aged_img = Image.blend(blended, edges_rgb, alpha=0.035)

        out_buffer = io.BytesIO()
        aged_img.save(out_buffer, format="PNG", quality=92)
        print(">>> PORTRAIT: Применён интеллектуальный fallback визуализации старения (Pillow) <<<")
        return out_buffer.getvalue()
    except Exception as e:
        print(f"[Portrait Fallback Error]: {e}")
        return None


def _call_gemini_image_api(api_key: str, photo_b64: str, mime_type: str, prompt: str) -> bytes | None:
    """Вызов Google Gemini / Google AI Studio для генерации изображения по фото и промпту."""
    # Список кандидатов моделей в Google AI Studio
    models = ["gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-2.5-flash-image"]
    
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
                            "text": prompt
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
                    if "iVBORw0KGgo" in raw_data:
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

    # Метод 1: Мультимодальный Chat Completions (идеально для zveno.ai — принимает фото пользователя + промпт)
    try:
        print(f">>> PORTRAIT: Запрос в {base_url or 'OpenAI'} (модель {model}, передача фото + промпт)... <<<")
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
                        "text": prompt
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
        print(f"[Chat Image Gen Exception ({model})]: {err_msg}")
        if "402" in err_msg or "insufficient funds" in err_msg.lower():
            raise RuntimeError("Недостаточно средств на балансе zveno.ai (пополните баланс на сайте zveno.ai)")

    # Метод 2: Стандартный images.generate (для DALL-E и провайдеров со специализированным /images/generations)
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
        url = "https://api.replicate.com/v1/models/black-forest-labs/flux-depth/predictions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "input": {
                "control_image": f"data:{mime_type};base64,{photo_b64}",
                "prompt": prompt[:1500]
            }
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        print(">>> PORTRAIT: Запрос к Replicate API... <<<")
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode())
        
        # Replicate асинхронный: получаем статус или URL
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
        
        # 1. Если явно указан базовый URL прокси/сервиса
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
            # Пробуем через OpenAI SDK
            img_bytes = _call_openai_image_api(api_key, base_url or None, portrait_prompt, photo_bytes, mime_type)
            if img_bytes:
                return img_bytes
            errors.append("OpenAI API вернул ошибку авторизации или квоты (проверьте URL сервиса)")

        else:
            errors.append(f"Неизвестный формат ключа ({api_key[:6]}...). Укажите IMAGE_GEN_BASE_URL")

        # Если был указан ключ, но генерация не удалась — сообщаем реальную ошибку,
        # а НЕ маскируем её фильтром
        err_msg = "; ".join(errors) if errors else "Не удалось сгенерировать изображение через внешний API"
        raise RuntimeError(err_msg)

    # Только если ключ вообще не задан — разрешаем демо-fallback
    print(">>> PORTRAIT: API-ключ не задан, используется демонстрационный режим (Pillow) <<<")
    return _simulate_aged_portrait(photo_bytes, portrait_prompt)
