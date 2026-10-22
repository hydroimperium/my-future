"""
portrait_service.py — Модуль «Портрет из будущего (+20 лет)»:
генерация промпта Full-Body визуального старения по прогнозу здоровья
и вызов Image Generation API для создания фотореалистичного портрета.

Промпт состоит из 2 частей:
1) Фиксированная — строгое сохранение узнаваемости, костного скелета, кадра
   и ОБЯЗАТЕЛЬНАЯ трансформация всего видимого тела (комплекция, мышечно-жировой
   баланс, посадка одежды, кожа рук и шеи).
2) Динамическая — генерируется LLM (с детерминированным fallback) в двух сценариях:
   - "current": текущие привычки, износ организма и прогрессирующие риски;
   - "potential": переход на здоровый образ жизни, спорт, нормализация веса.

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
FIXED_IDENTITY_PROMPT = f"""Transform this photograph to realistically show the EXACT SAME person aged by {YEARS_FORWARD} years into their future. This is an advanced forensic, health-modulated aging visualization.

BLOCK A — STRICT IDENTITY & SKELETAL LOCK (DO NOT ALTER):
1. ABSOLUTE FACIAL RECOGNIZABILITY: Exactly preserve the person's unique cranial and facial bone architecture, nose bridge width and tip shape, eye socket shape and aperture, eye color, ear anatomy, lip geometry and thickness, chin and jaw bone structure, and forehead proportions.
2. INDIVIDUAL IDENTIFIERS: Keep all unique identifying features completely intact: moles, birthmarks, freckles, scars, dimples, facial asymmetries, and visible tattoos.
3. SCENE & COMPOSITION LOCK: Maintain the exact same camera angle, framing, composition, perspective, focal length, distance from camera, and skeletal pose (position of limbs, torso angle). Preserve original background environment and lighting direction/quality.
4. CORE BIOLOGY: Do NOT alter gender, ethnicity, skeletal height, or limb bone lengths.
5. PHOTOREALISM: The output MUST be 100% photorealistic — indistinguishable from a genuine, unedited photograph taken {YEARS_FORWARD} years later. No airbrushed plastic skin, no illustration, no painting, and no caricature.

BLOCK B — MANDATORY FULL-BODY & CLOTHING TRANSFORMATION (MANDATORY ACROSS THE WHOLE BODY):
1. TRANSFORMATION MUST EXTEND TO THE ENTIRE VISIBLE BODY: Do NOT confine changes solely to the head! Age, metabolic changes, and physical fitness must visibly and coherently alter every visible anatomical zone from head to feet.
2. BODY COMPOSITION & SOFT TISSUES: Accurately modify the volume of subcutaneous adipose tissue and skeletal muscle mass across the neck, shoulders, chest, abdomen/waistline, back, arms, forearms, hands, hips, and legs according to the health profile below:
   - For high BMI, sedentary lifestyle, or poor nutrition: visible thickening of the neck, rounded slouching shoulders, visceral abdominal distension (belly fat), softened arms, and heavier thighs.
   - For severe chronic illness or muscle wasting (sarcopenia): thinning of limbs, loss of bicep and deltoid mass, sunken chest, prominent clavicles, and accentuated tendon lines.
   - For athletic habits or healthy potential: lean athletic frame, firm muscular definition in shoulders, arms, chest, and legs, flat trim abdomen, and confident upright posture.
3. CLOTHING FIT & DRAPE ADAPTATION: Retain the style, garment category, and color palette of the original clothing, BUT physically alter how the clothing drapes and tensions over the aged body:
   - If weight is gained: fabric of shirts, jackets, and trousers must stretch tightly across the enlarged abdomen, waist, and hips, creating realistic horizontal tension creases and pulling at buttons/seams.
   - If weight or muscle is lost: garments should hang noticeably looser, sagging slightly at the shoulders, chest, and waistband.
   - If athletic/toned: clothing conforms cleanly and neatly to a firm, athletic physique.
4. EXPOSED SKIN CONSISTENCY: The texture, elasticity, dermal thickness, age spots (solar lentigo), crepiness, and vascularity on all visible skin areas (neck, décolletage, forearms, dorsal hands, fingers, knees, legs) MUST strictly match the biological aging of the facial skin. Hands must show age-appropriate skin thinning, visible extensor tendons, and veins corresponding to the patient's cardiovascular and somatic condition."""


# =====================================================================================
# ЧАСТЬ 2: ПРАВИЛА МАППИНГА ФАКТОРОВ ЗДОРОВЬЯ → ВИЗУАЛЬНЫХ ЭФФЕКТОВ СТАРЕНИЯ (3 УРОВНЯ)
# =====================================================================================
AGING_RULES_FOR_LLM = """You are a forensic aging specialist and biomedical illustrator. Based on the health data and scenario below, write a DETAILED visual physical description (in English, 8-12 sentences) of exactly how this person's appearance will change over {years} years.

SCENARIO MODE: {scenario_title}
{scenario_context}

YOUR DESCRIPTION MUST RIGOROUSLY INTEGRATE ALL 3 ANATOMICAL LEVELS:
- LEVEL 1 (Face, Head & Hair): skin texture, undertone, micro-wrinkles, periorbital zone (dark circles, eye bags, hooded lids), smoker's lines, nasolabial folds, jawline definition/jowls/double chin, hair thinning and silver/gray progression.
- LEVEL 2 (Neck, Hands, Arms & Exposed Legs Skin): platysma neck bands (crepiness/sagging vs firm smoothness), dorsal hands and fingers (skin thinning, visible extensor tendons, protruding veins, age spots/lentigo, finger puffiness/edema), forearm and leg skin turgor.
- LEVEL 3 (Full-Body Somatotype, Musculature, Posture & Clothing Fit): visceral waist circumference, shoulder broadness vs kyphotic slump, muscle mass retention vs sarcopenia (muscle atrophy), posture alignment, and exactly how the original clothing stretches, drapes, or sags over the transformed figure.

MAPPING GUIDELINES TO APPLY:
1. SMOKING:
   - 0: clear healthy skin around mouth, natural lip color, healthy vascular glow on hands and face.
   - >10/day: deep vertical smoker's lines around mouth, yellowish tooth tint, ashy-gray sallow skin, premature thinning of dorsal hand skin.
   - Vape/hookah: dehydrated skin, premature fine lines, dull undertone.
2. ALCOHOL:
   - 0-2/week: firm defined jawline, clear bright eyes, no facial or ankle edema.
   - >7/week: facial puffiness, dilated capillaries (rosacea/spider veins) on nose and cheeks, bloated fingers, heavy jowls.
3. BMI & 20-YEAR METABOLIC DYNAMICS:
   - BMI <18.5: gaunt hollow cheeks, sunken temples, bony neck, thin limbs, loose sagging clothing.
   - BMI 18.5-24.8: proportional well-defined jawline, balanced healthy body composition, clothing fits naturally.
   - BMI 25-29.9: fuller face, developing double chin, thickened waistline, clothing tighter across midsection.
   - BMI >=30: prominent double chin, thick neck, visceral abdominal protrusion, heavy limbs, fabric stretching with tension folds.
   - Note on Sedentary + Fast food: over 20 years, even borderline BMI progresses into visceral abdominal adiposity with weakened shoulder and arm muscles (sarcopenic body composition).
4. PHYSICAL ACTIVITY:
   - Sedentary (<4000 steps): soft sagging body contours, rounded slumped forward-head posture, visible muscle atrophy in arms and thighs.
   - Active gym (>8000 steps): well-maintained muscle tone in shoulders, arms, chest, and legs, flat trim abdomen, upright confident stance.
5. SLEEP & STRESS:
   - Poor sleep (<6h): deep dark purple under-eye hollows, heavy drooping eyelids, exhausted dull skin.
   - High stress: prematurely bleached white/gray hair, deep horizontal forehead furrows and glabellar frown lines, tense neck tendons.
   - Low stress + good sleep: smooth forehead, relaxed peaceful expression, graceful silvering of hair.
6. CHRONIC ILLNESSES:
   - Hypertension: flushed face, visible temporal vascular prominence.
   - Diabetes: acanthosis nigricans on neck folds, metabolic abdominal weight.
   - CVD / Stroke: pale grayish skin tone, fragile frail appearance, translucent thinning skin on hands.

OUTPUT REQUIREMENT: Output ONLY a single coherent descriptive paragraph (8-12 sentences) in ENGLISH. Focus strictly on visible physical anatomy, skin textures, body contours, posture, and clothing tension. Do NOT include any intro, outro, headers, bullet points, or medical disclaimers."""


def _build_deterministic_aging_description(metrics: dict, user_data: UserHabitsInput, scenario: str = "current") -> str:
    """
    Детерминированный fallback: строит полное описание старения (3 анатомических уровня)
    без LLM, строго по правилам для выбранного сценария ("current" или "potential").
    """
    effects_level1 = []  # Лицо, голова, волосы
    effects_level2 = []  # Шея, кисти рук, открытая кожа
    effects_level3 = []  # Комплекция тела, осанка, посадка одежды

    if scenario == "potential":
        # СЦЕНАРИЙ ЗДОРОВОГО ПОТЕНЦИАЛА (отказ от вредных привычек, спорт, сон, нормальный вес)
        effects_level1.append(
            "Over 20 calendar years, the person exhibits remarkably graceful, slow biological aging, appearing visually only about 12 to 14 years older"
        )
        effects_level1.append(
            "The face radiates vibrant health with a clear, smooth, radiant complexion, minimal fine laugh lines around the eyes, completely absent smoker's lines, calm rested eyes without puffiness, a firm defined jawline, and natural, elegant silvering of well-maintained hair"
        )
        effects_level2.append(
            "The neck remains firm and toned with smooth skin elasticity, and the hands display healthy, well-hydrated skin with no premature liver spots or fragile thinning"
        )
        effects_level3.append(
            "The entire body maintains an upright, confident athletic posture with excellent muscle tone across the shoulders, arms, chest, and legs, and a lean, trim waistline with normalized healthy weight (BMI ~22.5)"
        )
        effects_level3.append(
            "The original clothing fits cleanly, comfortably, and neatly across a strong, fit, energetic physique, radiating vitality and preserved reserve"
        )
        return (
            "POTENTIAL HEALTHY AGING PROFILE (FULL BODY, EXPOSED SKIN & CLOTHING FIT):\n"
            + " ".join(effects_level1 + effects_level2 + effects_level3)
        )

    # СЦЕНАРИЙ ТЕКУЩЕГО ТРЕНДА (current)
    eff_flags = metrics.get("effective_flags", {})
    cigs = user_data.smoking_cigs_per_day
    drinks = user_data.alcohol_drinks_per_week
    bmi = metrics.get("bmi", 24.0)
    lifestyle = user_data.lifestyle_activity
    steps = user_data.daily_steps
    sleep = user_data.sleep_hours
    stress = user_data.stress_level
    ff = user_data.fastfood_times_per_week
    bio_age = metrics.get("biological_age", user_data.age)
    bio_delta = round(bio_age - user_data.age, 1)

    # --- Уровень 1: Лицо, глаза, кожа, волосы ---
    if cigs > 15:
        effects_level1.append(
            "The facial skin is markedly sallow, ashy-gray, and dehydrated, framed by deep perioral crevices radiating from the lips (pronounced smoker's lines), hollowed cheeks, and noticeably yellowed teeth"
        )
    elif cigs > 0 or eff_flags.get("has_hidden_nicotine"):
        effects_level1.append(
            "Noticeable fine smoker's lines border the lips, paired with a slightly dull grayish skin undertone and premature crow's feet"
        )
    else:
        effects_level1.append(
            "The skin around the mouth remains preserved with natural lip tone, without tobacco-induced perioral creasing"
        )

    if drinks > 10 or (eff_flags.get("has_hidden_alcohol") and drinks > 5):
        effects_level1.append(
            "Heavy facial bloating and edema are prominent, accompanied by visible spider veins (rosacea) across the nose and cheeks, swollen heavy jowls, and watery tired eyes"
        )
    elif drinks > 4:
        effects_level1.append(
            "Mild facial puffiness blurs the jawline with subtle redness across the nasal bridge"
        )

    if sleep < 6:
        effects_level1.append(
            "Deep dark purple circles and heavy swollen bags weigh down the lower eyelids, giving an exhausted, hollow-eyed expression"
        )
    elif sleep >= 7.5:
        effects_level1.append("The periorbital area looks rested with alert, clear eyes")

    if stress == "high":
        effects_level1.append(
            "The hair shows extensive, premature bleaching into stark white or coarse gray, accentuated by deep horizontal forehead furrows and prominent vertical glabellar frown lines"
        )
    elif stress == "medium":
        effects_level1.append("Moderate silver-gray streaks weave through the natural hair color with moderate forehead creases")

    # --- Уровень 2: Шея, кисти рук, открытая кожа ---
    if user_data.has_hypertension or drinks > 10:
        effects_level2.append(
            "The neck shows pronounced platysma banding and skin crepiness, while the hands and fingers appear mildly swollen and puffy with engorged veins"
        )
    elif cigs > 10 or bio_delta > 5:
        effects_level2.append(
            "The skin on the neck, forearms, and dorsal hands shows severe dermal thinning with papery fragility, protruding extensor tendons, visible blue veins, and scattered brown solar lentigo age spots"
        )
    else:
        effects_level2.append(
            "Skin on the neck and hands exhibits natural, graceful age-appropriate maturation without extreme thinning or heavy pigmentation"
        )

    # --- Уровень 3: Тело, осанка, мышечно-жировой баланс, посадка одежды ---
    is_sedentary_fat = (lifestyle == "sedentary" and steps < 4500 and (ff >= 3 or bmi >= 25))
    if bmi >= 32 or (bmi >= 28 and is_sedentary_fat):
        effects_level3.append(
            "The overall physique shows marked visceral adiposity with a heavily expanded abdomen, thick neck folds, and softened heavy thighs and arms"
        )
        effects_level3.append(
            "The original clothing is visibly strained and stretched tightly across the protruding belly and waist, displaying prominent horizontal tension wrinkles and pulling seams"
        )
    elif is_sedentary_fat or (bmi >= 25 and lifestyle != "active_gym"):
        effects_level3.append(
            "Over 20 years of sedentary habits have produced a sarcopenic softer silhouette: rounded slouched shoulders with forward head posture, diminished bicep and shoulder muscle tone, and a distinct softness and belly bulge around the midsection"
        )
        effects_level3.append(
            "The original garments fit noticeably tighter around the waistline while hanging slightly limp over weakened, rounded shoulders"
        )
    elif lifestyle == "active_gym" and steps >= 8000:
        effects_level3.append(
            "The full-body frame retains impressive muscular definition across the deltoids, arms, chest, and legs, paired with an upright, disciplined posture and flat abdomen"
        )
        effects_level3.append(
            "The original clothing drapes cleanly and comfortably over a robust, fit body structure"
        )
    else:
        effects_level3.append(
            "The body contours show moderate age-related softening with a mildly rounded stance and slight loss of upper-body muscle volume"
        )

    # Общая калибровка
    if bio_delta > 6:
        effects_level1.insert(
            0,
            f"Overall, the person shows accelerated somatic deterioration, looking approximately {YEARS_FORWARD + int(bio_delta * 0.6)} years older across the entire face and body"
        )
    elif bio_delta < -4:
        effects_level1.insert(
            0,
            f"Overall, the person is remarkably well-preserved, exhibiting only about {max(11, YEARS_FORWARD - int(abs(bio_delta) * 0.4))} years of visible wear"
        )

    all_sentences = effects_level1 + effects_level2 + effects_level3
    return "CURRENT HABITS AGING PROFILE (FULL BODY, EXPOSED SKIN & CLOTHING FIT):\n" + ". ".join(all_sentences) + "."


def _generate_dynamic_prompt_via_llm(metrics: dict, user_data: UserHabitsInput, scenario: str = "current") -> str | None:
    """
    Генерирует динамическую часть промпта через LLM по правилам 3-уровневого маппинга
    для выбранного сценария ("current" или "potential").
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
    primary_model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite").strip()

    if not api_key or api_key == "ПУСТО":
        return None

    bio_age = metrics.get("biological_age", user_data.age)
    bio_delta = round(bio_age - user_data.age, 1)
    aha_score = metrics.get("aha_le8_score", 50)
    eff_flags = metrics.get("effective_flags", {})

    if scenario == "potential":
        scenario_title = "OPTIMAL HEALTH POTENTIAL (SCENARIO 2 — WHAT THE PERSON CAN ACHIEVE)"
        scenario_context = f"""
The patient has committed fully to all medical recommendations starting today:
- Completely ceased smoking/vaping (0 cigarettes, 0 nicotine).
- Alcohol strictly minimized/eliminated (0-1 drink/week).
- Normalized BMI to an optimal athletic level ~22.5 kg/m² (height: {user_data.height_cm} cm).
- Physical activity: regular gym/fitness training (active athlete, >9000 steps/day).
- Sleep: 7.5 to 8.0 hours of restorative deep sleep every night.
- Stress: low, well-managed. Wholesome nutrient-dense diet without processed fast food.
- Chronic diseases (hypertension/diabetes) are under optimal clinical control.
VISUAL TARGET: The biological aging rate dramatically slows down. Over {YEARS_FORWARD} calendar years, the person visually ages only 12 to 14 years!
They must look remarkably vibrant, fit, toned, energetic, with smooth firm skin on face, neck, and hands, an upright athletic posture, and clothing fitting cleanly and neatly over a healthy physique.
"""
    else:
        scenario_title = "CURRENT HABITS & WEAR (SCENARIO 1 — CURRENT TRAJECTORY)"
        scenario_context = f"""
The patient continues their current lifestyle trajectory over the next {YEARS_FORWARD} years:
- Age: {user_data.age}, Gender: {"male" if user_data.gender == "male" else "female"}
- Cigarettes/day: {user_data.smoking_cigs_per_day} (hidden nicotine: {eff_flags.get("has_hidden_nicotine", False)})
- Alcohol drinks/week: {user_data.alcohol_drinks_per_week} (hidden alcohol: {eff_flags.get("has_hidden_alcohol", False)})
- Current BMI: {metrics.get("bmi", 24.0)} (height: {user_data.height_cm}cm, weight: {user_data.weight_kg}kg)
- Physical activity: {user_data.lifestyle_activity}, Daily steps: {user_data.daily_steps}
- Fast food times/week: {user_data.fastfood_times_per_week}
- Sleep: {user_data.sleep_hours} h/night, Stress level: {user_data.stress_level}
- Hypertension: {user_data.has_hypertension} (taking meds: {user_data.meds_antihypertensive})
- Diabetes: {user_data.has_diabetes}, CVD: {user_data.has_cvd_infarction_angina}
- Biological age delta: {bio_delta:+.1f} years, Heart/vascular resource: {aha_score}/100.
VISUAL TARGET: Accurately project 20 years of progressive wear across face, neck, hands, full-body composition, and clothing tension.
"""

    prompt_instructions = AGING_RULES_FOR_LLM.format(
        years=YEARS_FORWARD,
        scenario_title=scenario_title,
        scenario_context=scenario_context
    )

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
                            "You are an expert forensic aging specialist and biomedical illustrator. "
                            "Write a comprehensive, vivid English visual description covering Face, Neck/Hands, "
                            "Body Somatotype, Posture, and Clothing Fit. "
                            "Output ONLY a single cohesive descriptive paragraph (8-12 sentences) — no headers, "
                            "no bullet points, no markdown code fences, no disclaimers."
                        )
                    },
                    {"role": "user", "content": prompt_instructions}
                ],
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            result = re.sub(r"```\w*\n?", "", result)
            result = result.strip("`\"'").strip()
            if len(result) > 120:
                print(f">>> PORTRAIT PROMPT ({scenario}): Описание сгенерировано через LLM ({model}) <<<")
                return result
        except Exception as e:
            print(f"[Portrait prompt LLM ({model}, {scenario})]: {e}")

    return None


def build_full_portrait_prompt(metrics: dict, user_data: UserHabitsInput, scenario: str = "current") -> str:
    """
    Создаёт полный промпт для генерации портрета из будущего (+20 лет):
    - FIXED_IDENTITY_PROMPT (сохранение геометрии лица + трансформация всего тела и одежды)
    - Динамическая часть для выбранного сценария ("current" или "potential").
    """
    scenario_clean = "potential" if scenario == "potential" else "current"
    dynamic_part = _generate_dynamic_prompt_via_llm(metrics, user_data, scenario=scenario_clean)
    if not dynamic_part:
        dynamic_part = _build_deterministic_aging_description(metrics, user_data, scenario=scenario_clean)
        print(f">>> PORTRAIT PROMPT ({scenario_clean}): Использован детерминированный fallback <<<")

    scenario_header = (
        "HEALTH SCENARIO: OPTIMAL POTENTIAL (ADOPTED ALL RECOMMENDATIONS & TRAINING)"
        if scenario_clean == "potential"
        else "HEALTH SCENARIO: CURRENT WEAR & UNHEALTHY HABITS CONTINUATION"
    )

    full_prompt = f"{FIXED_IDENTITY_PROMPT}\n\n{scenario_header}:\n{dynamic_part}"
    return full_prompt


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
        is_potential = "potential" in prompt_lower or "graceful" in prompt_lower or "athletic" in prompt_lower
        is_unhealthy = any(keyword in prompt_lower for keyword in [
            "gray", "smoker", "rosacea", "bloat", "unwell", "decline", "puffiness", "cadaverous", "wasted", "sallow"
        ])

        if is_potential:
            # Здоровый потенциал: мягкое тёплое свечение, деликатное благородное взросление
            sat_factor = 0.95
            contrast_factor = 1.05
            sharp_factor = 1.15
            overlay_color = (150, 142, 136)
            alpha_overlay = 0.04
            alpha_edges = 0.02
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


def _call_gemini_image_api(api_key: str, photo_b64: str, mime_type: str, prompt: str) -> bytes | None:
    """Вызов Google Gemini / Google AI Studio для генерации изображения по фото и промпту."""
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
