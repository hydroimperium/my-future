import math
from schemas import UserHabitsInput, COUNTRY_PROFILES, RU_REGION_PROFILES


def get_location_baseline(country_code: str, ru_region_code: str, gender: str, age: int) -> dict:
    """
    Возвращает среднестатистическую ожидаемую продолжительность жизни (ОПЖ)
    для конкретного пола и региона (по таблицам Росстата и ВОЗ).
    """
    country = COUNTRY_PROFILES.get(country_code, COUNTRY_PROFILES["RU"])
    gender_label = "мужчин" if gender == "male" else "женщин"

    if country_code == "RU":
        reg = RU_REGION_PROFILES.get(ru_region_code, RU_REGION_PROFILES["RU_AVG"])
        gender_stat_avg = reg["male_stat_avg"] if gender == "male" else reg["female_stat_avg"]
        location_display = f"Россия ({reg['name']})" if ru_region_code != "RU_AVG" else "Россия (в среднем по РФ)"
        official_total_avg = reg["rosstat_avg"]
        score2_mult = reg["score2_mult"]
        env_note = (
            f"{reg['env_note']} Средняя ОПЖ {gender_label} в регионе по статистике Росстата: "
            f"{gender_stat_avg} лет (общая по населению: {official_total_avg} г.)."
        )
    else:
        gender_stat_avg = country["male_stat_avg"] if gender == "male" else country["female_stat_avg"]
        location_display = country["name"]
        official_total_avg = country["who_official_avg"]
        score2_mult = country["score2_mult"]
        env_note = (
            f"{country['env_note']} Средняя ОПЖ {gender_label} по данным ВОЗ: "
            f"{gender_stat_avg} лет (общая: {official_total_avg} г.)."
        )

    if age > 50:
        survival_bonus = (age - 50) * 0.25
        base_expectancy = round(max(gender_stat_avg + survival_bonus, age + 7.5), 1)
    else:
        base_expectancy = round(gender_stat_avg, 1)

    reference_base = 68.2 if gender == "male" else 78.6
    regional_delta_years = round(gender_stat_avg - reference_base, 1)

    return {
        "location_display": location_display,
        "official_stat_avg": gender_stat_avg,
        "official_total_avg": official_total_avg,
        "gender_label": gender_label,
        "base_expectancy": base_expectancy,
        "regional_delta_years": regional_delta_years,
        "score2_mult": score2_mult,
        "env_note": env_note
    }


def compute_pragmatic_lifespan(
    base_expectancy: float,
    current_age: int,
    raw_delta_years: float,
    has_severe_pathology: bool = False
) -> tuple[float, float, float]:
    """
    Переводит суммарный баланс факторов (raw_delta_years) в:
    1) эффективный сдвиг лет (effective_delta),
    2) ожидаемую продолжительность жизни (predicted_lifespan),
    3) остаток лет жизни (remaining_years).
    """
    baseline_remaining = max(5.0, base_expectancy - current_age)

    if raw_delta_years >= 0:
        max_positive_gain = 3.5 if has_severe_pathology else 8.8
        scale_div = 4.5 if has_severe_pathology else 7.8
        effective_delta = max_positive_gain * math.tanh(raw_delta_years / scale_div)
        remaining_years = baseline_remaining + effective_delta
    else:
        loss = abs(raw_delta_years)
        if loss <= 18.0:
            effective_loss = loss
        else:
            effective_loss = 18.0 + (loss - 18.0) * 0.68

        raw_remaining = baseline_remaining - effective_loss

        if raw_remaining > 6.0:
            remaining_years = raw_remaining
        else:
            deficit_below_six = 6.0 - raw_remaining
            remaining_years = 1.4 + 4.6 * math.exp(-deficit_below_six / 9.5)

        effective_delta = remaining_years - baseline_remaining

    remaining_years = round(max(1.2, remaining_years), 1)
    predicted_lifespan = round(current_age + remaining_years, 1)
    effective_delta = round(predicted_lifespan - base_expectancy, 1)

    return effective_delta, predicted_lifespan, remaining_years


def estimate_10yr_cvd_risk_score2(
    age: int,
    gender: str,
    bmi: float,
    cigs_per_day: int,
    has_hidden_nicotine: bool,
    alcohol_per_week: int,
    has_hidden_alcohol: bool,
    lifestyle_activity: str,
    has_hypertension: bool,
    meds_antihypertensive: bool,
    has_atherosclerosis: bool,
    meds_statins: bool,
    has_diabetes: bool,
    meds_antidiabetic: bool,
    has_established_cvd: bool,
    regional_multiplier: float = 1.30
) -> float:
    """
    Оценка 10-летнего риска сосудистых осложнений (инфаркт, инсульт) в %
    с учётом скрытого никотина (вейп/айкос/кальян), диагнозов из текста и анализов.
    """
    base_risk = 1.9 if gender == "male" else 0.85
    age_multiplier = math.exp(0.066 * (age - 40))

    if has_hypertension:
        bp_mult = 1.25 if meds_antihypertensive else 1.65
    else:
        bp_mult = 1.0 + max(0.0, (bmi - 25.0) * 0.035)

    if cigs_per_day > 0:
        if cigs_per_day < 10:
            smoke_mult = 1.55
        elif cigs_per_day <= 20:
            smoke_mult = 2.05
        else:
            smoke_mult = 2.55
    elif has_hidden_nicotine:
        # Вейп, электронные сигареты, айкос или кальян из текстового поля
        smoke_mult = 1.45
    else:
        smoke_mult = 0.88

    if alcohol_per_week == 0 and not has_hidden_alcohol:
        alc_mult = 0.94
    elif has_hidden_alcohol and alcohol_per_week == 0:
        alc_mult = 1.15
    elif alcohol_per_week <= 4:
        alc_mult = 1.0
    else:
        alc_mult = 1.0 + (alcohol_per_week - 4) * 0.018

    activity_mult_map = {
        "sedentary": 1.22,
        "light": 1.05,
        "moderate": 0.88,
        "active_gym": 0.76
    }
    act_mult = activity_mult_map.get(lifestyle_activity, 1.0)

    if has_atherosclerosis:
        lipid_mult = 1.22 if meds_statins else 1.60
    else:
        lipid_mult = 1.0 + max(0.0, (bmi - 24.0) * 0.025)

    diab_mult = (1.45 if meds_antidiabetic else 2.10) if has_diabetes else 1.0
    established_cvd_mult = 2.35 if has_established_cvd else 1.0

    risk = (
        base_risk
        * age_multiplier
        * bp_mult
        * smoke_mult
        * alc_mult
        * act_mult
        * lipid_mult
        * diab_mult
        * established_cvd_mult
        * regional_multiplier
    )
    return round(max(0.4, min(78.0, risk)), 1)


def calculate_heart_health_index(
    data: UserHabitsInput,
    bmi: float,
    effective_flags: dict,
    lab_impact_years: float = 0.0,
    custom_impact_years: float = 0.0
) -> int:
    """
    Комплексный индекс ресурса сердца и сосудов от 0 до 100 баллов.
    Учитывает перекрёстные флаги из текстового поля (вейп, инфаркт в тексте и др.) и прикреплённые анализы.
    """
    # 1. Питание
    if data.fastfood_times_per_week == 0 and not effective_flags.get("has_hidden_poor_diet"):
        diet_score = 96
    elif data.fastfood_times_per_week <= 2:
        diet_score = 75 if not effective_flags.get("has_hidden_poor_diet") else 50
    elif data.fastfood_times_per_week <= 5:
        diet_score = 48
    elif data.fastfood_times_per_week <= 10:
        diet_score = 22
    else:
        diet_score = 5

    # 2. Образ жизни, спортзал и шаги
    eff_activity = "sedentary" if effective_flags.get("has_activity_limitation") else data.lifestyle_activity
    gym_bonus_map = {"sedentary": -15, "light": 0, "moderate": 14, "active_gym": 24}
    if data.daily_steps >= 10000 and not effective_flags.get("has_activity_limitation"):
        pa_base = 90
    elif data.daily_steps >= 7500:
        pa_base = 78
    elif data.daily_steps >= 5000:
        pa_base = 58
    elif data.daily_steps >= 3000:
        pa_base = 38
    else:
        pa_base = 20
    pa_score = max(5, min(100, pa_base + gym_bonus_map.get(eff_activity, 0)))

    # 3. Никотин (с учётом вейпа/айкоса/кальяна в текстовом поле) и алкоголь
    if data.smoking_cigs_per_day == 0 and not effective_flags.get("has_hidden_nicotine"):
        nicotine_score = 100 if data.past_smoking_years == 0 else 82
    elif effective_flags.get("has_hidden_nicotine") and data.smoking_cigs_per_day == 0:
        nicotine_score = 35  # Вейп / электронные сигареты / кальян снижают сосудистый балл
    elif data.smoking_cigs_per_day < 10:
        nicotine_score = 28
    elif data.smoking_cigs_per_day <= 20:
        nicotine_score = 10
    else:
        nicotine_score = 0

    if data.alcohol_drinks_per_week == 0 and not effective_flags.get("has_hidden_alcohol"):
        nicotine_score = min(100, nicotine_score + 5)
    elif data.alcohol_drinks_per_week > 10 or effective_flags.get("has_hidden_alcohol"):
        nicotine_score = max(0, nicotine_score - 22)

    # 4. Сон и стресс
    if 7.0 <= data.sleep_hours <= 8.5:
        sleep_score = 96
    elif 6.0 <= data.sleep_hours < 7.0 or 8.5 < data.sleep_hours <= 9.5:
        sleep_score = 65
    else:
        sleep_score = 25
    if data.stress_level == "high":
        sleep_score = max(10, sleep_score - 20)

    # 5. ИМТ
    if 18.5 <= bmi <= 24.8:
        bmi_score = 100
    elif 24.8 < bmi <= 28.0:
        bmi_score = 74
    elif 28.0 < bmi <= 32.0:
        bmi_score = 48
    elif 32.0 < bmi <= 37.0:
        bmi_score = 25
    else:
        bmi_score = 5

    # 6. Сосудистый статус, диагнозы (из анкеты И из текста/анализов)
    vascular_score = 96
    if effective_flags.get("effective_has_hypertension"):
        vascular_score -= 15 if data.meds_antihypertensive else 32
    if effective_flags.get("effective_has_diabetes"):
        vascular_score -= 18 if data.meds_antidiabetic else 35
    if effective_flags.get("effective_has_atherosclerosis"):
        vascular_score -= 12 if data.meds_statins_lipid else 25
    if effective_flags.get("effective_has_cvd") or effective_flags.get("effective_has_stroke"):
        vascular_score -= 32
    if effective_flags.get("effective_has_thrombosis") or effective_flags.get("effective_has_oncology"):
        vascular_score -= 22
    vascular_score = max(0, vascular_score)

    raw_avg = (diet_score + pa_score + nicotine_score + sleep_score + bmi_score + vascular_score) / 6.0

    # Дополнительная коррекция по тяжести лабораторных отклонений и кастомных патологий
    if lab_impact_years < 0:
        raw_avg -= abs(lab_impact_years) * 2.2
    if custom_impact_years < -1.5:
        raw_avg -= (abs(custom_impact_years) - 1.0) * 1.5

    return int(round(max(4, min(99, raw_avg))))


def calculate_biological_age(
    chronological_age: int,
    years_delta: float,
    lab_impact_years: float,
    custom_impact_years: float,
    aha_le8_score: int
) -> float:
    """
    Рассчитывает фенотипический возраст («возраст по состоянию организма» по принципу PhenoAge / Klemera-Doubal).
    Явно реагирует на изменения в анализах крови, дополнительные заболевания из текста и общий сосудистый ресурс.
    """
    # 1. Основной вклад от сдвига ожидаемой продолжительности жизни (включает привычки, анализы и доп. сведения)
    lifespan_component = -years_delta * 0.65

    # 2. Прямой биомаркерный и клинический акцент (чтобы плохие или идеальные анализы всегда заметно сдвигали биовозраст)
    biomarker_accent = -(lab_impact_years * 0.45) - (custom_impact_years * 0.30)

    # 3. Вклад кардиометаболического индекса (отклонение от нейтральных 70 баллов)
    cardio_component = (70.0 - aha_le8_score) * 0.06

    total_shift = lifespan_component + biomarker_accent + cardio_component
    bio_age = round(chronological_age + total_shift, 1)

    # Реалистичные физиологические границы: от -12 лет (очень тренированный организм) до +25 лет (тяжелый износ)
    min_bio = max(16.0, float(chronological_age - 12.0))
    max_bio = float(chronological_age + 25.0)
    return round(max(min_bio, min(max_bio, bio_age)), 1)
