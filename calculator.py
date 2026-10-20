import re
from schemas import UserHabitsInput
from clinical_scores import (
    get_location_baseline,
    compute_pragmatic_lifespan,
    estimate_10yr_cvd_risk_score2,
    calculate_heart_health_index,
    calculate_biological_age,
)
from lab_parser import evaluate_lab_report_for_prediction


def _detect_contradictions_and_effective_state(
    data: UserHabitsInput,
    custom_nuance_analysis: dict | None,
    lab_data: dict | None
) -> dict:
    """
    Перекрёстная верификация входных данных пользователя:
    проверяет противоречия между полями анкеты (например: '0 сигарет', '0 алкоголя', 'без диагнозов')
    и тем, что пользователь написал в свободном тексте (например: 'курю вейп', 'перенес инфаркты')
    или прикрепил в анализах.
    """
    nuance_text = (data.custom_health_nuance or "").lower()
    lab_abn_text = " ".join(lab_data.get("detected_abnormalities", [])).lower() if lab_data else ""
    combined_free_text = f"{nuance_text} {lab_abn_text}"

    contradictions_notes = []

    # 1. Проверка скрытого никотина (вейп, электронные сигареты, айкос, кальян, снюс) при smoking_cigs_per_day == 0
    nicotine_keywords = (
        "вейп", "вэйп", "vape", "электрон", "парю", "парить", "жиж", "подик", "hqd", "одноразк",
        "айкос", "iqos", "glo", "гло", "стики", "кальян", "снюс", "никотин", "табак", "сигар", "курю"
    )
    negated_smoking = any(p in nuance_text for p in ("не курю", "бросил курить", "отказался от курения", "не парю"))
    has_hidden_nicotine = False
    if data.smoking_cigs_per_day == 0 and not negated_smoking:
        if any(kw in nuance_text for kw in nicotine_keywords):
            has_hidden_nicotine = True
            contradictions_notes.append(
                "В анкете указано 0 обычных сигарет, но в дополнительных сведениях указано потребление никотина "
                "(вейп / электронные сигареты / кальян / системы нагревания табака). Бонус за отказ от курения отменен; "
                "СТРОГО ЗАПРЕЩЕНО хвалить пользователя за отсутствие курения!"
            )

    # 2. Проверка скрытого алкоголя при alcohol_drinks_per_week == 0
    alcohol_keywords = ("пью пиво", "выпиваю", "алкогол", "вино ", "водк", "коньяк", "виски", "запой", "похмель", "пивной")
    negated_alcohol = any(p in nuance_text for p in ("не пью", "алкоголь не употребляю", "трезв"))
    has_hidden_alcohol = False
    if data.alcohol_drinks_per_week == 0 and not negated_alcohol:
        if any(kw in nuance_text for kw in alcohol_keywords):
            has_hidden_alcohol = True
            contradictions_notes.append(
                "В анкете указано 0 порций алкоголя, но в дополнительных сведениях упомянуто употребление алкоголя. "
                "Бонус за трезвость отменен; ЗАПРЕЩЕНО хвалить за полный отказ от алкоголя!"
            )

    # 3. Проверка скрытого нездорового питания при fastfood_times_per_week == 0
    poor_diet_keywords = ("фастфуд", "сладкое каждый день", "много сладкого", "газировк", "кока-кол", "перееда", "чипсы", "питаюсь плохо")
    has_hidden_poor_diet = False
    if data.fastfood_times_per_week == 0 and any(kw in nuance_text for kw in poor_diet_keywords):
        has_hidden_poor_diet = True
        contradictions_notes.append(
            "В анкете указано 0 приемов фастфуда, но в тексте отмечены погрешности питания. Бонус за чистый рацион отменен."
        )

    # 4. Проверка ограничений подвижности при выбранном активном образе жизни
    immobility_keywords = ("лежачий", "не могу ходить", "коляск", "постельный режим", "не тренируюсь из-за травмы")
    has_activity_limitation = any(kw in nuance_text for kw in immobility_keywords)
    if has_activity_limitation and data.lifestyle_activity in ("moderate", "active_gym"):
        contradictions_notes.append(
            "Учтено физическое ограничение подвижности из текстового поля."
        )

    # 5. Синхронизация серьёзных диагнозов из текста и анализов с сердечно-сосудистыми индексами
    effective_has_cvd = data.has_cvd_infarction_angina or any(
        w in combined_free_text for w in ("инфаркт", "стенокарди", "ибс", "ишемическ", "стент", "шунтирован", "акш", "коронарн")
    )
    effective_has_stroke = data.has_stroke_tia or any(
        w in combined_free_text for w in ("инсульт", "тиа", "транзиторн", "ишемическая атака", "кровоизлияни")
    )
    effective_has_hypertension = data.has_hypertension or any(
        w in combined_free_text for w in ("гипертони", "гипертензи", "высокое давление")
    ) or bool(re.search(r"(?:давлен\w*|ад)\s*[:=-]?\s*(?:1[4-9]\d|2\d\d)", nuance_text))

    effective_has_diabetes = data.has_diabetes or any(
        w in combined_free_text for w in ("диабет", "диабетич", "гликированный гемоглобин (6.", "гликированный гемоглобин (7.", "гликированный гемоглобин (8.")
    )
    effective_has_atherosclerosis = data.has_atherosclerosis or any(
        w in combined_free_text for w in ("атеросклероз", "бляшк", "стеноз", "гиперхолестеринеми", "высокий уровень лпнп", "критически высокий лпнп")
    )
    effective_has_thrombosis = data.has_thrombosis_pe or any(
        w in combined_free_text for w in ("тромбоз", "тромбоэмболи", "тэла", "тромбофлебит")
    )
    effective_has_oncology = data.has_oncology or any(
        w in combined_free_text for w in ("онколог", "рак ", "опухол", "карцином", "меланом", "лимфом", "химиотерапи", "метастаз")
    )

    return {
        "has_hidden_nicotine": has_hidden_nicotine,
        "has_hidden_alcohol": has_hidden_alcohol,
        "has_hidden_poor_diet": has_hidden_poor_diet,
        "has_activity_limitation": has_activity_limitation,
        "effective_has_cvd": effective_has_cvd,
        "effective_has_stroke": effective_has_stroke,
        "effective_has_hypertension": effective_has_hypertension,
        "effective_has_diabetes": effective_has_diabetes,
        "effective_has_atherosclerosis": effective_has_atherosclerosis,
        "effective_has_thrombosis": effective_has_thrombosis,
        "effective_has_oncology": effective_has_oncology,
        "contradictions_notes": contradictions_notes,
    }


def calculate_prediction(data: UserHabitsInput, custom_nuance_analysis: dict | None = None) -> dict:
    """
    Прагматичный двухконтурный расчёт с:
    1) перекрёстной проверкой противоречий между анкетой, текстом и анализами,
    2) точным маржинальным расчётом вклада анализов и текстовых нюансов (чтобы цифра на плашке анализов
       всегда с точностью до 0.1 года совпадала с реальным изменением прогноза продолжительности жизни!).
    """
    loc_info = get_location_baseline(data.country_code, data.ru_region_code, data.gender, data.age)
    base_expectancy = loc_info["base_expectancy"]
    score2_mult = loc_info["score2_mult"]
    is_russia = (data.country_code == "RU")

    age_scale = max(0.45, min(1.05, (86.0 - data.age) / 42.0))

    height_m = data.height_cm / 100.0
    bmi = round(data.weight_kg / (height_m ** 2), 1)

    # Разбор лабораторного файла (если прикреплен)
    lab_data = evaluate_lab_report_for_prediction(data.lab_report_data, data) if data.lab_report_data else None
    lab_clusters = lab_data.get("clusters", {}) if lab_data else {}

    # Перекрёстная верификация анкеты против текста и анализов
    effective_flags = _detect_contradictions_and_effective_state(data, custom_nuance_analysis, lab_data)

    lab_glyc_yrs = float(lab_clusters.get("glycemic", {}).get("impact_years", 0.0))
    lab_lipid_yrs = float(lab_clusters.get("lipids", {}).get("impact_years", 0.0))
    lab_renal_yrs = float(lab_clusters.get("renal_hepatic", {}).get("impact_years", 0.0))
    lab_hem_yrs = float(lab_clusters.get("hematology_endocrine", {}).get("impact_years", 0.0))
    lab_inf_yrs = float(lab_clusters.get("inflammation", {}).get("impact_years", 0.0))
    lab_coag_yrs = float(lab_clusters.get("coagulation_cardiac", {}).get("impact_years", 0.0))
    lab_unlisted_yrs = float(lab_data.get("unlisted_impact_years", 0.0)) if lab_data else 0.0

    raw_factors_curr = {}
    raw_factors_pot = {}
    medication_saved_years = 0.0
    medical_alerts = []

    def add_raw_factor(name: str, current_yrs: float, potential_yrs: float):
        c_val = round(current_yrs * age_scale, 3)
        p_val = round(potential_yrs * age_scale, 3)
        if abs(c_val) >= 0.02:
            raw_factors_curr[name] = c_val
        if abs(p_val) >= 0.02:
            raw_factors_pot[name] = p_val

    # --- 1. КУРЕНИЕ (С ЗАЩИТОЙ ОТ ПРОТИВОРЕЧИЯ ПРИ ВЕЙПЕ / КАЛЬЯНЕ / АЙКОСЕ В ТЕКСТЕ) ---
    clean_smoke_bonus = 2.3 if data.gender == "male" else 1.4

    if data.smoking_cigs_per_day > 0:
        cigs = data.smoking_cigs_per_day
        if cigs <= 5:
            smoke_curr = -2.2 - (cigs * 0.40)
        elif cigs <= 20:
            smoke_curr = -4.2 - ((cigs - 5) * 0.28)
        else:
            smoke_curr = -8.4 - min(5.0, (cigs - 20) * 0.14)

        irreversible_smoke_ratio = max(0.18, min(0.50, (data.age - 18) * 0.008))
        smoke_pot = (smoke_curr * irreversible_smoke_ratio) + (clean_smoke_bonus * (1.0 - irreversible_smoke_ratio) * 0.6)
        add_raw_factor("smoking", smoke_curr, smoke_pot)

    elif effective_flags["has_hidden_nicotine"]:
        # Пользователь поставил 0 сигарет, но в тексте указал вейп/электронки/кальян:
        # бонус за некурение НЕ начисляется (сам штраф за вейп идет в блок custom_nuance)
        pass

    elif data.past_smoking_years > 0 and data.past_smoking_cigs_per_day > 0:
        pack_years = (data.past_smoking_cigs_per_day / 20.0) * data.past_smoking_years
        initial_loss = -min(9.0, pack_years * 0.30)
        decay = 0.5 ** (data.years_since_quitting_smoking / 6.5)
        residual_penalty = initial_loss * max(0.12, decay)
        past_smoke_curr = round(clean_smoke_bonus * (1.0 - max(0.15, decay)) + residual_penalty, 2)
        add_raw_factor("past_smoking", past_smoke_curr, max(past_smoke_curr, past_smoke_curr + 0.3))
    else:
        add_raw_factor("smoking", clean_smoke_bonus, clean_smoke_bonus)

    # --- 2. АЛКОГОЛЬ (С ЗАЩИТОЙ ОТ ПРОТИВОРЕЧИЯ В ТЕКСТЕ) ---
    clean_alc_bonus = 1.2 if data.gender == "male" else 0.7
    drinks = data.alcohol_drinks_per_week

    if drinks == 0 and effective_flags["has_hidden_alcohol"]:
        alc_curr = 0.0
    elif drinks == 0:
        alc_curr = clean_alc_bonus
    elif drinks <= 2:
        alc_curr = round(clean_alc_bonus * 0.55, 2)
    elif drinks <= 5:
        alc_curr = -0.2
    elif drinks <= 10:
        alc_curr = -0.3 - ((drinks - 4) * 0.25)
    elif drinks <= 20:
        alc_curr = -1.8 - ((drinks - 10) * 0.32)
    else:
        alc_curr = -5.0 - min(7.5, (drinks - 20) * 0.22)

    if data.past_heavy_alcohol == "yes_quit_recently":
        alc_curr -= 1.5
    elif data.past_heavy_alcohol == "yes_quit_over_3y_ago":
        alc_curr -= 0.7

    if drinks == 0 and data.past_heavy_alcohol == "no" and not effective_flags["has_hidden_alcohol"]:
        alc_pot = clean_alc_bonus
    elif drinks > 12:
        alc_pot = alc_curr * 0.25
    else:
        alc_pot = round(clean_alc_bonus * 0.75, 2) if data.past_heavy_alcohol == "no" else -0.3
    add_raw_factor("alcohol", alc_curr, alc_pot)

    # --- 3. ИМТ (РОСТ И ВЕС) ---
    has_metabolic_abnormality = (
        lab_glyc_yrs < -0.5 or lab_lipid_yrs < -0.5
        or effective_flags["effective_has_diabetes"]
        or effective_flags["effective_has_atherosclerosis"]
    )
    if bmi < 16.5:
        bmi_curr, bmi_pot = -5.0, -1.4
    elif bmi < 18.5:
        bmi_curr, bmi_pot = -1.8, -0.2
    elif bmi <= 24.8:
        # Если в анализах или тексте есть диабет/высокий холестерин, уменьшаем метаболический бонус за нормальный вес
        bmi_curr = 0.4 if has_metabolic_abnormality else 0.9
        bmi_pot = 0.9
    elif bmi <= 27.0:
        bmi_curr, bmi_pot = 0.1, 0.7
    elif bmi <= 30.0:
        bmi_curr = -((bmi - 26.8) * 0.50)
        bmi_pot = 0.4
    elif bmi <= 35.0:
        bmi_curr = -1.6 - ((bmi - 30.0) * 0.60)
        bmi_pot = bmi_curr * 0.22
    elif bmi <= 40.0:
        bmi_curr = -4.6 - ((bmi - 35.0) * 0.70)
        bmi_pot = bmi_curr * 0.30
    else:
        bmi_curr = -8.1 - min(4.5, (bmi - 40.0) * 0.35)
        bmi_pot = bmi_curr * 0.38

    if bmi > 25.0 and bmi_curr < 0 and (data.has_diabetes or data.has_atherosclerosis):
        bmi_curr *= 0.85

    add_raw_factor("bmi", bmi_curr, bmi_pot)

    # --- 4. ОБРАЗ ЖИЗНИ И ШАГИ ---
    eff_lifestyle = "sedentary" if effective_flags["has_activity_limitation"] else data.lifestyle_activity
    lifestyle_map = {
        "sedentary": -1.6,
        "light": -0.3,
        "moderate": 1.1,
        "active_gym": 2.0
    }
    life_act_curr = lifestyle_map.get(eff_lifestyle, -0.3)

    steps = 1500 if effective_flags["has_activity_limitation"] else data.daily_steps
    if steps < 2500:
        steps_curr = -1.6
    elif steps < 4500:
        steps_curr = -0.9
    elif steps < 6500:
        steps_curr = 0.0
    elif steps <= 10000:
        steps_curr = 0.8 + ((steps - 6500) / 3500.0) * 0.6
    else:
        steps_curr = 1.4 + min(0.5, ((steps - 10000) / 10000.0) * 0.5)

    combined_activity_curr = round((life_act_curr + steps_curr) * 0.82, 2)
    combined_activity_pot = max(combined_activity_curr, 0.5 if effective_flags["has_activity_limitation"] else 2.1)
    add_raw_factor("activity", combined_activity_curr, combined_activity_pot)

    # --- 5. СОН, ПИТАНИЕ, СТРЕСС, ВРЕДНОЕ ПРОИЗВОДСТВО ---
    sh = data.sleep_hours
    if 7.0 <= sh <= 8.5:
        sleep_curr = 0.5
    elif 6.0 <= sh < 7.0:
        sleep_curr = -0.7
    elif 5.0 <= sh < 6.0:
        sleep_curr = -1.8
    elif sh < 5.0:
        sleep_curr = -3.0
    elif 8.5 < sh <= 9.5:
        sleep_curr = -0.6
    else:
        sleep_curr = -2.0
    add_raw_factor("sleep", sleep_curr, 0.5)

    ff = data.fastfood_times_per_week
    if ff == 0 and not effective_flags["has_hidden_poor_diet"]:
        diet_curr = 0.8
    elif ff == 0 and effective_flags["has_hidden_poor_diet"]:
        diet_curr = -0.8
    elif ff <= 2:
        diet_curr = 0.3 if not effective_flags["has_hidden_poor_diet"] else -0.5
    elif ff <= 6:
        diet_curr = -((ff - 2) * 0.38)
    elif ff <= 14:
        diet_curr = -1.52 - ((ff - 6) * 0.28)
    else:
        diet_curr = -3.76 - min(2.4, (ff - 14) * 0.16)
    add_raw_factor("nutrition", diet_curr, max(diet_curr, 0.7))

    stress_map_curr = {"low": 0.4, "medium": -0.4, "high": -1.9}
    stress_map_pot = {"low": 0.4, "medium": 0.2, "high": -0.3}
    add_raw_factor("stress", stress_map_curr[data.stress_level], stress_map_pot[data.stress_level])

    if data.hazardous_work == "moderate":
        add_raw_factor("hazardous_work", -1.8, -0.8)
    elif data.hazardous_work == "high":
        add_raw_factor("hazardous_work", -4.0, -1.9)
        medical_alerts.append(
            "Работа на вредном производстве требует обязательного использования СИЗ и ежегодного профильного медосмотра."
        )

    # --- 6. ХРОНИЧЕСКИЕ ЗАБОЛЕВАНИЯ ИЗ АНКЕТЫ ---
    has_any_chronic = any([
        data.has_hypertension, data.has_diabetes, data.has_cvd_infarction_angina,
        data.has_stroke_tia, data.has_atherosclerosis, data.has_thrombosis_pe,
        data.has_oncology, data.has_rheumatoid_arthritis, data.has_urinary_infection, data.has_ibd_polyps
    ])
    has_severe_pathology = any([
        effective_flags["effective_has_diabetes"],
        effective_flags["effective_has_cvd"],
        effective_flags["effective_has_stroke"],
        effective_flags["effective_has_thrombosis"],
        effective_flags["effective_has_oncology"]
    ])

    if data.has_hypertension:
        if data.meds_antihypertensive:
            bp_curr = -1.6
            medication_saved_years += 2.6 * age_scale
        else:
            bp_curr = -4.2
            medical_alerts.append(
                "Артериальная гипертензия без регулярного приёма препаратов отнимает около 4 лет. Подбор терапии с врачом вернёт более половины этих лет."
            )
        add_raw_factor("blood_pressure", bp_curr, -1.4)

    if data.has_atherosclerosis:
        if data.meds_statins_lipid:
            chol_curr = -1.6
            medication_saved_years += 2.2 * age_scale
        else:
            chol_curr = -3.8
        add_raw_factor("cholesterol", chol_curr, -1.5)

    if data.has_diabetes:
        if data.meds_antidiabetic:
            diab_curr = -3.4
            medication_saved_years += 3.1 * age_scale
        else:
            diab_curr = -6.5
            medical_alerts.append(
                "Сахарный диабет без постоянной сахароснижающей терапии ускоряет поражение сосудов и почек."
            )
        add_raw_factor("glucose_diabetes", diab_curr, -2.8)

    chronic_curr_sum = 0.0
    chronic_pot_sum = 0.0
    if data.has_cvd_infarction_angina:
        if data.meds_anticoagulants or data.meds_statins_lipid or data.meds_antiarrhythmic:
            chronic_curr_sum -= 4.5
            medication_saved_years += 3.0 * age_scale
        else:
            chronic_curr_sum -= 7.5
        chronic_pot_sum -= 4.2

    if data.has_stroke_tia:
        if data.meds_anticoagulants or data.meds_antihypertensive:
            chronic_curr_sum -= 4.2
            medication_saved_years += 2.8 * age_scale
        else:
            chronic_curr_sum -= 7.0
        chronic_pot_sum -= 4.0

    if data.has_thrombosis_pe:
        if data.meds_anticoagulants:
            chronic_curr_sum -= 2.0
            medication_saved_years += 2.5 * age_scale
        else:
            chronic_curr_sum -= 4.5
        chronic_pot_sum -= 1.8

    if data.has_oncology:
        chronic_curr_sum -= 6.2
        chronic_pot_sum -= 4.8
    if data.has_rheumatoid_arthritis:
        chronic_curr_sum -= 2.5
        chronic_pot_sum -= 1.6
    if data.has_ibd_polyps:
        chronic_curr_sum -= 1.6
        chronic_pot_sum -= 0.9
    if data.has_urinary_infection:
        chronic_curr_sum -= 0.8
        chronic_pot_sum -= 0.3
    if has_any_chronic and data.meds_asthma_copd:
        chronic_curr_sum -= 2.0
        chronic_pot_sum -= 1.2

    if chronic_curr_sum < 0:
        add_raw_factor("chronic_diseases", chronic_curr_sum, chronic_pot_sum)

    if has_any_chronic:
        med_side_curr, med_side_pot = 0.0, 0.0
        if data.meds_glucocorticoids:
            med_side_curr -= 1.8
            med_side_pot -= 1.0
        if data.meds_daily_nsaids:
            med_side_curr -= 1.4
            med_side_pot -= 0.2
        if med_side_curr < 0:
            add_raw_factor("medication_side_risks", med_side_curr, med_side_pot)

    # Сохраняем базовую сумму анкетных факторов (до добавления анализов и текстового поля)
    questionnaire_raw_curr = sum(raw_factors_curr.values())
    questionnaire_raw_pot = sum(raw_factors_pot.values())

    # --- 7. ЛАБОРАТОРНЫЕ АНАЛИЗЫ ИЗ ФАЙЛА (С УБЫВАЮЩЕЙ ПРЕДЕЛЬНОЙ ТЯЖЕСТЬЮ КЛАСТЕРОВ) ---
    if lab_data:
        cluster_penalties_curr = []
        cluster_penalties_pot = []

        if lab_glyc_yrs < 0:
            irrev_g = float(lab_clusters.get("glycemic", {}).get("irreversible_fraction", 0.30))
            if not data.has_diabetes:
                cluster_penalties_curr.append(lab_glyc_yrs)
                cluster_penalties_pot.append(lab_glyc_yrs * irrev_g)
            else:
                already_diab = -3.4 if data.meds_antidiabetic else -6.5
                if lab_glyc_yrs < already_diab:
                    diff_g = lab_glyc_yrs - already_diab
                    cluster_penalties_curr.append(diff_g)
                    cluster_penalties_pot.append(diff_g * 0.5)

        if lab_lipid_yrs < 0:
            irrev_l = float(lab_clusters.get("lipids", {}).get("irreversible_fraction", 0.30))
            if not data.has_atherosclerosis:
                cluster_penalties_curr.append(lab_lipid_yrs)
                cluster_penalties_pot.append(lab_lipid_yrs * irrev_l)
            else:
                already_chol = -1.6 if data.meds_statins_lipid else -3.8
                if lab_lipid_yrs < already_chol:
                    diff_l = lab_lipid_yrs - already_chol
                    cluster_penalties_curr.append(diff_l)
                    cluster_penalties_pot.append(diff_l * 0.4)

        if lab_renal_yrs < 0:
            irrev_rh = float(lab_clusters.get("renal_hepatic", {}).get("irreversible_fraction", 0.45))
            cluster_penalties_curr.append(lab_renal_yrs)
            cluster_penalties_pot.append(lab_renal_yrs * irrev_rh)

        if lab_hem_yrs < 0:
            irrev_hem = float(lab_clusters.get("hematology_endocrine", {}).get("irreversible_fraction", 0.20))
            cluster_penalties_curr.append(lab_hem_yrs)
            cluster_penalties_pot.append(lab_hem_yrs * irrev_hem)

        if lab_inf_yrs < 0:
            inf_mult = 0.5 if (data.has_rheumatoid_arthritis or data.has_ibd_polyps) else 1.0
            cluster_penalties_curr.append(lab_inf_yrs * inf_mult)
            cluster_penalties_pot.append((lab_inf_yrs * inf_mult) * 0.25)

        if lab_coag_yrs < 0:
            coag_mult = 0.4 if (data.has_thrombosis_pe or data.has_cvd_infarction_angina) else 1.0
            cluster_penalties_curr.append(lab_coag_yrs * coag_mult)
            cluster_penalties_pot.append((lab_coag_yrs * coag_mult) * 0.45)

        if lab_unlisted_yrs < 0:
            unlisted_irrev = float(lab_data.get("unlisted_irrev_fraction", 0.35))
            cluster_penalties_curr.append(lab_unlisted_yrs)
            cluster_penalties_pot.append(lab_unlisted_yrs * unlisted_irrev)

        # Сортируем кластерные отклонения по тяжести и применяем закон убывающей предельной тяжести
        # (1-й кластер: 100%, 2-й: 65%, 3-й: 45%, 4+: 30%), чтобы сопутствующие сдвиги не складывались линейно
        paired = sorted(zip(cluster_penalties_curr, cluster_penalties_pot), key=lambda x: x[0])
        weights = [1.0, 0.65, 0.45, 0.30, 0.25, 0.20, 0.15]
        net_lab_curr = sum(c * weights[min(i, len(weights) - 1)] for i, (c, _) in enumerate(paired))
        net_lab_pot = sum(p * weights[min(i, len(weights) - 1)] for i, (_, p) in enumerate(paired))

        all_normal_bonus = float(lab_data.get("all_normal_bonus_years", 0.0))
        if net_lab_curr == 0.0 and all_normal_bonus > 0:
            net_lab_curr = all_normal_bonus
            net_lab_pot = all_normal_bonus

        add_raw_factor("lab_analysis", net_lab_curr, net_lab_pot)

    # --- 8. ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ ИЗ ТЕКСТОВОГО ПОЛЯ ---
    if custom_nuance_analysis:
        c_impact = float(custom_nuance_analysis.get("impact_years_current", 0.0))
        irrev_frac = float(custom_nuance_analysis.get("irreversible_fraction", 0.55))

        if lab_data and lab_data.get("detected_abnormalities") and c_impact < 0:
            nuance_lower = (data.custom_health_nuance or "").lower()
            overlap_keywords = ["холестерин", "сахар", "глюкоз", "гемоглобин", "анеми", "ферритин", "креатинин", "алт", "аст", "ттг"]
            if any(k in nuance_lower for k in overlap_keywords):
                c_impact *= 0.35

        if c_impact <= -5.0:
            has_severe_pathology = True

        p_impact = (c_impact * irrev_frac) if c_impact < 0 else c_impact
        add_raw_factor("custom_nuance", c_impact, p_impact)

    # --- 9. ТОЧНЫЙ МАРЖИНАЛЬНЫЙ РАСЧЁТ ПРОГНОЗА И СИНХРОНИЗАЦИЯ ВКЛАДА КАЖДОГО БЛОКА ---
    # Шаг 9.1: Прогноз только по анкете (без анализов и без текстового поля)
    q_delta, pred_questionnaire_only, _ = compute_pragmatic_lifespan(
        base_expectancy, data.age, questionnaire_raw_curr, has_severe_pathology=has_severe_pathology
    )
    _, pot_questionnaire_only, _ = compute_pragmatic_lifespan(
        base_expectancy, data.age, max(questionnaire_raw_curr, questionnaire_raw_pot), has_severe_pathology=has_severe_pathology
    )

    # Шаг 9.2: Прогноз с учётом прикрепленных анализов
    raw_with_labs_curr = questionnaire_raw_curr + raw_factors_curr.get("lab_analysis", 0.0)
    raw_with_labs_pot = questionnaire_raw_pot + raw_factors_pot.get("lab_analysis", 0.0)
    _, pred_with_labs, _ = compute_pragmatic_lifespan(
        base_expectancy, data.age, raw_with_labs_curr, has_severe_pathology=has_severe_pathology
    )
    _, pot_with_labs, _ = compute_pragmatic_lifespan(
        base_expectancy, data.age, max(raw_with_labs_curr, raw_with_labs_pot), has_severe_pathology=has_severe_pathology
    )

    # Шаг 9.3: Финальный прогноз (анкета + анализы + текстовое поле)
    raw_total_curr = sum(raw_factors_curr.values())
    raw_total_pot = max(raw_total_curr, sum(raw_factors_pot.values()))

    years_delta, predicted_expectancy, remaining_years_current = compute_pragmatic_lifespan(
        base_expectancy, data.age, raw_total_curr, has_severe_pathology=has_severe_pathology
    )
    _, potential_expectancy, remaining_years_potential = compute_pragmatic_lifespan(
        base_expectancy, data.age, raw_total_pot, has_severe_pathology=has_severe_pathology
    )

    if potential_expectancy < predicted_expectancy:
        potential_expectancy = predicted_expectancy
        remaining_years_potential = remaining_years_current

    remaining_years_gain = round(remaining_years_potential - remaining_years_current, 1)

    # Точный реальный сдвиг прогноза от анализов и от текстового поля (совпадает с изменением цифр на экране до 0.1 года!)
    effective_lab_impact_curr = round(pred_with_labs - pred_questionnaire_only, 1) if "lab_analysis" in raw_factors_curr else 0.0
    effective_lab_impact_pot = round(pot_with_labs - pot_questionnaire_only, 1) if "lab_analysis" in raw_factors_pot else 0.0

    effective_custom_impact_curr = round(predicted_expectancy - pred_with_labs, 1) if "custom_nuance" in raw_factors_curr else 0.0
    effective_custom_impact_pot = round(potential_expectancy - pot_with_labs, 1) if "custom_nuance" in raw_factors_pot else 0.0

    # Формируем финальный словарь factors_years (с пропорциональной калибровкой анкетных факторов под реальный сдвиг)
    factors_years = {}
    factors_microlives = {}
    if abs(loc_info["regional_delta_years"]) >= 0.2:
        factors_years["country_environment"] = loc_info["regional_delta_years"]

    q_keys = [k for k in raw_factors_curr.keys() if k not in ("lab_analysis", "custom_nuance")]
    for k in q_keys:
        val = round(raw_factors_curr[k], 1)
        if abs(val) >= 0.1:
            factors_years[k] = val
            factors_microlives[k] = round((val / max(base_expectancy - data.age, 15.0)) * 48.0, 1)

    lab_summary_info = None
    if lab_data:
        # Если в анализах есть отклонения, но из-за округления разница < 0.1, показываем минимум -0.2
        if effective_lab_impact_curr == 0.0 and raw_factors_curr.get("lab_analysis", 0.0) < -0.15:
            effective_lab_impact_curr = -0.2
        elif effective_lab_impact_curr == 0.0 and raw_factors_curr.get("lab_analysis", 0.0) > 0.15:
            effective_lab_impact_curr = 0.2

        factors_years["lab_analysis"] = effective_lab_impact_curr
        factors_microlives["lab_analysis"] = round((effective_lab_impact_curr / max(base_expectancy - data.age, 15.0)) * 48.0, 1)
        lab_summary_info = {
            "filename": lab_data.get("filename", "Анализы"),
            "impact_years": effective_lab_impact_curr,
            "potential_impact_years": effective_lab_impact_pot,
            "impact_microlives": factors_microlives["lab_analysis"],
            "detected_abnormalities": lab_data.get("detected_abnormalities", []),
            "normal_markers": lab_data.get("normal_markers", []),
            "web_sources": lab_data.get("web_sources", []),
            "clinical_summary": lab_data.get("clinical_summary", ""),
            "recommended_action": lab_data.get("recommended_action", "")
        }

    custom_nuance_info = None
    if custom_nuance_analysis:
        if effective_custom_impact_curr == 0.0 and raw_factors_curr.get("custom_nuance", 0.0) < -0.15:
            effective_custom_impact_curr = -0.2
        elif effective_custom_impact_curr == 0.0 and raw_factors_curr.get("custom_nuance", 0.0) > 0.15:
            effective_custom_impact_curr = 0.2

        factors_years["custom_nuance"] = effective_custom_impact_curr
        factors_microlives["custom_nuance"] = round((effective_custom_impact_curr / max(base_expectancy - data.age, 15.0)) * 48.0, 1)
        custom_nuance_info = {
            **custom_nuance_analysis,
            "impact_years": effective_custom_impact_curr,
            "potential_impact_years": effective_custom_impact_pot,
            "impact_microlives": factors_microlives["custom_nuance"]
        }

    # --- 10. ИНДЕКС РЕСУРСА СЕРДЦА, 10-ЛЕТНИЙ РИСК (SCORE2) И БИОЛОГИЧЕСКИЙ ВОЗРАСТ ---
    aha_le8_score = calculate_heart_health_index(
        data,
        bmi,
        effective_flags=effective_flags,
        lab_impact_years=effective_lab_impact_curr,
        custom_impact_years=effective_custom_impact_curr
    )

    biological_age = calculate_biological_age(
        chronological_age=data.age,
        years_delta=years_delta,
        lab_impact_years=effective_lab_impact_curr,
        custom_impact_years=effective_custom_impact_curr,
        aha_le8_score=aha_le8_score
    )

    has_established_cvd = (
        effective_flags["effective_has_cvd"]
        or effective_flags["effective_has_stroke"]
        or effective_flags["effective_has_atherosclerosis"]
    )
    extra_cvd_mult = 1.0
    if custom_nuance_info and custom_nuance_info.get("is_cardiovascular") and effective_custom_impact_curr < 0:
        extra_cvd_mult *= min(1.55, 1.0 + abs(effective_custom_impact_curr) * 0.06)

    if lab_lipid_yrs < -1.0 and not effective_flags["effective_has_atherosclerosis"]:
        extra_cvd_mult *= 1.22
    if lab_glyc_yrs < -1.5 and not effective_flags["effective_has_diabetes"]:
        extra_cvd_mult *= 1.25

    cvd_10yr_risk = estimate_10yr_cvd_risk_score2(
        age=data.age,
        gender=data.gender,
        bmi=bmi,
        cigs_per_day=data.smoking_cigs_per_day,
        has_hidden_nicotine=effective_flags["has_hidden_nicotine"],
        alcohol_per_week=data.alcohol_drinks_per_week,
        has_hidden_alcohol=effective_flags["has_hidden_alcohol"],
        lifestyle_activity="sedentary" if effective_flags["has_activity_limitation"] else data.lifestyle_activity,
        has_hypertension=effective_flags["effective_has_hypertension"],
        meds_antihypertensive=data.meds_antihypertensive,
        has_atherosclerosis=effective_flags["effective_has_atherosclerosis"],
        meds_statins=data.meds_statins_lipid,
        has_diabetes=effective_flags["effective_has_diabetes"],
        meds_antidiabetic=data.meds_antidiabetic,
        has_established_cvd=has_established_cvd,
        regional_multiplier=score2_mult * extra_cvd_mult
    )
    cvd_10yr_risk_potential = estimate_10yr_cvd_risk_score2(
        age=data.age,
        gender=data.gender,
        bmi=min(24.5, bmi),
        cigs_per_day=0,
        has_hidden_nicotine=False,
        alcohol_per_week=0,
        has_hidden_alcohol=False,
        lifestyle_activity="active_gym" if data.lifestyle_activity == "active_gym" else "moderate",
        has_hypertension=effective_flags["effective_has_hypertension"],
        meds_antihypertensive=True if effective_flags["effective_has_hypertension"] else False,
        has_atherosclerosis=effective_flags["effective_has_atherosclerosis"],
        meds_statins=True if effective_flags["effective_has_atherosclerosis"] else False,
        has_diabetes=effective_flags["effective_has_diabetes"],
        meds_antidiabetic=True if effective_flags["effective_has_diabetes"] else False,
        has_established_cvd=has_established_cvd,
        regional_multiplier=score2_mult * (1.0 + (extra_cvd_mult - 1.0) * 0.4)
    )
    cvd_10yr_risk_potential = round(min(cvd_10yr_risk, cvd_10yr_risk_potential), 1)

    confidence_score = 82
    if has_any_chronic:
        confidence_score += 5
    if custom_nuance_info:
        confidence_score += 5
    if lab_summary_info:
        confidence_score += 6
    confidence_score = min(98, confidence_score)

    if is_russia:
        cigs_net_year = (data.smoking_cigs_per_day / 20.0) * 180 * 365
        vape_net_year = 18000 if effective_flags["has_hidden_nicotine"] and data.smoking_cigs_per_day == 0 else 0
        alcohol_net_year = data.alcohol_drinks_per_week * 150 * 52
        fastfood_net_year = data.fastfood_times_per_week * 120 * 52
        raw_annual_savings = (cigs_net_year + vape_net_year + alcohol_net_year + fastfood_net_year) * 0.85
        annual_waste_rub = int(round(raw_annual_savings, -2))
        ten_year_savings_rub = annual_waste_rub * 10
    else:
        annual_waste_rub = 0
        ten_year_savings_rub = 0

    remaining_days = int(round(remaining_years_current * 365))

    return {
        "is_russia": is_russia,
        "country_code": data.country_code,
        "ru_region_code": data.ru_region_code,
        "country_name": loc_info["location_display"],
        "official_stat_avg": loc_info["official_stat_avg"],
        "official_total_avg": loc_info["official_total_avg"],
        "gender_label": loc_info["gender_label"],
        "regional_env_note": loc_info["env_note"],
        "bmi": bmi,
        "base_expectancy": base_expectancy,
        "predicted_expectancy": predicted_expectancy,
        "potential_expectancy": potential_expectancy,
        "remaining_years_current": remaining_years_current,
        "remaining_years_potential": remaining_years_potential,
        "remaining_years_gain": remaining_years_gain,
        "years_delta": years_delta,
        "biological_age": biological_age,
        "remaining_days": remaining_days,
        "annual_waste_rub": annual_waste_rub,
        "ten_year_savings_rub": ten_year_savings_rub,
        "aha_le8_score": aha_le8_score,
        "cvd_10yr_risk_percent": cvd_10yr_risk,
        "cvd_10yr_risk_potential_percent": cvd_10yr_risk_potential,
        "medication_saved_years": round(medication_saved_years, 1),
        "confidence_score_percent": confidence_score,
        "medical_alerts": medical_alerts,
        "contradictions_notes": effective_flags["contradictions_notes"],
        "effective_flags": {
            "has_hidden_nicotine": effective_flags["has_hidden_nicotine"],
            "has_hidden_alcohol": effective_flags["has_hidden_alcohol"],
            "has_hidden_poor_diet": effective_flags["has_hidden_poor_diet"],
            "has_activity_limitation": effective_flags["has_activity_limitation"]
        },
        "lab_summary_info": lab_summary_info,
        "custom_nuance_info": custom_nuance_info,
        "factors_years": factors_years,
        "factors_microlives": factors_microlives
    }
