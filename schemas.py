from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional

# Справочник стран мира (ВОЗ / WHO Global Health Observatory).
# male_stat_avg / female_stat_avg — реальная среднестатистическая ОПЖ мужчин и женщин в популяции.
COUNTRY_PROFILES = {
    "RU": {
        "name": "Россия",
        "who_official_avg": 73.4,
        "male_stat_avg": 68.2,
        "female_stat_avg": 78.6,
        "risk_region": "very_high",
        "score2_mult": 1.30,
        "env_note": "Высокий базовый популяционный сосудистый риск и выраженный разрыв продолжительности жизни между мужчинами и женщинами."
    },
    "BY": {
        "name": "Беларусь",
        "who_official_avg": 74.2,
        "male_stat_avg": 69.3,
        "female_stat_avg": 79.1,
        "risk_region": "very_high",
        "score2_mult": 1.25,
        "env_note": "Регион высокого базового сердечно-сосудистого риска."
    },
    "KZ": {
        "name": "Казахстан",
        "who_official_avg": 74.4,
        "male_stat_avg": 70.1,
        "female_stat_avg": 78.4,
        "risk_region": "very_high",
        "score2_mult": 1.28,
        "env_note": "Резко континентальный климат и повышенная кардиометаболическая нагрузка."
    },
    "UZ": {
        "name": "Узбекистан",
        "who_official_avg": 73.8,
        "male_stat_avg": 71.2,
        "female_stat_avg": 76.6,
        "risk_region": "very_high",
        "score2_mult": 1.28,
        "env_note": "Тёплый климат при высокой доле быстрых углеводов и соли в традиционном рационе."
    },
    "AM_GE": {
        "name": "Армения / Грузия",
        "who_official_avg": 75.2,
        "male_stat_avg": 70.8,
        "female_stat_avg": 79.4,
        "risk_region": "very_high",
        "score2_mult": 1.20,
        "env_note": "Благоприятная инсоляция при высокой распространенности артериальной гипертензии и курения."
    },
    "TR_PL": {
        "name": "Турция / Восточная Европа (Польша, Чехия, Сербия)",
        "who_official_avg": 77.6,
        "male_stat_avg": 74.2,
        "female_stat_avg": 81.0,
        "risk_region": "high",
        "score2_mult": 1.05,
        "env_note": "Умеренно высокий уровень превентивной медицины."
    },
    "CN_BR": {
        "name": "Китай / Латинская Америка / ОАЭ",
        "who_official_avg": 78.2,
        "male_stat_avg": 75.4,
        "female_stat_avg": 81.0,
        "risk_region": "high",
        "score2_mult": 1.02,
        "env_note": "Развитая городская инфраструктура при повышенной нагрузке воздуха в мегаполисах."
    },
    "US_CA": {
        "name": "США / Канада",
        "who_official_avg": 79.5,
        "male_stat_avg": 76.8,
        "female_stat_avg": 82.1,
        "risk_region": "moderate",
        "score2_mult": 0.92,
        "env_note": "Высокая доступность кардиохирургии при высокой распространенности ультрапереработанной пищи."
    },
    "DE_IT": {
        "name": "Западная и Северная Европа (Германия, Италия, Швеция)",
        "who_official_avg": 81.8,
        "male_stat_avg": 79.4,
        "female_stat_avg": 84.0,
        "risk_region": "moderate",
        "score2_mult": 0.85,
        "env_note": "Сильная система ранней диагностики и профилактики сердечно-сосудистых заболеваний."
    },
    "JP_CH": {
        "name": "Япония / Швейцария / Испания / Франция / Израиль",
        "who_official_avg": 83.8,
        "male_stat_avg": 81.1,
        "female_stat_avg": 86.3,
        "risk_region": "low",
        "score2_mult": 0.72,
        "env_note": "Кластер минимального сосудистого риска и передовой превентивной медицины."
    }
}

# Региональная статистика РФ по данным Росстата (с разделением на мужскую и женскую популяционную ОПЖ)
RU_REGION_PROFILES = {
    "RU_AVG": {
        "name": "Средний показатель по РФ",
        "rosstat_avg": 73.4,
        "male_stat_avg": 68.2,
        "female_stat_avg": 78.6,
        "score2_mult": 1.30,
        "env_note": "Среднероссийский демографический профиль Росстата."
    },
    "RU_MOW": {
        "name": "Москва",
        "rosstat_avg": 79.4,
        "male_stat_avg": 75.2,
        "female_stat_avg": 82.8,
        "score2_mult": 1.08,
        "env_note": "Высокая плотность сосудистых центров, быстрый доезд скорой помощи и доступность ранней диагностики."
    },
    "RU_SPE": {
        "name": "Санкт-Петербург",
        "rosstat_avg": 76.8,
        "male_stat_avg": 72.1,
        "female_stat_avg": 80.6,
        "score2_mult": 1.16,
        "env_note": "Развитая городская медицина при влажном климате и дефиците солнечной инсоляции."
    },
    "RU_MOS": {
        "name": "Московская область",
        "rosstat_avg": 75.1,
        "male_stat_avg": 70.3,
        "female_stat_avg": 79.5,
        "score2_mult": 1.22,
        "env_note": "Инфраструктура столичной агломерации при высокой маятниковой нагрузке."
    },
    "RU_DAG_ING": {
        "name": "Республика Дагестан / Ингушетия / Кабардино-Балкария",
        "rosstat_avg": 79.8,
        "male_stat_avg": 76.4,
        "female_stat_avg": 82.6,
        "score2_mult": 1.10,
        "env_note": "Горно-морской климат, высокая инсоляция и традиционно низкое потребление табака и алкоголя."
    },
    "RU_TAT": {
        "name": "Республика Татарстан",
        "rosstat_avg": 75.3,
        "male_stat_avg": 69.8,
        "female_stat_avg": 80.2,
        "score2_mult": 1.20,
        "env_note": "Развитая сеть высокотехнологичной медицинской помощи в Приволжье."
    },
    "RU_KDA_STA": {
        "name": "Краснодарский край / Ставропольский край / Адыгея",
        "rosstat_avg": 75.0,
        "male_stat_avg": 70.1,
        "female_stat_avg": 79.6,
        "score2_mult": 1.20,
        "env_note": "Благоприятный южный климат и доступность свежих овощей и фруктов."
    },
    "RU_KHM_TYU": {
        "name": "Тюменская область / ХМАО — Югра / ЯНАО",
        "rosstat_avg": 75.8,
        "male_stat_avg": 70.9,
        "female_stat_avg": 80.3,
        "score2_mult": 1.20,
        "env_note": "Высокая обеспеченность регионального здравоохранения при северном климате."
    },
    "RU_BEL_VOR": {
        "name": "Центральное Черноземье (Белгородская, Воронежская, Липецкая, Тамбовская обл.)",
        "rosstat_avg": 74.1,
        "male_stat_avg": 68.8,
        "female_stat_avg": 79.2,
        "score2_mult": 1.25,
        "env_note": "Умеренно-континентальный климат."
    },
    "RU_ROS_VOL": {
        "name": "Ростовская / Волгоградская / Астраханская область",
        "rosstat_avg": 74.0,
        "male_stat_avg": 69.0,
        "female_stat_avg": 78.9,
        "score2_mult": 1.25,
        "env_note": "Южный степной регион с летней тепловой нагрузкой на сердце."
    },
    "RU_SAM_SAR": {
        "name": "Поволжье (Самарская, Саратовская, Ульяновская, Пензенская обл.)",
        "rosstat_avg": 73.5,
        "male_stat_avg": 68.2,
        "female_stat_avg": 78.7,
        "score2_mult": 1.28,
        "env_note": "Показатели соответствуют среднероссийскому демографическому уровню."
    },
    "RU_NIZ_KIROV": {
        "name": "Нижегородская / Кировская область / Чувашия / Марий Эл / Мордовия",
        "rosstat_avg": 73.0,
        "male_stat_avg": 67.5,
        "female_stat_avg": 78.4,
        "score2_mult": 1.30,
        "env_note": "Промышленно-аграрный регион средней полосы."
    },
    "RU_CTR": {
        "name": "Центральная Россия (Ярославская, Тульская, Калужская, Рязанская, Тверская, Владимирская обл.)",
        "rosstat_avg": 72.7,
        "male_stat_avg": 67.2,
        "female_stat_avg": 78.1,
        "score2_mult": 1.32,
        "env_note": "Повышенная сосудистая нагрузка в осенне-зимний период."
    },
    "RU_NW": {
        "name": "Северо-Запад (Калининградская, Ленинградская, Вологодская, Новгородская, Псковская обл.)",
        "rosstat_avg": 72.8,
        "male_stat_avg": 67.4,
        "female_stat_avg": 78.2,
        "score2_mult": 1.30,
        "env_note": "Северо-западный климатический пояс с дефицитом солнечного света."
    },
    "RU_NORTH": {
        "name": "Европейский Север (Мурманская, Архангельская обл., Карелия, Коми)",
        "rosstat_avg": 71.8,
        "male_stat_avg": 66.2,
        "female_stat_avg": 77.5,
        "score2_mult": 1.35,
        "env_note": "Северный регион: холодовой спазм сосудов и полярная ночь."
    },
    "RU_URAL": {
        "name": "Урал и Предуралье (Свердловская, Челябинская, Оренбургская обл., Башкортостан, Пермский край, Удмуртия)",
        "rosstat_avg": 72.4,
        "male_stat_avg": 66.8,
        "female_stat_avg": 78.0,
        "score2_mult": 1.33,
        "env_note": "Крупный индустриальный регион с промышленной нагрузкой среды."
    },
    "RU_WSIB": {
        "name": "Западная Сибирь (Новосибирская, Омская, Томская обл., Алтайский край)",
        "rosstat_avg": 72.6,
        "male_stat_avg": 67.0,
        "female_stat_avg": 78.2,
        "score2_mult": 1.32,
        "env_note": "Резко континентальный климат с перепадами температур и давления."
    },
    "RU_KUZ_ESIB": {
        "name": "Кузбасс и Восточная Сибирь (Кемеровская, Иркутская обл., Красноярский край, Хакасия)",
        "rosstat_avg": 70.6,
        "male_stat_avg": 64.9,
        "female_stat_avg": 76.5,
        "score2_mult": 1.38,
        "env_note": "Высокая концентрация добывающей и металлургической промышленности в сочетании с суровой зимой."
    },
    "RU_FE": {
        "name": "Дальний Восток и Забайкалье (Приморский, Хабаровский, Забайкальский край, Амурская обл., Саха, Бурятия)",
        "rosstat_avg": 70.5,
        "male_stat_avg": 65.0,
        "female_stat_avg": 76.3,
        "score2_mult": 1.38,
        "env_note": "Сложные климатические условия и удалённость профильных медцентров."
    },
    "RU_EXTREME": {
        "name": "Республика Тыва / Чукотский АО / Магаданская область",
        "rosstat_avg": 67.5,
        "male_stat_avg": 62.2,
        "female_stat_avg": 73.0,
        "score2_mult": 1.45,
        "env_note": "Экстремальные природно-климатические условия."
    }
}


class UserHabitsInput(BaseModel):
    # === БЛОК 1: РЕГИОН, БАЗОВЫЕ ПАРАМЕТРЫ, ОБРАЗ ЖИЗНИ И ПРОИЗВОДСТВО ===
    country_code: str = Field("RU", description="Код страны проживания")
    ru_region_code: str = Field("RU_AVG", description="Код региона РФ (если country_code == 'RU')")
    age: int = Field(..., ge=18, le=95, description="Возраст в годах")
    gender: Literal["male", "female"] = Field(..., description="Пол: male / female")
    height_cm: float = Field(..., ge=120, le=230, description="Рост в см")
    weight_kg: float = Field(..., ge=35, le=250, description="Вес в кг")
    sleep_hours: float = Field(..., ge=3.0, le=14.0, description="Часов сна в сутки")
    daily_steps: int = Field(..., ge=0, le=50000, description="Шагов в день")
    lifestyle_activity: Literal["sedentary", "light", "moderate", "active_gym"] = Field(
        "light",
        description="Уровень физической активности: sedentary (сидячий), light (легкая бытовая), moderate (физкультура 1-2 раза/нед), active_gym (регулярный спортзал/тренировки 3+ раза/нед)"
    )
    smoking_cigs_per_day: int = Field(..., ge=0, le=60, description="Сигарет в день сейчас (0 = не курит)")
    alcohol_drinks_per_week: int = Field(..., ge=0, le=70, description="Стандартных порций алкоголя в неделю")
    fastfood_times_per_week: int = Field(..., ge=0, le=28, description="Приемов фастфуда и сладкого в неделю")
    stress_level: Literal["low", "medium", "high"] = Field(..., description="Уровень стресса")
    hazardous_work: Literal["no", "moderate", "high"] = Field("no", description="Работа на тяжелом/вредном производстве")

    # === БЛОК 2: ВРЕДНЫЕ ПРИВЫЧКИ В ПРОШЛОМ ===
    past_smoking_years: int = Field(0, ge=0, le=70, description="Сколько лет курил в прошлом")
    past_smoking_cigs_per_day: int = Field(0, ge=0, le=60, description="Сколько сигарет в день курил в прошлом")
    years_since_quitting_smoking: int = Field(0, ge=0, le=70, description="Сколько лет назад бросил курить")
    past_heavy_alcohol: Literal["no", "yes_quit_over_3y_ago", "yes_quit_recently", "unknown"] = Field("no")

    # === БЛОК 3: ХРОНИЧЕСКИЕ ЗАБОЛЕВАНИЯ ===
    has_hypertension: bool = Field(False)
    has_diabetes: bool = Field(False)
    has_cvd_infarction_angina: bool = Field(False)
    has_stroke_tia: bool = Field(False)
    has_atherosclerosis: bool = Field(False)
    has_thrombosis_pe: bool = Field(False)
    has_oncology: bool = Field(False)
    has_rheumatoid_arthritis: bool = Field(False)
    has_urinary_infection: bool = Field(False)
    has_ibd_polyps: bool = Field(False)

    # === БЛОК 4: РЕГУЛЯРНЫЙ ПРИЕМ ПРЕПАРАТОВ ===
    meds_antihypertensive: bool = Field(False)
    meds_statins_lipid: bool = Field(False)
    meds_anticoagulants: bool = Field(False)
    meds_antidiabetic: bool = Field(False)
    meds_antiarrhythmic: bool = Field(False)
    meds_asthma_copd: bool = Field(False)
    meds_glucocorticoids: bool = Field(False)
    meds_daily_nsaids: bool = Field(False)

    # === БЛОК 5: СВОБОДНЫЙ ВВОД ДОПОЛНИТЕЛЬНЫХ ДАННЫХ И РАСПОЗНАННЫЙ ФАЙЛ АНАЛИЗОВ ===
    custom_health_nuance: Optional[str] = Field(None, max_length=1200)
    lab_report_data: Optional[dict] = Field(None)
    