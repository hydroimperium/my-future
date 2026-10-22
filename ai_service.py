import os
import re
import json
import hashlib
import urllib.request
import urllib.parse
from dotenv import load_dotenv
from openai import OpenAI
from schemas import UserHabitsInput

load_dotenv()

NON_AUTHORITATIVE_DOMAINS = (
    "otvet.mail.ru", "dzen.ru", "pikabu.ru", "irecommend.ru", "otzovik.com", "woman.ru", "babyblog.ru"
)

LIFESTYLE_NAMES_RU = {
    "sedentary": "сидячий образ жизни (мало движения, работа сидя без тренировок)",
    "light": "малоподвижный образ жизни (лёгкая бытовая активность без спорта)",
    "moderate": "умеренно активный образ жизни (физкультура и кардио 1–2 раза в неделю)",
    "active_gym": "активный спортивный образ жизни (регулярные тренировки в спортзале / спорт 3+ раза в неделю)"
}

# Детерминированные кэши (чтобы одинаковые запросы давали идентичный результат от итерации к итерации)
_NUANCE_CACHE: dict[str, dict] = {}
_REPORT_CACHE: dict[str, str] = {}


def _build_already_accounted_summary(user_data: UserHabitsInput) -> str:
    items = [
        f"Возраст {user_data.age} лет, пол {user_data.gender}, рост {user_data.height_cm} см, вес {user_data.weight_kg} кг",
        f"Обычные сигареты: {user_data.smoking_cigs_per_day} шт/день, алкоголь: {user_data.alcohol_drinks_per_week} порц/нед",
        f"Образ жизни: {user_data.lifestyle_activity}, шагов: {user_data.daily_steps}/день, вредное производство: {user_data.hazardous_work}"
    ]
    if user_data.has_hypertension:
        items.append("гипертония")
    if user_data.has_diabetes:
        items.append("сахарный диабет")
    if user_data.has_cvd_infarction_angina:
        items.append("ИБС/инфаркт/стенокардия")
    if user_data.has_stroke_tia:
        items.append("инсульт/ТИА")
    if user_data.has_atherosclerosis:
        items.append("атеросклероз")
    if user_data.has_thrombosis_pe:
        items.append("тромбоз/ТЭЛА")
    if user_data.has_oncology:
        items.append("онкология")
    if user_data.has_rheumatoid_arthritis:
        items.append("ревматоидный артрит")
    return "; ".join(items)


def search_authoritative_medical_web(query_text: str) -> tuple[str, list[dict], list[str]]:
    """Поиск строго по актуальным и авторитетным медицинским источникам."""
    search_query = f"{query_text} клинические рекомендации мета-анализ прогноз риск осложнений медицина ВОЗ PubMed"
    snippets = []
    sources = []
    executed_queries = [search_query]

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key and tavily_key != "ПУСТО":
        try:
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=json.dumps({
                    "api_key": tavily_key,
                    "query": search_query,
                    "search_depth": "advanced",
                    "max_results": 5
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("results", []):
                url = item.get("url", "")
                if any(bad in url.lower() for bad in NON_AUTHORITATIVE_DOMAINS):
                    continue
                snippets.append(f"[{item.get('title')}]: {item.get('content')}")
                sources.append({"title": item.get("title", "Медицинский источник"), "uri": url})
            if snippets:
                return "\n".join(snippets[:4]), sources[:3], executed_queries
        except Exception as e:
            print(f"[Предупреждение Tavily Search]: {e}")

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS(timeout=6) as ddgs:
            results = list(ddgs.text(search_query, region="ru-ru", max_results=6))
            for r in results:
                title = r.get("title", "Медицинская публикация")
                body = r.get("body", "")
                href = r.get("href", "")
                if any(bad in href.lower() for bad in NON_AUTHORITATIVE_DOMAINS):
                    continue
                if body:
                    snippets.append(f"[{title}]: {body}")
                    sources.append({"title": title, "uri": href})
        if snippets:
            print(f">>> АВТОРИТЕТНЫЙ ПОИСК (DDGS) УСПЕШЕН: найдено {len(sources)} источников по запросу «{query_text[:40]}» <<<")
            return "\n".join(snippets[:4]), sources[:3], executed_queries
    except Exception as e:
        print(f"[Предупреждение DDGS Search]: {e}")

    try:
        encoded_q = urllib.parse.quote(query_text)
        wiki_url = (
            f"https://ru.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={encoded_q}&utf8=&format=json&srlimit=3"
        )
        req = urllib.request.Request(
            wiki_url,
            headers={"User-Agent": "MyFutureHealthMVP/1.0 (https://localhost:8000)"}
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            wdata = json.loads(resp.read().decode("utf-8"))
        for item in wdata.get("query", {}).get("search", []):
            title = item.get("title", "")
            clean_snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            page_url = f"https://ru.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            snippets.append(f"[{title}]: {clean_snippet}")
            sources.append({"title": f"Мед. энциклопедия: {title}", "uri": page_url})
        if snippets:
            return "\n".join(snippets), sources[:3], executed_queries
    except Exception as e:
        print(f"[Предупреждение Wiki API]: {e}")

    return "Опирайся строго на данные ВОЗ, PubMed, ESC и клинические рекомендации.", [], executed_queries


def _local_clinical_nuance_parser(text: str, user_data: UserHabitsInput) -> dict:
    """
    Детерминированный клинический анализатор текста пользователя (якорь стабильности расчётов).
    Включает распознавание вейпов/электронок/кальянов, инфарктов, инсультов, операций и анализов.
    """
    t = text.lower()
    total_impact = 0.0
    irrev_weights = []
    is_cvd = False
    matched_titles = []
    summaries = []
    actions = []

    # 1. Вейп, электронные сигареты, айкос, кальян, снюс (если не отмечены обычные сигареты)
    if any(w in t for w in ["вейп", "вэйп", "vape", "электрон", "парю", "подик", "hqd", "одноразк", "айкос", "iqos", "glo", "кальян", "снюс"]):
        if not any(neg in t for neg in ["не курю вейп", "бросил вейп"]):
            total_impact -= 2.4 if user_data.smoking_cigs_per_day == 0 else 1.0
            irrev_weights.append(0.22)
            is_cvd = True
            matched_titles.append("Курение вейпа / электронных систем доставки никотина")
            summaries.append(
                "Аэрозоли вейпов и систем нагревания содержат никотин, альдегиды и микрочастицы металлов, вызывающие спазм сосудов, повреждение их внутренней стенки и повышение риска тромбозов."
            )
            actions.append("Полностью отказаться от вейпа и любых никотинсодержащих аэрозолей.")

    # 2. Перенесенный инфаркт или инсульт, указанный в тексте (если не отмечена галочка)
    if any(w in t for w in ["инфаркт", "миокарда", "ибс", "стенокарди"]) and not user_data.has_cvd_infarction_angina:
        total_impact -= 6.5
        irrev_weights.append(0.58)
        is_cvd = True
        matched_titles.append("Перенесённый инфаркт миокарда / ИБС")
        summaries.append("Постинфарктное рубцевание миокарда существенно повышает риск повторных сосудистых событий и требует постоянной кардиопротективной терапии.")
        actions.append("Постоянный контроль у кардиолога, приём антиагрегантов, статинов и контроль давления.")

    if any(w in t for w in ["инсульт", "ишемическая атака", "тиа"]) and not user_data.has_stroke_tia:
        total_impact -= 6.0
        irrev_weights.append(0.58)
        is_cvd = True
        matched_titles.append("Перенесённое нарушение мозгового кровообращения")
        summaries.append("Перенесённый инсульт требует пожизненного контроля артериального давления и свёртываемости крови.")
        actions.append("Регулярное наблюдение у невролога и кардиолога.")

    # 3. Артериальное давление в тексте
    bp_match = re.search(r"(?:давлен\w*|ад)\s*[:=-]?\s*(\d{3})\s*[/\\на-]\s*(\d{2,3})", t)
    if bp_match and not user_data.has_hypertension:
        sbp = int(bp_match.group(1))
        if sbp >= 160:
            total_impact -= 4.2
            irrev_weights.append(0.35)
            is_cvd = True
            matched_titles.append(f"АД {sbp} мм рт. ст.")
            summaries.append("Стойкое повышение систолического давления >=160 мм рт. ст. двукратно повышает риск сосудистых осложнений.")
            actions.append("Обратиться к кардиологу для подбора антигипертензивной терапии.")
        elif sbp >= 140:
            total_impact -= 2.4
            irrev_weights.append(0.30)
            is_cvd = True
            matched_titles.append(f"Повышенное АД ({sbp})")
            summaries.append("Артериальное давление 140–159 мм рт. ст. создает хроническую перегрузку миокарда.")
            actions.append("Контролировать АД утром и вечером и ограничить соль до 5 г/сут.")

    # 4. Тяжелые органные патологии
    if any(w in t for w in ["цирроз", "печеночная недостаточность", "хпн", "диализ", "сердечная недостаточность", "метастаз"]):
        total_impact -= 9.5
        irrev_weights.append(0.78)
        is_cvd = True
        matched_titles.append("Тяжелая органная патология")
        summaries.append("Декомпенсированные хронические заболевания печени, почек или миокарда существенно ограничивают резерв выживаемости.")
        actions.append("Постоянное профильное наблюдение и соблюдение поддерживающего протокола лечения.")

    # 5. Кардиохирургия и сосудистые операции
    if any(w in t for w in ["стент", "шунтирован", "акш", "клапан", "кардиостимулятор", "аневризм", "операция на сосуд", "операция на сердц"]):
        total_impact -= 3.8 if not user_data.has_cvd_infarction_angina else 1.8
        irrev_weights.append(0.60)
        is_cvd = True
        matched_titles.append("Кардиохирургическое вмешательство в анамнезе")
        summaries.append("Перенесенная операция на сердце или сосудах требует пожизненной вторичной профилактики.")
        actions.append("Регулярный контроль липидограммы, ЭхоКГ и прием назначенных препаратов.")

    # 6. Храп и апноэ сна
    if any(w in t for w in ["апноэ", "остановки дыхания", "задыхаюсь во сне"]):
        total_impact -= 2.8
        irrev_weights.append(0.25)
        is_cvd = True
        matched_titles.append("Синдром обструктивного апноэ сна")
        summaries.append("Остановки дыхания во сне вызывают ночную гипоксию и утренние скачки давления.")
        actions.append("Пройти ночную пульсоксиметрию у сомнолога.")
    elif "храп" in t:
        total_impact -= 0.7
        irrev_weights.append(0.25)
        matched_titles.append("Хронический храп")

    # 7. Сахар, холестерин, анемия, аритмия
    if any(w in t for w in ["сахар повышен", "высокий сахар", "предиабет", "преддиабет", "инсулинорезистент"]) and not user_data.has_diabetes:
        total_impact -= 2.0
        irrev_weights.append(0.30)
        is_cvd = True
        matched_titles.append("Нарушение углеводного обмена")
        summaries.append("Повышенная глюкоза натощак ускоряет повреждение мелких сосудов.")
        actions.append("Сдать гликированный гемоглобин (HbA1c) и исключить быстрые сахара.")

    if any(w in t for w in ["холестерин повышен", "высокий холестерин", "бляшк"]) and not user_data.has_atherosclerosis:
        total_impact -= 1.8
        irrev_weights.append(0.30)
        is_cvd = True
        matched_titles.append("Дислипидемия")
        summaries.append("Избыток холестерина низкой плотности ускоряет отложение бляшек в артериях.")
        actions.append("Сдать липидный спектр и выполнить УЗИ сосудов шеи.")

    if any(w in t for w in ["анеми", "низкий гемоглобин", "низкий ферритин"]):
        total_impact -= 1.4
        irrev_weights.append(0.20)
        matched_titles.append("Анемия / дефицит железа")
        summaries.append("Сниженный гемоглобин вызывает тканевую гипоксию и утомляемость.")
        actions.append("Проверить уровень ферритина, сывороточного железа и витамина B12.")

    if any(w in t for w in ["аритми", "фибрилляц", "мерцательн", "тахикарди"]):
        total_impact -= 2.4
        irrev_weights.append(0.45)
        is_cvd = True
        matched_titles.append("Нарушение сердечного ритма")
        summaries.append("Аритмии и тахикардия в покое повышают износ миокарда и тромбоэмболический риск.")
        actions.append("Выполнить суточное мониторирование ЭКГ по Холтеру.")

    if any(w in t for w in ["желчн", "холецистэктоми", "гипотиреоз", "щитовид", "язва", "гастрит", "панкреатит", "астм", "хобл"]):
        total_impact -= 1.3
        irrev_weights.append(0.45)
        matched_titles.append("Хроническая соматическая патология")

    if any(w in t for w in ["у родителей", "у отца", "у мамы", "наследствен", "в семье был"]):
        total_impact -= 1.1
        irrev_weights.append(0.60)
        matched_titles.append("Отягощенный семейный анамнез")

    if total_impact == 0.0 and any(w in t for w in ["спортсмен", "кмс", "плавани", "бегаю", "марафон", "идеальные анализы"]):
        total_impact = 1.2
        irrev_weights.append(0.80)
        matched_titles.append("Высокий кардиореспираторный резерв")

    total_impact = round(max(-15.0, min(1.8, total_impact)), 2)
    avg_irrev = round(sum(irrev_weights) / len(irrev_weights), 2) if irrev_weights else 0.50

    return {
        "impact_years_current": total_impact,
        "irreversible_fraction": avg_irrev,
        "is_cardiovascular": is_cvd,
        "title": ", ".join(matched_titles[:2]) if matched_titles else "Дополнительные сведения о здоровье",
        "clinical_summary": " ".join(summaries[:2]) if summaries else f"Учтены индивидуальные сведения: «{text[:120]}».",
        "recommended_action": actions[0] if actions else "Обсудить указанные особенности с профильным врачом."
    }


def analyze_custom_nuance_with_search(user_data: UserHabitsInput) -> dict | None:
    """
    Оценивает текстовое поле пользователя с SHA-256 кэшированием и жёсткой привязкой к детерминированному сканеру,
    чтобы при повторных нажатиях кнопки результаты не плавали.
    """
    nuance_text = (user_data.custom_health_nuance or "").strip()
    if not nuance_text or len(nuance_text) < 3:
        return None

    already_accounted = _build_already_accounted_summary(user_data)
    cache_key = hashlib.sha256(f"{nuance_text.lower()}|{already_accounted}".encode("utf-8")).hexdigest()
    if cache_key in _NUANCE_CACHE:
        return json.loads(json.dumps(_NUANCE_CACHE[cache_key]))

    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
    primary_model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite").strip()

    web_context, sources, search_queries = search_authoritative_medical_web(nuance_text)
    local_parsed = _local_clinical_nuance_parser(nuance_text, user_data)

    if not api_key or api_key == "ПУСТО":
        res = {
            "raw_text": nuance_text,
            **local_parsed,
            "search_queries": search_queries,
            "sources": sources
        }
        _NUANCE_CACHE[cache_key] = json.loads(json.dumps(res))
        return res

    prompt = f"""
Ты — врач-актуарий и специалист по доказательной медицине. Пользователь (возраст {user_data.age} лет, пол {user_data.gender}) указал дополнительные сведения о здоровье:
«{nuance_text}»

Факторы, которые УЖЕ УЧТЕНЫ в анкете галочками (НЕ дублируй их!):
{already_accounted}

ДАННЫЕ ИЗ ПОИСКА ПО МЕДИЦИНСКИМ ИСТОЧНИКАМ:
{web_context}

КРИТИЧЕСКОЕ ТРЕБОВАНИЕ К ДОСТОВЕРНОСТИ:
Опирайся СТРОГО на актуальную, рецензируемую и достоверную информацию из авторитетных источников (клинические рекомендации Минздрава РФ, ВОЗ / WHO, PubMed, Cochrane, ESC, AHA, The Lancet, NEJM).

ШКАЛА КАЛИБРОВКИ ВКЛАДА В ГОДАХ (`impact_years_current`):
- От -8.0 до -15.0 лет: тяжелые органные поражения (цирроз печени, ХПН 3б-5 ст., сердечная недостаточность, множественные инфаркты, активная онкология).
- От -3.5 до -7.5 лет: перенесенный инфаркт/инсульт (если не отмечен в анкете), шунтирование/стентирование/протезирование клапана, давление >=160/100, нелеченое апноэ сна.
- От -1.5 до -3.2 лет: курение вейпа/электронных сигарет/айкоса/кальяна, давление 140-155, предиабет, высокий холестерин, анемия, астма, удаленный желчный пузырь.
- От -0.3 до -1.2 года: легкие функциональные расстройства.
- 0.0 лет: не влияет на продолжительность жизни или уже отмечено в анкете.
- От +0.5 до +1.8 лет: документально подтвержденный высокий уровень выносливости.

Верни СТРОГО валидный JSON-объект:
{{
  "impact_years_current": -3.5,
  "irreversible_fraction": 0.45,
  "is_cardiovascular": true,
  "title": "Краткое медицинское название (до 6 слов)",
  "clinical_summary": "Научное объяснение на 2 предложения строго по данным авторитетных исследований.",
  "recommended_action": "Конкретная клиническая рекомендация (1 предложение)."
}}
"""

    candidate_models = [primary_model]
    for backup_m in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        if backup_m not in candidate_models:
            candidate_models.append(backup_m)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=18.0)

    for model_to_try in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_to_try,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты — эксперт доказательной медицины. Используй только авторитетные рецензируемые источники. Отвечай строго чистым JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            raw_reply = response.choices[0].message.content.strip()
            json_match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
            if not json_match:
                continue

            parsed = json.loads(json_match.group(0))
            llm_impact = float(parsed.get("impact_years_current", local_parsed["impact_years_current"]))
            det_impact = local_parsed["impact_years_current"]

            # Якорение: если детерминированный парсер тоже распознал состояние, усредняем (60% детерминированный + 40% LLM) для 100% объективности
            if abs(det_impact) >= 0.8 and abs(llm_impact) >= 0.5:
                final_impact = (det_impact * 0.6) + (llm_impact * 0.4)
            elif abs(llm_impact) < 0.2 and abs(det_impact) >= 1.0:
                final_impact = det_impact
            else:
                final_impact = llm_impact

            impact_yrs = round(max(-15.0, min(1.8, final_impact)), 2)
            irrev_frac = round(max(0.15, min(0.90, float(parsed.get("irreversible_fraction", local_parsed["irreversible_fraction"])))), 2)

            res = {
                "raw_text": nuance_text,
                "impact_years_current": impact_yrs,
                "irreversible_fraction": irrev_frac,
                "is_cardiovascular": bool(parsed.get("is_cardiovascular", local_parsed["is_cardiovascular"])),
                "title": str(parsed.get("title", local_parsed["title"])),
                "clinical_summary": str(parsed.get("clinical_summary", local_parsed["clinical_summary"])),
                "recommended_action": str(parsed.get("recommended_action", local_parsed["recommended_action"])),
                "search_queries": search_queries,
                "sources": sources
            }
            _NUANCE_CACHE[cache_key] = json.loads(json.dumps(res))
            return res
        except Exception as e:
            print(f"[Попытка оценки нюанса ({model_to_try}) не удалась]: {e}")

    res = {
        "raw_text": nuance_text,
        **local_parsed,
        "search_queries": search_queries,
        "sources": sources
    }
    _NUANCE_CACHE[cache_key] = json.loads(json.dumps(res))
    return res


def _verify_and_sanitize_ai_report(text: str, metrics: dict) -> str:
    """
    Пост-верификатор текста отчёта ИИ:
    1) Проверяет, не похвалил ли ИИ пользователя за отказ от курения/алкоголя при наличии скрытого вейпа/алкоголя в тексте.
    2) Удаляет случайные англоязычные медицинские аббревиатуры из основного текста.
    """
    eff_flags = metrics.get("effective_flags", {})
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned_sentences = []

    for s in sentences:
        s_low = s.lower()
        # Если у пользователя вейп/айкос/кальян при 0 сигарет, вырезаем ошибочные похвалы за «отсутствие курения»
        if eff_flags.get("has_hidden_nicotine"):
            if any(p in s_low for p in ("отсутствие курения", "отказ от курения", "жизнь без табака", "не курите", "отказались от сигарет")) and not any(
                v in s_low for v in ("вейп", "электрон", "никотин", "кальян", "айкос", "однако", "но ")
            ):
                continue
        # Если скрытый алкоголь при 0 порций в анкете, вырезаем ошибочные похвалы за «трезвость»
        if eff_flags.get("has_hidden_alcohol"):
            if any(p in s_low for p in ("трезвость", "полный отказ от алкоголя", "не употребляете алкоголь")):
                continue
        cleaned_sentences.append(s)

    verified_text = " ".join(cleaned_sentences)
    # Убираем запрещенные аббревиатуры, если модель случайно их вставила
    verified_text = re.sub(r"\b(?:ESC\s*SCORE2|SCORE2|AHA\s*Life's\s*Essential\s*8|AHA\s*LE8|Hazard\s*Ratio|CKD-EPI)\b", "", verified_text)
    verified_text = re.sub(r"  +", " ", verified_text)
    return verified_text.strip()


def generate_future_explanation(user_data: UserHabitsInput, metrics: dict, who_facts: list[dict]) -> str:
    """
    Генерирует верифицированный, непротиворечивый и стабильный аналитический прогноз.
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
    primary_model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite").strip()

    rem_curr = metrics["remaining_years_current"]
    rem_pot = metrics["remaining_years_potential"]
    gain_years = metrics["remaining_years_gain"]
    is_russia = metrics.get("is_russia", True)
    savings_formatted = f"{metrics['ten_year_savings_rub']:,}".replace(",", " ")
    med_saved = metrics.get("medication_saved_years", 0.0)
    alerts = metrics.get("medical_alerts", [])
    contradictions_notes = metrics.get("contradictions_notes", [])
    eff_flags = metrics.get("effective_flags", {})
    lab_info = metrics.get("lab_summary_info")
    custom_info = metrics.get("custom_nuance_info")
    factors_years = metrics.get("factors_years", {})
    pace_curr = metrics.get("pace_of_aging_current", 1.0)
    pace_pot = metrics.get("pace_of_aging_potential", 0.85)
    cvd_30_curr = metrics.get("cvd_30yr_risk_percent", 10.0)
    cvd_30_pot = metrics.get("cvd_30yr_risk_potential_percent", 5.0)

    gender_label = metrics.get("gender_label", "жителей")
    stat_avg = metrics.get("official_stat_avg", 73.4)
    diff_vs_stat = round(metrics["predicted_expectancy"] - stat_avg, 1)
    diff_vs_stat_str = f"на +{diff_vs_stat} г. выше" if diff_vs_stat >= 0 else f"на {abs(diff_vs_stat)} г. ниже"

    potential_line = (
        f"• Прогноз при следовании рекомендациям: {metrics['potential_expectancy']} лет (останется около {rem_pot} лет, можно выиграть +{gain_years} лет жизни)."
        if gain_years >= 0.4
        else f"• Оценка потенциала: {metrics['potential_expectancy']} лет (вы уже реализуете почти максимум своего здорового резерва!)."
    )

    summary_header_paragraph = (
        f"📊 КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ ПРОГНОЗА ({metrics['country_name']}):\n"
        f"• Ваш возраст сейчас: {user_data.age} лет (по состоянию организма — около {metrics['biological_age']} лет).\n"
        f"• Ожидаемая продолжительность жизни (при текущих данных): {metrics['predicted_expectancy']} лет — осталось около {rem_curr} лет ({diff_vs_stat_str} средней статистики {gender_label} в регионе: {stat_avg} г.).\n"
        f"{potential_line}\n"
        f"• Темп старения организма: {pace_curr} г. износа за 1 календарный год (при коррекции привычек замедлится до {pace_pot} г./год)."
    )

    # Положительные факторы формируются СТРОГО с учетом перекрёстной проверки противоречий
    positive_notes = []
    if factors_years.get("smoking", 0) > 0 and not eff_flags.get("has_hidden_nicotine"):
        positive_notes.append(f"отсутствие курения (+{factors_years['smoking']} г. к средней статистике)")
    if factors_years.get("alcohol", 0) > 0 and not eff_flags.get("has_hidden_alcohol"):
        positive_notes.append(f"трезвость / отказ от алкоголя (+{factors_years['alcohol']} г.)")
    if factors_years.get("activity", 0) > 0 and not eff_flags.get("has_activity_limitation"):
        positive_notes.append(f"высокая физическая активность и тренировки (+{factors_years['activity']} г.)")
    if factors_years.get("bmi", 0) > 0:
        positive_notes.append(f"нормальный вес (+{factors_years['bmi']} г.)")
    if factors_years.get("nutrition", 0) > 0 and not eff_flags.get("has_hidden_poor_diet"):
        positive_notes.append(f"здоровый рацион без фастфуда (+{factors_years['nutrition']} г.)")

    positive_summary_str = ", ".join(positive_notes) if positive_notes else "сейчас перекрываются факторами риска"

    savings_sentence_fallback = ""
    savings_prompt_instruction = "Не упоминай финансовую экономию в рублях."
    if is_russia and metrics["ten_year_savings_rub"] > 0:
        savings_sentence_fallback = (
            f" Отказ от вредных привычек также позволит условно сберечь около {savings_formatted} ₽ за 10 лет."
        )
        savings_prompt_instruction = (
            f"Упомяни, что отказ от вредных привычек позволит ориентировочно сохранить около {savings_formatted} ₽ за 10 лет."
        )

    lab_prompt_block = "Файл с анализами не прикреплен."
    lab_fallback_paragraph = ""
    if lab_info:
        lab_y = lab_info.get("impact_years", 0.0)
        lab_sign = f"+{lab_y}" if lab_y > 0 else f"{lab_y}"
        abn_list = ", ".join(lab_info.get("detected_abnormalities", [])) or "все показатели в пределах нормы"
        norm_list = ", ".join(lab_info.get("normal_markers", [])) or "не указаны"
        lab_prompt_block = (
            f"Прикреплен файл «{lab_info['filename']}». Точное влияние анализов на итоговый прогноз: {lab_sign} лет. "
            f"Отклонения: {abn_list}. Показатели в норме: {norm_list}. "
            f"Клиническое резюме: {lab_info['clinical_summary']} Рекомендация по анализам: {lab_info['recommended_action']}"
        )
        lab_fallback_paragraph = (
            f"\n\n🧪 Разбор ваших прикреплённых анализов (влияние на прогноз: {lab_sign} г.):\n"
            f"{lab_info['clinical_summary']} Рекомендуемый шаг по анализам: {lab_info['recommended_action']}"
        )

    custom_context_prompt = "Не указаны."
    custom_fallback_paragraph = ""
    if custom_info:
        impact_y = custom_info.get("impact_years", 0.0)
        sign_str = f"+{impact_y}" if impact_y > 0 else f"{impact_y}"
        custom_context_prompt = (
            f"Пользователь дополнительно указал в тексте: «{custom_info['raw_text']}» "
            f"(точное влияние на итоговый прогноз: {sign_str} лет). Суть: {custom_info['clinical_summary']} "
            f"Рекомендация: {custom_info['recommended_action']}"
        )
        custom_fallback_paragraph = (
            f"\n\n🔍 Учёт ваших дополнительных сведений («{custom_info['title']}», влияние: {sign_str} г.):\n"
            f"{custom_info['clinical_summary']} Рекомендуемый шаг: {custom_info['recommended_action']}"
        )

    meds_note = ""
    if med_saved > 0:
        meds_note = (
            f" Регулярный приём назначенных вам препаратов уже защищает сосуды и сохраняет около +{med_saved} лет жизни."
        )

    alerts_block = ""
    if alerts:
        alerts_block = "\n\nОбратите внимание:\n" + "\n".join([f"⚠️ {a}" for a in alerts])

    organ_systems = metrics.get("organ_systems", [])
    systems_summary_parts = []
    worst_system = None
    min_score = 999
    for s in organ_systems:
        t = s.get("title", s.get("id"))
        cs = s.get("current_score", 0)
        ps = s.get("potential_score", 0)
        lbl = s.get("status_label", "")
        systems_summary_parts.append(f"{t}: {cs}% (потенциал до {ps}%, статус: {lbl})")
        if cs < min_score:
            min_score = cs
            worst_system = t
    systems_summary_str = "; ".join(systems_summary_parts) if systems_summary_parts else "Данные рассчитываются"
    worst_system_hint = f"Самая уязвимая система сейчас — «{worst_system}» ({min_score}% ресурса)." if worst_system else ""

    pos_fallback_note = f" Ваши сильные стороны, которые уже прибавляют годы к прогнозу: {positive_summary_str}." if positive_notes else ""

    fallback_body = (
        f"Сейчас общий ресурс вашего сердца и сосудов оценивается на {metrics['aha_le8_score']} из 100 баллов, "
        f"темп биологического старения — {pace_curr} г./год (при коррекции замедлится до {pace_pot} г./год), "
        f"а риск серьёзных сосудистых осложнений в ближайшие 10 лет составляет около {metrics['cvd_10yr_risk_percent']}% "
        f"(на 30 лет: {cvd_30_curr}% с возможностью снижения до {cvd_30_pot}%). {worst_system_hint}{pos_fallback_note} "
        f"При корректировке управляемых факторов ваш запас предстоящих лет составит до {rem_pot} лет.{savings_sentence_fallback}{meds_note}"
        f"{lab_fallback_paragraph}"
        f"{custom_fallback_paragraph}"
        f"{alerts_block}\n\n"
        f"Рекомендуемый фокус внимания: {who_facts[0]['action']}"
    )

    fallback_full = f"{summary_header_paragraph}\n\n{fallback_body}"

    if not api_key or api_key == "ПУСТО":
        return fallback_full

    # Проверяем кэш финального текста (для 100% повторяемости при повторном нажатии на ту же анкету)
    report_cache_key = hashlib.sha256(
        json.dumps({
            "u": user_data.model_dump(exclude={"lab_report_data"}),
            "m": {
                "pred": metrics["predicted_expectancy"],
                "pot": metrics["potential_expectancy"],
                "bio": metrics["biological_age"],
                "pace": pace_curr,
                "cvd30": cvd_30_curr,
                "lab": lab_info.get("impact_years") if lab_info else None,
                "cust": custom_info.get("impact_years") if custom_info else None,
            }
        }, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    if report_cache_key in _REPORT_CACHE:
        return _REPORT_CACHE[report_cache_key]

    facts_simple = "\n".join(
        [f"- «{f['title']}» (вклад в прогноз: {'+' if f['impact_years'] > 0 else ''}{f['impact_years']} лет): {f['action']}" for f in who_facts]
    )
    lifestyle_desc = LIFESTYLE_NAMES_RU.get(user_data.lifestyle_activity, user_data.lifestyle_activity)
    contradictions_block = (
        "\n".join([f"- ВНИМАНИЕ: {c}" for c in contradictions_notes])
        if contradictions_notes
        else "- Противоречий между полями анкеты и текстом не обнаружено."
    )

    prompt = f"""
Ты — прагматичный, объективный и доброжелательный консультант по здоровью сервиса «Моё будущее».
Напиши честный, логически непротиворечивый разбор прогноза для пользователя во втором лице («У вас», «Вы можете»).

ПРАВИЛА САМОВЕРИФИКАЦИИ И ОТСУТСТВИЯ ПРОТИВОРЕЧИЙ (ПРОВЕРЬ ПЕРЕД ВЫВОДОМ!):
{contradictions_block}
- Если в дополнительных сведениях указаны вейп, электронные сигареты, айкос или кальян — НИ В КОЕМ СЛУЧАЕ не хвали пользователя за отсутствие курения, даже если в поле обычных сигарет стоит 0!
- Если в тексте и в анализах есть связанные состояния (например, вейп + инфаркт или лишний вес + высокий сахар), логично объединяй их объяснение, показывая их взаимное влияние на сосуды и биологический возраст ({metrics['biological_age']} лет).
- Называй ТОЧНО те цифры вклада анализов и дополнительных сведений, которые переданы ниже (не выдумывай свои годы).
- Пиши простым русским языком БЕЗ научных аббревиатур и английских названий шкал (НЕ пиши: ВОЗ, ESC, SCORE2, AHA, Life's Essential 8, NEJM, Lancet, HR, Hazard Ratio, CKD-EPI, ERFC, ИМТ).

ВСЕ ДАННЫЕ ПОЛЬЗОВАТЕЛЯ:
1. Регион и статистика: {metrics['country_name']}. Средняя продолжительность жизни {gender_label} в регионе: {stat_avg} лет. Прогноз пользователя: {metrics['predicted_expectancy']} лет ({diff_vs_stat_str} среднего по региону, осталось прожить около {rem_curr} лет). Потенциал: {metrics['potential_expectancy']} лет (останется около {rem_pot} лет, резерв +{gain_years} лет). Возраст по состоянию организма: {metrics['biological_age']} лет.
2. Подтвержденные положительные привычки: {positive_summary_str}.
3. Образ жизни и физическая активность: {lifestyle_desc}, шагов в день: {user_data.daily_steps}.
4. Сердце, сосуды и системы органов: ресурс сердца {metrics['aha_le8_score']} из 100 баллов. Темп старения организма: {pace_curr} г. износа за год (при коррекции замедлится до {pace_pot} г./год). Риск сосудистых осложнений за 10 лет: сейчас {metrics['cvd_10yr_risk_percent']}% -> потенциал {metrics['cvd_10yr_risk_potential_percent']}%; за 30 лет: {cvd_30_curr}% -> потенциал {cvd_30_pot}%. Стадия CKM: {metrics.get('ckm_label', '')}. Профиль 4 систем организма: {systems_summary_str}. {worst_system_hint} Защита от принимаемых лекарств: +{med_saved} лет. {savings_prompt_instruction}
5. ДАННЫЕ ИЗ ПРИКРЕПЛЕННОГО ФАЙЛА АНАЛИЗОВ: {lab_prompt_block}
6. ДОПОЛНИТЕЛЬНЫЕ ТЕКСТОВЫЕ СВЕДЕНИЯ: {custom_context_prompt}
7. Детализация вклада факторов в годах:
{facts_simple}

СТРУКТУРА ОТВЕТА (четко по абзацам, без приветствий):
- Абзац 1 (Общий ресурс, возраст организма и сравнение со статистикой): объясни текущий прогноз ({metrics['predicted_expectancy']} лет, осталось около {rem_curr} лет, возраст по состоянию организма {metrics['biological_age']} лет, оценка сердца {metrics['aha_le8_score']} из 100) и покажи, как выполнение рекомендаций увеличит запас до {rem_pot} лет (+{gain_years} лет), замедлит темп старения с {pace_curr} до {pace_pot} г./год и снизит риски сосудистых осложнений (на 10 лет: с {metrics['cvd_10yr_risk_percent']}% до {metrics['cvd_10yr_risk_potential_percent']}%, на 30 лет: с {cvd_30_curr}% до {cvd_30_pot}%). {savings_prompt_instruction}
- Абзац 2 (Комплексный разбор систем организма, привычек и диагнозов): разбери факторы риска и положительные привычки ({positive_summary_str}), обязательно упомяни наиболее нагруженную систему организма ({worst_system_hint}) и объясни понятные пути её укрепления, связывая привычки из анкеты и дополнительные сведения из текста (если они указаны).
- Абзац 3 (Разбор анализов — пиши его, если прикреплен файл анализов): понятно распиши результаты загруженных анализов, укажи их точное влияние на прогноз и объясни, как они отразились на биологическом возрасте и сосудах.
- Абзац 4 (План действий): дай 2–3 конкретных, практичных шага на ближайшие две недели.
"""

    candidate_models = [primary_model]
    for backup_m in ["gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        if backup_m not in candidate_models:
            candidate_models.append(backup_m)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=18.0)

    for model_to_try in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_to_try,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты — объективный врач-превентолог. Перед выдачей ответа всегда проверяй текст на отсутствие логических противоречий. Пиши простым русским языком без научных аббревиатур."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            llm_body = response.choices[0].message.content.strip()
            verified_body = _verify_and_sanitize_ai_report(llm_body, metrics)
            full_report = f"{summary_header_paragraph}\n\n{verified_body}"
            _REPORT_CACHE[report_cache_key] = full_report
            print(f">>> УСПЕХ: Верифицированный прогноз сгенерирован через LLM API (модель: {model_to_try}) <<<")
            return full_report
        except Exception as e:
            print(f"[Попытка LLM ({model_to_try}) не удалась]: {e}")

    return fallback_full