from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from schemas import UserHabitsInput, COUNTRY_PROFILES, RU_REGION_PROFILES
from calculator import calculate_prediction
from database import init_db, save_prediction, get_recent_predictions, get_portrait_prompt
from knowledge_base import retrieve_relevant_facts
from ai_service import analyze_custom_nuance_with_search, generate_future_explanation
from lab_parser import extract_text_from_file
from portrait_service import build_full_portrait_prompt, generate_portrait_image, PORTRAITS_DIR

app = FastAPI(
    title="Моё будущее — AI-предиктор здоровья и долголетия",
    description="API с поддержкой загрузки лабораторных анализов (.pdf, .docx), регионов РФ (Росстат) и авторитетного интернет-поиска.",
    version="0.9.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

FRONTEND_FILE = Path(__file__).parent / "index.html"


@app.get("/")
def serve_ui():
    """Отдает веб-интерфейс анкеты по адресу http://127.0.0.1:8000/"""
    return FileResponse(FRONTEND_FILE)


@app.get("/api/locations")
def get_locations():
    return {
        "countries": COUNTRY_PROFILES,
        "ru_regions": RU_REGION_PROFILES
    }


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Моё будущее MVP v0.9.0"}


@app.post("/api/parse-labs")
async def upload_and_parse_labs(file: UploadFile = File(...)):
    """
    Проверяет и загружает файл с анализами (.pdf, .docx, .txt).
    Не выводит разбор сразу — сохраняет извлеченный текст для полного расчёта при нажатии «Рассчитать прогноз».
    """
    filename = file.filename or "labs.txt"
    ext = filename.lower().split(".")[-1]
    if ext not in ("pdf", "docx", "doc", "txt", "csv"):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только файлы форматов PDF (.pdf), Word (.docx) и текстовые (.txt)"
        )

    file_bytes = await file.read()
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Размер файла не должен превышать 15 МБ")

    raw_text = extract_text_from_file(filename, file_bytes)
    if not raw_text or len(raw_text.strip()) < 5:
        raise HTTPException(
            status_code=400,
            detail="Не удалось прочитать текст из файла (возможно, документ пуст или является сканом без текста)"
        )

    return {
        "filename": filename,
        "status": "ok",
        "raw_text": raw_text
    }


@app.post("/api/predict")
def create_prediction(user_data: UserHabitsInput):
    # 1. Поиск в авторитетных источниках по текстовому полю дополнительных сведений
    custom_nuance_analysis = analyze_custom_nuance_with_search(user_data)

    # 2. Полный расчёт (включая разбор загруженного файла анализов с учётом пола, возраста и интернет-поиска редких маркеров)
    metrics = calculate_prediction(user_data, custom_nuance_analysis=custom_nuance_analysis)

    # 3. Подбор научных фактов для блока «Доказательная база»
    who_facts = retrieve_relevant_facts(user_data, metrics)

    # 4. Генерация структурированного текста прогноза
    ai_explanation = generate_future_explanation(user_data, metrics, who_facts)

    # 5. Генерация промпта для портрета из будущего (сохраняется в БД для последующей генерации фото)
    portrait_prompt = build_full_portrait_prompt(metrics, user_data)

    # 6. Сохранение в базу данных SQLite (включая portrait_prompt)
    record_id = save_prediction(user_data, metrics, ai_explanation, portrait_prompt=portrait_prompt)

    return {
        "id": record_id,
        "metrics": metrics,
        "who_facts": who_facts,
        "ai_explanation": ai_explanation
    }


@app.post("/api/generate-portrait/{prediction_id}")
async def generate_portrait(prediction_id: int, file: UploadFile = File(...)):
    """
    Генерирует «портрет из будущего»:
    принимает фото пользователя, загружает сохранённый промпт из БД,
    вызывает Image Generation API и возвращает оба изображения.
    """
    # 1. Проверяем наличие промпта для этого расчёта
    portrait_prompt = get_portrait_prompt(prediction_id)
    if not portrait_prompt:
        raise HTTPException(
            status_code=404,
            detail="Прогноз не найден или промпт портрета не был сгенерирован. Сначала выполните расчёт прогноза."
        )

    # 2. Валидация загруженного файла
    filename = file.filename or "photo.jpg"
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("jpg", "jpeg", "png", "webp"):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только изображения форматов JPG, PNG и WebP"
        )

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Размер фото не должен превышать 10 МБ")
    if len(file_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Файл слишком мал — загрузите качественное фото")

    # Определяем MIME-тип
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    # 3. Сохраняем оригинальное фото
    original_path = PORTRAITS_DIR / f"{prediction_id}_original.{ext}"
    original_path.write_bytes(file_bytes)

    # 4. Вызываем генерацию портрета
    try:
        generated_bytes = generate_portrait_image(file_bytes, mime_type, portrait_prompt)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка генерации нейросетью: {str(e)}"
        )

    if not generated_bytes:
        raise HTTPException(
            status_code=502,
            detail="Не удалось получить изображение от нейросети. Проверьте настройки API-ключа."
        )

    # 5. Сохраняем сгенерированное изображение (определяем формат по байтам)
    is_jpeg = generated_bytes.startswith(b"\xff\xd8")
    future_ext = "jpg" if is_jpeg else "png"
    future_filename = f"{prediction_id}_future.{future_ext}"
    future_path = PORTRAITS_DIR / future_filename
    future_path.write_bytes(generated_bytes)

    return {
        "status": "ok",
        "original_url": f"/api/portraits/{prediction_id}_original.{ext}",
        "future_url": f"/api/portraits/{future_filename}"
    }


@app.get("/api/portraits/{filename}")
def serve_portrait(filename: str):
    """Отдаёт сохранённые файлы портретов (оригинал и сгенерированный)."""
    # Защита от path traversal
    safe_name = Path(filename).name
    file_path = PORTRAITS_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Файл портрета не найден")

    # Определяем content-type
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    media_types = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(file_path, media_type=media_type)


@app.get("/api/history")
def read_history():
    return {"history": get_recent_predictions(limit=10)}
