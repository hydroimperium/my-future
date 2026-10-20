from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from schemas import UserHabitsInput, COUNTRY_PROFILES, RU_REGION_PROFILES
from calculator import calculate_prediction
from database import init_db, save_prediction, get_recent_predictions
from knowledge_base import retrieve_relevant_facts
from ai_service import analyze_custom_nuance_with_search, generate_future_explanation
from lab_parser import extract_text_from_file

app = FastAPI(
    title="Моё будущее — AI-предиктор здоровья и долголетия",
    description="API с поддержкой загрузки лабораторных анализов (.pdf, .docx), регионов РФ (Росстат) и авторитетного интернет-поиска.",
    version="0.8.0"
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
    return {"status": "ok", "service": "Моё будущее MVP v0.8.0"}


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

    # 5. Сохранение в базу данных SQLite
    record_id = save_prediction(user_data, metrics, ai_explanation)

    return {
        "id": record_id,
        "metrics": metrics,
        "who_facts": who_facts,
        "ai_explanation": ai_explanation
    }


@app.get("/api/history")
def read_history():
    return {"history": get_recent_predictions(limit=10)}
