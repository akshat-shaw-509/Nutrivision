"""Authenticated API shared by the web and mobile clients."""

import os
import json
import sqlite3
import uuid
import io
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, Annotated
from urllib.parse import urlparse

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Response,
    Request,
    UploadFile,
    File,
    Form,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from data.nutrition_data_store import NutritionDataStore
from data.identity import normalize_phone
from backend.providers import AIService, FoodService, VerificationService, ProviderError


class Input(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, str_strip_whitespace=True
    )


class Login(Input):
    phone: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("phone")
    @classmethod
    def canonical_phone(cls, value):
        return normalize_phone(value)

    # Passwords are intentionally not stripped.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class Register(Login):
    timezone: str = Field(default="UTC", max_length=100)

    @field_validator("timezone")
    @classmethod
    def valid_zone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("Choose a valid IANA timezone.") from None
        return value

    full_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=10, max_length=128)
    primary_goal: Literal[
        "fat_loss",
        "muscle_gain",
        "maintenance",
        "athletic_performance",
        "metabolic_health",
        "general_wellness",
    ] = "maintenance"
    daily_calorie_target: float = Field(default=2000, ge=800, le=10000)
    daily_protein_target_g: float = Field(default=140, ge=0, le=500)

    @field_validator("password")
    @classmethod
    def strong_password(cls, value):
        if len(value.strip()) < 10:
            raise ValueError(
                "Use at least 10 characters, excluding surrounding spaces."
            )
        return value


class Profile(Input):
    full_name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=18, le=120)
    sex: Literal["male", "female", "other"]
    height_cm: float = Field(ge=80, le=250)
    current_weight_kg: float = Field(ge=25, le=400)
    target_weight_kg: float = Field(ge=25, le=400)
    activity_level: Literal[
        "sedentary",
        "lightly_active",
        "moderately_active",
        "very_active",
        "extra_active",
    ]
    primary_goal: Literal[
        "fat_loss",
        "muscle_gain",
        "maintenance",
        "athletic_performance",
        "metabolic_health",
        "general_wellness",
    ]
    dietary_preference: Literal[
        "omnivore",
        "vegetarian",
        "vegan",
        "pescatarian",
        "keto",
        "paleo",
        "mediterranean",
        "low_carb",
        "custom",
    ]
    allergies: str = Field(default="", max_length=500)
    medical_conditions: str = Field(default="", max_length=500)
    timezone: str = Field(max_length=100)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("Choose a valid IANA timezone.") from None
        return value


class Reauthenticate(Input):
    current_password: str = Field(min_length=1, max_length=128)
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class ChangePassword(Reauthenticate):
    new_password: str = Field(min_length=10, max_length=128)

    @field_validator("new_password")
    @classmethod
    def strong(cls, value):
        return Register.strong_password(value)


class Recovery(Login):
    recovery_code: str = Field(min_length=6, max_length=100)

    @field_validator("password")
    @classmethod
    def strong(cls, value):
        return Register.strong_password(value)


class PhoneRequest(Input):
    phone: str = Field(max_length=100)

    @field_validator("phone")
    @classmethod
    def canonical(cls, value):
        return normalize_phone(value)


class ChallengeCheck(Input):
    challenge_id: str = Field(min_length=20, max_length=100)
    code: str = Field(pattern=r"^\d{4,10}$")


class SMSReset(ChallengeCheck):
    new_password: str = Field(min_length=10, max_length=128)

    @field_validator("new_password")
    @classmethod
    def strong(cls, value):
        return Register.strong_password(value)


class Targets(Input):
    daily_calorie_target: float = Field(ge=800, le=10000)
    daily_protein_target_g: float = Field(ge=0, le=500)
    daily_carbs_target_g: float = Field(ge=0, le=1500)
    daily_fat_target_g: float = Field(ge=0, le=500)
    daily_water_target_ml: float = Field(ge=250, le=10000)


class Meal(Input):
    food_name: str = Field(min_length=1, max_length=200)
    meal_type: Literal[
        "breakfast",
        "lunch",
        "dinner",
        "morning_snack",
        "afternoon_snack",
        "evening_snack",
        "pre_workout",
        "post_workout",
    ] = "lunch"
    log_date: date | None = None
    calories: float = Field(ge=0, le=10000)
    protein_g: float = Field(default=0, ge=0, le=1000)
    carbs_g: float = Field(default=0, ge=0, le=2000)
    fat_g: float = Field(default=0, ge=0, le=1000)
    fiber_g: float = Field(default=0, ge=0, le=500)

    nutrition_source: str | None = Field(default=None, max_length=150)
    source_url: str | None = Field(default=None, max_length=500)
    serving_g: float | None = Field(default=None, gt=0, le=5000)
    micronutrients: dict[
        Literal[
            "potassium_mg",
            "calcium_mg",
            "iron_mg",
            "magnesium_mg",
            "vitamin_c_mg",
            "vitamin_d_mcg",
            "vitamin_b12_mcg",
        ],
        Annotated[float, Field(ge=0, le=50000)],
    ] = Field(default_factory=dict)

    @field_validator("source_url")
    @classmethod
    def safe_source(cls, value):
        if value is not None and urlparse(value).scheme not in ("http", "https"):
            raise ValueError("Source links must use HTTP or HTTPS.")
        return value


class Water(Input):
    amount_ml: float = Field(gt=0, le=5000)
    log_date: date | None = None


class Biometrics(Input):
    log_date: date | None = None
    weight_kg: float | None = Field(default=None, gt=0, le=700)
    body_fat_pct: float | None = Field(default=None, ge=0, le=100)
    energy_level: int | None = Field(default=None, ge=1, le=10)
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    sleep_quality: int | None = Field(default=None, ge=1, le=10)
    digestive_symptoms: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)


class Chat(Input):
    message: str = Field(min_length=1, max_length=4000)


class Oracle(Input):
    query: str = Field(min_length=1, max_length=4000)
    category: str = Field(default="general", max_length=100)


def create_app(store: NutritionDataStore):
    app = FastAPI(title="NutriAI", version="3.0.0")
    app.state.store = store
    if os.getenv("NUTRIAI_LOG_REQUESTS") == "true":
        logger = logging.getLogger("nutriai.requests")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.propagate = False

    ai = AIService()
    sms = VerificationService()
    foods = FoodService(store)
    app.state.ai = ai
    app.state.sms = sms
    app.state.foods = foods
    if os.getenv("NUTRIAI_REQUIRE_VERIFIED_PHONE") == "true" and not sms.enabled:
        raise RuntimeError(
            "Phone verification is required but no SMS provider is configured."
        )

    @app.exception_handler(sqlite3.Error)
    async def database_error(request, exc):
        logging.getLogger("nutriai.database").error(
            "Database operation failed: %s", type(exc).__name__
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "The database is temporarily unavailable. Please try again."
            },
        )

    @app.exception_handler(ProviderError)
    async def provider_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    def today(user):
        return datetime.now(ZoneInfo(user.get("timezone") or "UTC")).date()

    def record_day(payload, user):
        values = payload.model_dump(mode="json", exclude_none=True)
        values["log_date"] = values.get("log_date") or today(user).isoformat()
        return values

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Avoid echoing passwords or non-finite floats in error responses.
        errors = [
            {
                key: value
                for key, value in error.items()
                if key in ("loc", "msg", "type")
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    origins = [
        x.strip()
        for x in os.getenv(
            "NUTRIAI_CORS_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://localhost:8081,http://127.0.0.1:8081",
        ).split(",")
        if x.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def guard_cookie_origin(request: Request, call_next):
        # Cookies are same-site; require a same-origin or explicitly allowed origin for mutations.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            if origin not in origins and origin != str(request.base_url).rstrip("/"):
                return Response(status_code=403)
        if request.url.path == "/api/predict/food-image":
            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse(
                    status_code=400, content={"detail": "Invalid upload length."}
                )
            if not length:
                return JSONResponse(
                    status_code=411, content={"detail": "Upload length is required."}
                )
            if length > 6 * 1024 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Use a photo smaller than 5 MB."},
                )
        started = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = str(uuid.uuid4())
        if os.getenv("NUTRIAI_LOG_REQUESTS") == "true":
            route = getattr(request.scope.get("route"), "path", "/unknown")
            logging.getLogger("nutriai.requests").info(
                "%s %s %s %.3fs",
                request.method,
                route,
                response.status_code,
                time.monotonic() - started,
            )
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def token(request: Request):
        auth = request.headers.get("authorization", "")
        return (
            auth[7:]
            if auth.startswith("Bearer ")
            else request.cookies.get("nutriai_session", "")
        )

    def current_user(request: Request):
        user = store.session_user(token(request))
        if not user:
            raise HTTPException(401, "Please sign in")
        if os.getenv("NUTRIAI_REQUIRE_VERIFIED_PHONE") == "true" and not user.get(
            "phone_verified"
        ):
            if not request.url.path.startswith(
                ("/api/auth/", "/api/profile", "/api/account")
            ):
                raise HTTPException(
                    403, "Verify your phone from Settings before logging records."
                )
        return user

    def owner(user_id: str, user=Depends(current_user)):
        if user_id != user["id"]:
            raise HTTPException(403, "Access denied")
        return user

    def check_auth_limit(request, phone):
        address = request.client.host if request.client else "unknown"
        if not store.allow_auth_attempt(
            "ip:" + address, 30
        ) or not store.allow_auth_attempt("phone:" + phone.strip(), 10):
            raise HTTPException(
                429,
                "Too many authentication attempts. Try again in five minutes.",
                headers={"Retry-After": "300"},
            )

    def session_response(user, response, session=None):
        session = session or store.create_session(user["id"])
        response.set_cookie(
            "nutriai_session",
            session,
            httponly=True,
            samesite="lax",
            secure=os.getenv("NUTRIAI_SECURE_COOKIE", "false").lower() == "true",
            max_age=604800,
        )
        return {"user": user, "access_token": session, "token_type": "bearer"}

    @app.get("/api/health")
    def health():
        try:
            with store._get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
        except sqlite3.Error:
            raise HTTPException(503, "Database unavailable") from None
        return {"status": "ok", "database": "connected"}

    @app.get("/api/capabilities")
    def capabilities():
        return {
            "ai": ai.enabled,
            "vision": ai.enabled,
            "phone_verification": sms.enabled,
            "food_source": foods.provider,
            "require_verified_phone": os.getenv("NUTRIAI_REQUIRE_VERIFIED_PHONE")
            == "true",
        }

    @app.post("/api/auth/register", status_code=201)
    @app.post("/api/users/register", status_code=201)
    def register(payload: Register, response: Response, request: Request):
        check_auth_limit(request, payload.phone)
        record = payload.model_dump()
        record["phone_number"] = record.pop("phone").strip()
        if not record["phone_number"] or not record["full_name"].strip():
            raise HTTPException(422, "Name and phone are required")
        record["id"] = str(uuid.uuid4())
        record["username"] = record["id"]
        try:
            user, token, codes = store.register_account(record)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Phone number is already registered")
        result = session_response(user, response, token)
        result["recovery_codes"] = codes
        return result

    @app.post("/api/auth/login")
    def login(payload: Login, response: Response, request: Request):
        check_auth_limit(request, payload.phone)
        user = store.authenticate_user(payload.phone, payload.password)
        if not user:
            raise HTTPException(401, "Invalid credentials")
        return session_response(user, response)

    @app.post("/api/auth/logout", status_code=204)
    def logout(request: Request, response: Response):
        store.revoke_session(token(request))
        response.delete_cookie("nutriai_session")

    @app.get("/api/auth/me")
    def me(user=Depends(current_user)):
        return user

    @app.get("/api/users")
    def users(user=Depends(current_user)):
        return [user]

    @app.put("/api/profile/targets")
    def targets(payload: Targets, user=Depends(current_user)):
        store.create_or_update_user({**user, **payload.model_dump()})
        return store.get_user(user["id"])

    @app.get("/api/summary/{user_id}")
    def summary(user_id: str, date_str: date | None = None, user=Depends(owner)):
        return store.get_daily_summary(user_id, (date_str or today(user)).isoformat())

    @app.get("/api/meals")
    def meals(day: date | None = None, user=Depends(current_user)):
        return store.get_daily_summary(user["id"], (day or today(user)).isoformat())[
            "meals_list"
        ]

    @app.post("/api/meals", status_code=201)
    def add_meal(payload: Meal, user=Depends(current_user)):
        return {
            "id": store.log_meal({**record_day(payload, user), "user_id": user["id"]})
        }

    @app.put("/api/meals/{meal_id}")
    def update_meal(meal_id: str, payload: Meal, user=Depends(current_user)):
        values = record_day(payload, user)
        values["micronutrients_json"] = json.dumps(values.pop("micronutrients", {}))
        with store._get_connection() as conn:
            result = conn.execute(
                "UPDATE meal_logs SET "
                + ", ".join(f"{key} = ?" for key in values)
                + " WHERE id = ? AND user_id = ?",
                (*values.values(), meal_id, user["id"]),
            )
            if not result.rowcount:
                raise HTTPException(404, "Meal not found")
        return {"id": meal_id}

    @app.delete("/api/meals/{meal_id}", status_code=204)
    def delete_meal(meal_id: str, user=Depends(current_user)):
        with store._get_connection() as conn:
            result = conn.execute(
                "DELETE FROM meal_logs WHERE id = ? AND user_id = ?",
                (meal_id, user["id"]),
            )
            if not result.rowcount:
                raise HTTPException(404, "Meal not found")

    @app.post("/api/hydration", status_code=201)
    def hydration(payload: Water, user=Depends(current_user)):
        return {
            "id": store.log_water(
                user["id"],
                payload.amount_ml,
                log_date=(payload.log_date or today(user)).isoformat(),
            )
        }

    @app.delete("/api/hydration", status_code=204)
    def reset_hydration(day: date | None = None, user=Depends(current_user)):
        with store._get_connection() as conn:
            conn.execute(
                "DELETE FROM water_logs WHERE user_id = ? AND log_date = ?",
                (user["id"], (day or today(user)).isoformat()),
            )

    @app.put("/api/biometrics")
    def biometrics(payload: Biometrics, user=Depends(current_user)):
        store.log_biometrics(user["id"], record_day(payload, user))
        return store.get_daily_summary(
            user["id"], (payload.log_date or today(user)).isoformat()
        )["biometrics"]

    @app.get("/api/coach/weekly-report/{user_id}")
    def weekly_report(user_id: str, user=Depends(owner)):
        last = today(user)
        days = store.daily_history(user_id, last - timedelta(days=6), last)
        tracked = [day for day in days if day["meal_count"]]
        return {
            "days_with_meals": len(tracked),
            "window_days": 7,
            "average_calories_on_logged_days": (
                round(sum(d["calories"] for d in tracked) / len(tracked), 1)
                if tracked
                else None
            ),
            "average_protein_on_logged_days": (
                round(sum(d["protein_g"] for d in tracked) / len(tracked), 1)
                if tracked
                else None
            ),
            "total_water_ml": sum(d["water_ml"] for d in days),
            "daily_calorie_target": user["daily_calorie_target"],
        }

    @app.get("/api/patterns/eating-pattern/{user_id}")
    def patterns(user_id: str, user=Depends(owner)):
        last = today(user)
        history = store.daily_history(user_id, last - timedelta(days=6), last)
        from ml.rnn_pattern_analyzer import RecurrentEatingPatternAnalyzer
        analyzer = RecurrentEatingPatternAnalyzer()
        rnn_res = analyzer.analyze_user_meal_sequence(
            history, target_calories=user.get("daily_calorie_target", 2000.0)
        )
        return {
            "logged_days": sum(1 for d in history if d["meal_count"]),
            "status": "rnn_analyzed",
            "message": rnn_res["chrononutrition_insight"],
            "detected_pattern": rnn_res["detected_pattern"],
            "pattern_confidence": rnn_res["pattern_confidence"],
            "forecasted_next_day_calories": rnn_res["forecasted_next_day_calories"],
            "forecast_category": rnn_res["forecast_category"],
            "daily_calories": [
                {"date": d["date"], "calories": d["calories"]}
                for d in history
                if d["meal_count"]
            ],
        }

    @app.post("/api/coach/chat")
    def chat(payload: Chat, user=Depends(current_user)):
        if not store.allow_auth_attempt("ai:" + user["id"], 15):
            raise HTTPException(429, "Please wait before sending another message.")
        summary = store.get_daily_summary(user["id"], today(user).isoformat())
        context = {
            "date": summary["date"],
            "calories": summary["macros"]["calories"]["consumed"],
            "protein_g": summary["macros"]["protein_g"]["consumed"],
            "target_calories": user["daily_calorie_target"],
            "dietary_preference": user["dietary_preference"],
            "allergies": user["allergies"],
        }
        history = store.get_chat_history(user["id"], "default", 12)
        result = ai.chat(payload.message, context, history)
        # Save both turns in one transaction so a failed write never creates half a conversation.
        with store._get_connection() as conn:
            for sender, message in [
                ("user", payload.message),
                ("assistant", result["reply"]),
            ]:
                conn.execute(
                    "INSERT INTO ai_consultations(id,user_id,session_id,sender,message_text) VALUES (?,?,?,?,?)",
                    (str(uuid.uuid4()), user["id"], "default", sender, message),
                )
        return result

    @app.get("/api/coach/history")
    def history(user=Depends(current_user)):
        return store.get_chat_history(user["id"], "default")

    @app.delete("/api/coach/history", status_code=204)
    def clear_history(user=Depends(current_user)):
        with store._get_connection() as conn:
            conn.execute("DELETE FROM ai_consultations WHERE user_id=?", (user["id"],))

    @app.post("/api/oracle/ask")
    def oracle(payload: Oracle, user=Depends(current_user)):
        from ml.genai_coach import GenAINutritionCoach
        coach = GenAINutritionCoach()
        res = coach.consult_omni_oracle(payload.query, payload.category)
        reply = (
            f"**{res['topic']}**\n\n"
            f"{res['summary']}\n\n"
            f"**Biological Mechanism:**\n{res['scientific_mechanism']}\n\n"
            f"**Actionable Protocol:**\n"
            + "\n".join(f"- {p}" for p in res["actionable_protocol"])
            + f"\n\n*Clinical Evidence:* {res['clinical_evidence']}"
        )
        return {"reply": reply, "oracle": res, "mode": "omni_oracle"}

    @app.post("/api/predict/food-image")
    def predict(
        file: UploadFile = File(...),
        serving_size_g: float = Form(100, gt=0, le=5000),
        user=Depends(current_user),
    ):
        if not ai.enabled:
            raise ProviderError(
                "Photo analysis is not configured. You can still log food manually."
            )
        if not store.allow_auth_attempt("vision:" + user["id"], 5):
            raise HTTPException(429, "Please wait before analyzing another photo.")
        raw = file.file.read(5 * 1024 * 1024 + 1)
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(413, "Use a photo smaller than 5 MB.")
        from PIL import Image, UnidentifiedImageError, ImageOps

        try:
            with Image.open(io.BytesIO(raw)) as source:
                if source.width * source.height > 12_000_000:
                    raise HTTPException(
                        413, "Choose a photo smaller than 12 megapixels."
                    )
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((1280, 1280))
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=85)
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            raise HTTPException(
                422, "Choose a valid JPEG, PNG, or WebP photo."
            ) from None
        result = ai.analyze(output.getvalue())
        return {"estimate": result, "requires_confirmation": True, "saved": False}

    @app.get("/api/foods/search")
    def search_foods(
        q: str = Query(min_length=2, max_length=100), user=Depends(current_user)
    ):
        if not store.allow_auth_attempt("food:" + user["id"], 30):
            raise HTTPException(429, "Please wait before searching again.")
        return {"foods": foods.search(q), "source": foods.provider}

    @app.put("/api/profile")
    def profile(payload: Profile, user=Depends(current_user)):
        values = payload.model_dump()
        weight = values.pop("current_weight_kg")
        return store.save_profile_with_weight(
            user["id"], values, weight, today({**user, **values}).isoformat()
        )

    @app.get("/api/progress")
    def progress(
        days: int = Query(default=14, ge=7, le=90),
        end: date | None = None,
        user=Depends(current_user),
    ):
        last = end or today(user)
        records = store.daily_history(user["id"], last - timedelta(days=days - 1), last)
        totals = [row["calories"] for row in records if row["meal_count"]]
        return {
            "days": records,
            "logged_days": len(totals),
            "average_calories": round(sum(totals) / len(totals), 1) if totals else None,
            "trend": "insufficient_data" if len(totals) < 7 else "descriptive",
            "note": "Unlogged days are excluded. These are recorded totals, not a prediction of intake.",
        }

    @app.post("/api/auth/password", status_code=204)
    def password_change(payload: ChangePassword, user=Depends(current_user)):
        check_user_limit(user["id"])
        if not store.change_password(
            user["id"], payload.current_password, payload.new_password
        ):
            raise HTTPException(401, "Current password is incorrect.")

    def check_user_limit(user_id):
        if not store.allow_auth_attempt("security:" + user_id, 10):
            raise HTTPException(429, "Too many attempts. Try again in five minutes.")

    @app.post("/api/auth/recovery-codes")
    def recovery_codes(payload: Reauthenticate, user=Depends(current_user)):
        check_user_limit(user["id"])
        if not store.verify_current_password(user["id"], payload.current_password):
            raise HTTPException(401, "Current password is incorrect.")
        return {"recovery_codes": store.generate_recovery_codes(user["id"])}

    @app.post("/api/auth/recover", status_code=204)
    def recover(payload: Recovery, request: Request):
        check_auth_limit(request, payload.phone)
        if not store.reset_with_code(
            payload.phone, payload.recovery_code, payload.password
        ):
            raise HTTPException(400, "Phone number or recovery code is invalid.")

    @app.post("/api/auth/verification/start")
    def start_verification(user=Depends(current_user)):
        check_user_limit(user["id"])
        sms.send(user["phone_number"])
        return {
            "challenge_id": store.create_challenge(
                user["id"], user["phone_number"], "verify"
            )
        }

    @app.post("/api/auth/verification/check")
    def check_verification(payload: ChallengeCheck, user=Depends(current_user)):
        check_user_limit(user["id"])
        challenge = store.challenge(payload.challenge_id, "verify", user["id"])
        if (
            not challenge
            or not sms.check(challenge["phone"], payload.code)
            or not store.complete_challenge(payload.challenge_id, "verify")
        ):
            raise HTTPException(400, "The code is invalid or expired.")
        return store.get_user(user["id"])

    @app.post("/api/auth/reset/start")
    def start_reset(payload: PhoneRequest, request: Request):
        check_auth_limit(request, payload.phone)
        if not sms.enabled:
            raise ProviderError(
                "SMS recovery is not configured. Use a saved recovery code."
            )
        account = store.get_user_by_phone(payload.phone)
        if account and account.get("phone_verified"):
            sms.send(payload.phone)
        # Unknown/unverified numbers receive the same response and cannot complete a reset.
        uid = account["id"] if account and account.get("phone_verified") else None
        return {
            "challenge_id": store.create_challenge(uid, payload.phone, "reset"),
            "message": "If this number is verified on an account, a code has been sent.",
        }

    @app.post("/api/auth/reset/check", status_code=204)
    def check_reset(payload: SMSReset, request: Request):
        check_auth_limit(request, "reset-check")
        challenge = store.challenge(payload.challenge_id, "reset")
        if (
            not challenge
            or not challenge["user_id"]
            or not sms.check(challenge["phone"], payload.code)
            or not store.complete_challenge(
                payload.challenge_id, "reset", payload.new_password
            )
        ):
            raise HTTPException(400, "The code is invalid or expired.")

    @app.get("/api/account/export")
    def export(user=Depends(current_user)):
        return JSONResponse(
            content=store.export_user(user["id"]),
            headers={
                "Content-Disposition": 'attachment; filename="nutriai-export.json"'
            },
        )

    @app.delete("/api/account", status_code=204)
    def delete_account(
        payload: Reauthenticate, response: Response, user=Depends(current_user)
    ):
        check_user_limit(user["id"])
        if not store.delete_user(user["id"], payload.current_password):
            raise HTTPException(401, "Current password is incorrect.")
        response.delete_cookie("nutriai_session")

    web = Path(__file__).resolve().parent.parent / "web"

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(web / "index.html")

    @app.get("/styles.css", include_in_schema=False)
    def styles():
        return FileResponse(web / "styles.css", media_type="text/css")

    @app.get("/app.js", include_in_schema=False)
    def app_js():
        return FileResponse(web / "app.js", media_type="application/javascript")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    app.mount("/web", StaticFiles(directory=web, html=True), name="web")
    return app
