
import time

from django.conf import settings
from django.shortcuts import render

from pathlib import Path
from datetime import datetime, timedelta
import json

from search_engine.search import search as run_search
from search_engine.storage import load_json
from classifier.predict import predict_label
from classifier.benchmark import run_benchmark

INDEX_PATH = settings.BASE_DIR / "data" / "index.json"
MODEL_PATH = settings.BASE_DIR / "data" / "model.joblib"


def load_index():
    return load_json(str(INDEX_PATH))


def home(request):
    return render(request, "index.html")


def search(request):
    start_time = time.time()
    q = (request.GET.get("q") or "").strip()
    use_stemming = request.GET.get("stem") == "1"
    specialized = request.GET.get("specialized") == "1"
    payload = load_index()
    results = []

    org_filter = "CSM" if specialized else None

    if q and payload:
        results = run_search(q, payload, top_k=15, use_stemming=use_stemming, org_filter=org_filter)

    end_time = time.time()
    response_time = (end_time - start_time) * 100

    context = {
        "q": q,
        "results": results,
        "use_stemming": use_stemming,
        "specialized": specialized,
        "has_index": bool(payload),
        "response_time": response_time,
    }
    return render(request, "results.html", context)


def classify(request):
    start_time = time.time()
    text = ""
    label = None
    model_ready = MODEL_PATH.exists()

    if request.method == "POST":
        text = (request.POST.get("text") or "").strip()
        if text and model_ready:
            label = predict_label(text)

    end_time = time.time()
    response_time = (end_time - start_time) * 100

    context = {
        "text": text,
        "label": label,
        "model_ready": model_ready,
        "response_time": response_time,
    }
    return render(request, "classify.html", context)


def model_selection(request):
    results = run_benchmark()
    return render(request, "model_selection.html", {"results": results})
