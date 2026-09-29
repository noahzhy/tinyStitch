"""Local asynchronous stitching service. Input contract contains RGB references only."""

import base64
import json
import os
import re
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .stitch import stitch, load_image

DATA = Path(
    os.environ.get("tinyStitch_DATA", Path(__file__).resolve().parents[1] / "data")
).resolve()
for subdir in ["inputs", "jobs", "results"]:
    (DATA / subdir).mkdir(parents=True, exist_ok=True)
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shelf-stitch")
LOCK = threading.RLock()
CANCEL = {}
for path in (DATA / "jobs").glob("*.json"):
    saved = json.loads(path.read_text())
    if saved["status"] in {"queued", "running"}:
        saved.update(status="interrupted", message="服务重启，请重新拼接")
        path.write_text(json.dumps(saved, ensure_ascii=False))
app = FastAPI(title="tinyStitch 货架拼图", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5180", "http://localhost:5180"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def valid_id(value):
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise HTTPException(400, "无效的数据 ID")
    return value


def save_job(job):
    target = DATA / "jobs" / f"{job['id']}.json"
    tmp = target.with_suffix(".tmp")
    with LOCK:
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2))
        tmp.replace(target)


def read_job(id):
    path = DATA / "jobs" / f"{valid_id(id)}.json"
    if not path.exists():
        raise HTTPException(404, "任务不存在")
    with LOCK:
        return json.loads(path.read_text())


def input_info(id):
    directory = DATA / "inputs" / valid_id(id)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        raise HTTPException(404, "图片序列不存在")
    info = json.loads(manifest.read_text())
    return {
        "id": id,
        "count": len(info["files"]),
        "names": info["names"],
        "preview": [f"/api/inputs/{id}/{name}" for name in info["files"]],
    }


def finish_input(id, files, names):
    directory = DATA / "inputs" / id
    (directory / "manifest.json").write_text(
        json.dumps({"files": files, "names": names}, ensure_ascii=False)
    )
    return input_info(id)


class Captures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: list[str] = Field(min_length=2, max_length=48)


class StitchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rgb_id: str
    method: Literal["jepa", "sift"] = "jepa"


@app.get("/api/health")
def health():
    from .jepa import get_checkpoint

    info = None
    try:
        _, info = get_checkpoint()
    except Exception:
        pass
    path = Path(__file__).resolve().parents[1] / "artifacts/models/training-state.json"
    training = json.loads(path.read_text()) if path.exists() else None
    return {
        "status": "ok",
        "algorithm": "dense JEPA planar mosaic",
        "jepa_ready": info is not None,
        "model": info,
        "training": training,
    }


@app.post("/api/captures")
def capture(request: Captures):
    id = uuid.uuid4().hex
    directory = DATA / "inputs" / id
    directory.mkdir()
    files = []
    try:
        for i, frame in enumerate(request.frames):
            if not frame.startswith(
                ("data:image/jpeg;base64,", "data:image/png;base64,")
            ):
                raise ValueError("只接受 JPEG 或 PNG 的 RGB 图片")
            data = base64.b64decode(frame.split(",", 1)[1], validate=True)
            if len(data) > 25_000_000:
                raise ValueError("单张图片超过 25 MB")
            name = f"{i:04d}.{'png' if frame.startswith('data:image/png') else 'jpg'}"
            path = directory / name
            path.write_bytes(data)
            load_image(path)
            files.append(name)
        return finish_input(id, files, files)
    except Exception as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/uploads")
async def upload(files: list[UploadFile] = File(...)):
    if not 2 <= len(files) <= 48:
        raise HTTPException(400, "请选择 2–48 张同一货架的连续拍摄照片")
    id = uuid.uuid4().hex
    directory = DATA / "inputs" / id
    directory.mkdir()
    # Preserve natural filename order, not lexicographic 1,10,2 or asynchronous completion order.
    files = sorted(
        files,
        key=lambda f: [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", f.filename or "")
        ],
    )
    stored, names, total = [], [], 0
    try:
        for i, file in enumerate(files):
            name = Path(file.filename or f"frame{i}.jpg").name
            suffix = Path(name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise ValueError("支持 JPG、PNG、WebP；HEIC 请在手机分享时转换为 JPEG")
            data = await file.read(25_000_001)
            total += len(data)
            if len(data) > 25_000_000 or total > 300_000_000:
                raise ValueError("单张限制 25 MB，整组限制 300 MB")
            filename = f"{i:04d}{suffix}"
            path = directory / filename
            path.write_bytes(data)
            load_image(path)
            stored.append(filename)
            names.append(name)
        return finish_input(id, stored, names)
    except Exception as error:
        raise HTTPException(400, f"图片导入失败：{error}") from error


class Cancelled(Exception):
    pass


@app.post("/api/stitch")
def create(request: StitchRequest):
    info = input_info(request.rgb_id)
    id = uuid.uuid4().hex
    event = threading.Event()
    CANCEL[id] = event
    job = {
        "id": id,
        "status": "queued",
        "progress": 0.0,
        "message": "等待拼接",
        "rgb_id": request.rgb_id,
    }
    save_job(job)

    def run():
        def progress(value, message):
            if event.is_set():
                raise Cancelled()
            job.update(status="running", progress=value, message=message)
            save_job(job)

        try:
            progress(0.0, "开始自动拼接")
            directory = DATA / "inputs" / request.rgb_id
            manifest = json.loads((directory / "manifest.json").read_text())
            paths = [directory / name for name in manifest["files"]]
            report = stitch(
                paths, DATA / "results" / id, progress, method=request.method
            )
            if event.is_set():
                raise Cancelled()
            job.update(
                status="completed",
                progress=1.0,
                message="拼图完成",
                result={
                    "report": report,
                    "png": f"/api/files/{id}/panorama.png",
                    "jpg": f"/api/files/{id}/panorama.jpg",
                    "json": f"/api/files/{id}/report.json",
                },
            )
        except Cancelled:
            job.update(status="cancelled", message="已取消")
        except Exception as error:
            job.update(status="failed", message="拼接失败", error=str(error))
        finally:
            save_job(job)
            CANCEL.pop(id, None)

    POOL.submit(run)
    return job


@app.get("/api/jobs/{id}")
def job(id: str):
    return read_job(id)


@app.post("/api/jobs/{id}/cancel")
def cancel(id: str):
    saved = read_job(id)
    event = CANCEL.get(id)
    if event is not None:
        event.set()
    return saved


@app.get("/api/inputs/{id}/download")
def download_input(id: str):
    info = input_info(id)
    directory = DATA / "inputs" / id
    path = directory / "sequence.zip"
    with LOCK:
        if not path.exists():
            manifest = json.loads((directory / "manifest.json").read_text())
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for name in manifest["files"]:
                    archive.write(directory / name, name)
    return FileResponse(path, filename="shelf-sequence.zip")


@app.get("/api/inputs/{id}/{name}")
def input_file(id: str, name: str):
    directory = DATA / "inputs" / valid_id(id)
    if (
        not re.fullmatch(r"\d{4}\.(jpg|jpeg|png|webp)", name)
        or not (directory / name).exists()
    ):
        raise HTTPException(404, "图片不存在")
    return FileResponse(directory / name)


@app.get("/api/files/{id}/{name}")
def result_file(id: str, name: str):
    saved = read_job(id)
    if name not in {"panorama.png", "panorama.jpg", "report.json", "diagnostics.json"}:
        raise HTTPException(404, "文件不存在")
    path = DATA / "results" / valid_id(id) / name
    if (
        not path.exists()
        or saved["status"] != "completed"
        and name != "diagnostics.json"
    ):
        raise HTTPException(404, "结果尚未完成")
    return FileResponse(path)


@app.get("/api/experiment-report")
def experiment_report():
    path = Path(__file__).resolve().parents[1] / "artifacts/report.html"
    if not path.exists():
        raise HTTPException(404, "尚未生成实验报告")
    return FileResponse(path, media_type="text/html")
