from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

latest_command = {
    "command": None,
    "value": None,
    "reply": None
}

device_status = {
    "brightness": 0,
    "auto_mode": True,
    "ambient_lux": None,
    "distance_cm": None,
    "online": False
}

class UserInput(BaseModel):
    text: str

class DeviceReport(BaseModel):
    brightness: int | None = None
    auto_mode: bool | None = None
    ambient_lux: float | None = None
    distance_cm: float | None = None

@app.get("/")
def root():
    return {"ok": True, "message": "backend running"}

@app.post("/ask")
def ask_ai(data: UserInput):
    global latest_command

    prompt = f"""
Bạn là trợ lý điều khiển đèn học thông minh.
Hãy trả về JSON hợp lệ duy nhất.

Các command hợp lệ:
- lamp_on
- lamp_off
- brighter
- dimmer
- set_brightness
- auto_mode
- manual_mode
- status
- none

Quy tắc:
- reply phải là tiếng Việt tự nhiên, ngắn gọn, thân thiện.
- Nếu người dùng yêu cầu đặt độ sáng cụ thể, dùng set_brightness và value 0-100.
- Nếu không phải lệnh điều khiển, dùng command = none.
- Nếu câu hỏi là hỏi trạng thái, dùng command = status.

Trạng thái thiết bị hiện tại:
{json.dumps(device_status, ensure_ascii=False)}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": data.text}
        ]
    )

    raw = response.output_text.strip()

    try:
      parsed = json.loads(raw)
    except Exception:
      parsed = {
          "reply": "Mình chưa hiểu rõ, bạn nói lại giúp mình nhé.",
          "command": "none"
      }

    if parsed.get("command") == "status":
        parsed["reply"] = (
            f"Đèn đang ở {device_status['brightness']} phần trăm, "
            f"chế độ tự động là {'bật' if device_status['auto_mode'] else 'tắt'}."
        )

    latest_command = parsed
    return parsed

@app.get("/device/pull")
def device_pull():
    global latest_command
    cmd = latest_command.copy()
    latest_command = {"command": None, "value": None, "reply": None}
    return cmd

@app.post("/device/report")
def report_device(data: DeviceReport):
    global device_status
    if data.brightness is not None:
        device_status["brightness"] = data.brightness
    if data.auto_mode is not None:
        device_status["auto_mode"] = data.auto_mode
    if data.ambient_lux is not None:
        device_status["ambient_lux"] = data.ambient_lux
    if data.distance_cm is not None:
        device_status["distance_cm"] = data.distance_cm
    device_status["online"] = True
    return {"ok": True}

@app.get("/device/status")
def get_status():
    return device_status
