#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <BH1750.h>
#include <Preferences.h>
#include <WebServer.h>
#include "driver/i2s.h"

#if __has_include("secrets.h")
  #include "secrets.h"
#elif __has_include("secrets.example.h")
  #include "secrets.example.h"
#endif

#ifndef BACKEND_URL
  #define BACKEND_URL "https://smart-lamp-backend.onrender.com"
#endif

#ifndef CONFIG_AP_SSID_PREFIX
  #define CONFIG_AP_SSID_PREFIX "SmartLamp"
#endif

#ifndef CONFIG_AP_PASS
  #define CONFIG_AP_PASS "12345678"
#endif

// ================= ESP32-S3 44P PIN MAP =================

// BH1750
constexpr uint8_t PIN_I2C_SDA = 8;
constexpr uint8_t PIN_I2C_SCL = 9;

// HC-SR04
constexpr uint8_t PIN_TRIG = 4;
constexpr uint8_t PIN_ECHO = 5;

// PWM MOS + Relay
constexpr uint8_t PIN_PWM_LED = 6;
constexpr uint8_t PIN_RELAY = 7;

// INMP441
constexpr int PIN_MIC_SCK = 15;
constexpr int PIN_MIC_WS  = 16;
constexpr int PIN_MIC_SD  = 17;

// MAX98357A
constexpr int PIN_SPK_BCLK = 18;
constexpr int PIN_SPK_LRC  = 21;
constexpr int PIN_SPK_DIN  = 47;

constexpr bool RELAY_ACTIVE_LOW = true;

// ================= CONFIG =================

constexpr uint32_t PWM_FREQ = 20000;
constexpr uint8_t PWM_RES_BITS = 10;
constexpr uint16_t PWM_MAX = (1 << PWM_RES_BITS) - 1;

constexpr float POSTURE_BAD_CM = 30.0;
constexpr float POSTURE_OK_CM = 35.0;
constexpr float ABSENT_CM = 50.0;

constexpr uint32_t POSTURE_CONFIRM_MS = 2000;
constexpr uint32_t POSTURE_ALERT_REPEAT_MS = 15000;
constexpr uint32_t ABSENT_CONFIRM_MS = 10000;

constexpr int MIC_SAMPLE_RATE = 8000;
constexpr int RECORD_SECONDS = 2;
constexpr int MIC_BITS = 16;
constexpr int MIC_SAMPLES = MIC_SAMPLE_RATE * RECORD_SECONDS;

constexpr i2s_port_t I2S_MIC_PORT = I2S_NUM_0;
constexpr i2s_port_t I2S_SPK_PORT = I2S_NUM_1;

// ================= GLOBAL =================

Preferences prefs;
WebServer server(80);
BH1750 lightMeter;
WiFiClientSecure secureClient;

String savedSsid;
String savedPass;
String apSsid;

bool wifiConnecting = false;
uint32_t wifiStartMs = 0;
uint32_t lastWifiRetryMs = 0;

bool lampPower = false;
uint8_t brightness = 100;

float lux = NAN;
float distanceCm = NAN;

bool present = false;
bool postureBad = false;

uint32_t absentSinceMs = 0;
uint32_t postureBadSinceMs = 0;
uint32_t lastPostureAlertMs = 0;
bool noPersonAlertSent = false;

// ================= TIMER =================

struct Timer {
  uint32_t last = 0;
  uint32_t period;

  bool due() {
    uint32_t now = millis();
    if (now - last >= period) {
      last = now;
      return true;
    }
    return false;
  }
};

Timer tWifi{0, 3000};
Timer tDistance{0, 300};
Timer tLux{0, 1000};
Timer tLogic{0, 300};
Timer tReport{0, 5000};
Timer tPull{0, 1500};
Timer tDebug{0, 2000};

// ================= UTILS =================

bool relayLevel(bool on) {
  return RELAY_ACTIVE_LOW ? !on : on;
}

String htmlEscape(String s) {
  s.replace("&", "&amp;");
  s.replace("<", "&lt;");
  s.replace(">", "&gt;");
  s.replace("\"", "&quot;");
  return s;
}

String makeApSsid() {
  uint64_t chip = ESP.getEfuseMac();
  char buf[32];
  snprintf(buf, sizeof(buf), "%s-%04X", CONFIG_AP_SSID_PREFIX, (uint16_t)(chip & 0xFFFF));
  return String(buf);
}

uint16_t le16(const uint8_t* p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

uint32_t le32(const uint8_t* p) {
  return (uint32_t)p[0] |
         ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

bool readExact(Stream* s, uint8_t* buf, size_t len, uint32_t timeoutMs = 10000) {
  size_t got = 0;
  uint32_t start = millis();

  while (got < len) {
    if (s->available()) {
      got += s->readBytes(buf + got, len - got);
      start = millis();
    } else {
      if (millis() - start > timeoutMs) return false;
      delay(1);
      server.handleClient();
      yield();
    }
  }

  return true;
}

// ================= LAMP =================

void pwmBegin() {
  ledcAttach(PIN_PWM_LED, PWM_FREQ, PWM_RES_BITS);
}

void pwmWritePercent(uint8_t percent) {
  percent = constrain(percent, 0, 100);
  uint16_t duty = lampPower ? map(percent, 0, 100, 0, PWM_MAX) : 0;
  ledcWrite(PIN_PWM_LED, duty);
}

void setLamp(bool on) {
  lampPower = on;
  digitalWrite(PIN_RELAY, relayLevel(on));
  pwmWritePercent(brightness);
}

void setBrightness(uint8_t value) {
  brightness = constrain(value, 0, 100);
  pwmWritePercent(brightness);
}

// ================= WIFI SAVE =================

void loadWifi() {
  prefs.begin("wifi", true);
  savedSsid = prefs.getString("ssid", "");
  savedPass = prefs.getString("pass", "");
  prefs.end();
}

void saveWifi(const String& ssid, const String& pass) {
  prefs.begin("wifi", false);
  prefs.putString("ssid", ssid);
  prefs.putString("pass", pass);
  prefs.end();
  savedSsid = ssid;
  savedPass = pass;
}

void clearWifi() {
  prefs.begin("wifi", false);
  prefs.clear();
  prefs.end();
  savedSsid = "";
  savedPass = "";
}

// ================= I2S SPEAKER =================

void speakerBegin() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate = 24000;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 512;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = true;

  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = PIN_SPK_BCLK;
  pins.ws_io_num = PIN_SPK_LRC;
  pins.data_out_num = PIN_SPK_DIN;
  pins.data_in_num = I2S_PIN_NO_CHANGE;

  i2s_driver_install(I2S_SPK_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_SPK_PORT, &pins);
  i2s_zero_dma_buffer(I2S_SPK_PORT);

  Serial.println("MAX98357A ready");
}

bool playWavData(Stream* stream, uint32_t dataSize, uint16_t channels) {
  uint8_t buffer[1024];
  uint32_t remain = dataSize;
  uint16_t frameSize = channels * 2;

  while (remain > 0) {
    int avail = stream->available();

    if (avail <= 0) {
      delay(1);
      server.handleClient();
      yield();
      continue;
    }

    uint32_t toRead = sizeof(buffer);
    if (toRead > remain) toRead = remain;
    if ((uint32_t)avail < toRead) toRead = avail;

    toRead -= toRead % frameSize;
    if (toRead == 0) {
      delay(1);
      continue;
    }

    int n = stream->readBytes(buffer, toRead);
    if (n <= 0) break;

    if (channels == 1) {
      size_t written = 0;
      i2s_write(I2S_SPK_PORT, buffer, n, &written, portMAX_DELAY);
    } else {
      int16_t* stereo = (int16_t*)buffer;
      int samples = n / 4;
      int16_t mono[256];

      if (samples > 256) samples = 256;

      for (int i = 0; i < samples; i++) {
        int16_t left = stereo[i * 2];
        int16_t right = stereo[i * 2 + 1];
        mono[i] = (int16_t)(((int32_t)left + right) / 2);
      }

      size_t written = 0;
      i2s_write(I2S_SPK_PORT, mono, samples * sizeof(int16_t), &written, portMAX_DELAY);
    }

    remain -= n;
    server.handleClient();
    yield();
  }

  i2s_zero_dma_buffer(I2S_SPK_PORT);
  return true;
}

bool playWavUrl(const String& url) {
  if (url.length() == 0 || WiFi.status() != WL_CONNECTED) return false;

  Serial.print("Playing WAV: ");
  Serial.println(url);

  WiFiClientSecure c;
  c.setInsecure();

  HTTPClient http;
  if (!http.begin(c, url)) {
    Serial.println("WAV http begin failed");
    return false;
  }

  http.setTimeout(30000);
  int code = http.GET();

  if (code != 200) {
    Serial.print("WAV HTTP code: ");
    Serial.println(code);
    http.end();
    return false;
  }

  WiFiClient* stream = http.getStreamPtr();

  uint8_t riff[12];
  if (!readExact(stream, riff, 12)) {
    Serial.println("WAV RIFF read failed");
    http.end();
    return false;
  }

  if (memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
    Serial.println("Invalid WAV RIFF");
    http.end();
    return false;
  }

  uint16_t channels = 1;
  uint32_t sampleRate = 24000;
  uint16_t bits = 16;
  bool fmtFound = false;
  bool dataPlayed = false;

  while (http.connected()) {
    uint8_t chunkHeader[8];
    if (!readExact(stream, chunkHeader, 8, 5000)) break;

    char chunkId[5] = {0};
    memcpy(chunkId, chunkHeader, 4);
    uint32_t chunkSize = le32(chunkHeader + 4);

    if (memcmp(chunkHeader, "fmt ", 4) == 0) {
      uint8_t fmt[32];
      uint32_t toRead = chunkSize > sizeof(fmt) ? sizeof(fmt) : chunkSize;

      if (!readExact(stream, fmt, toRead)) break;

      if (chunkSize > toRead) {
        uint8_t dump[64];
        uint32_t left = chunkSize - toRead;
        while (left > 0) {
          uint32_t n = left > sizeof(dump) ? sizeof(dump) : left;
          if (!readExact(stream, dump, n)) break;
          left -= n;
        }
      }

      channels = le16(fmt + 2);
      sampleRate = le32(fmt + 4);
      bits = le16(fmt + 14);
      fmtFound = true;

      Serial.print("WAV fmt rate=");
      Serial.print(sampleRate);
      Serial.print(" channels=");
      Serial.print(channels);
      Serial.print(" bits=");
      Serial.println(bits);

      if (bits != 16) {
        Serial.println("Only 16-bit WAV supported");
        http.end();
        return false;
      }

      i2s_set_clk(I2S_SPK_PORT, sampleRate, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
    }
    else if (memcmp(chunkHeader, "data", 4) == 0) {
      if (!fmtFound) {
        Serial.println("WAV data before fmt");
      }

      Serial.print("WAV data size=");
      Serial.println(chunkSize);

      dataPlayed = playWavData(stream, chunkSize, channels);
      break;
    }
    else {
      uint8_t dump[64];
      uint32_t left = chunkSize;

      while (left > 0) {
        uint32_t n = left > sizeof(dump) ? sizeof(dump) : left;
        if (!readExact(stream, dump, n)) break;
        left -= n;
      }
    }
  }

  http.end();

  if (dataPlayed) Serial.println("WAV finished");
  else Serial.println("WAV not played");

  return dataPlayed;
}

// ================= I2S MIC =================

void micBegin() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = MIC_SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;

  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = PIN_MIC_SCK;
  pins.ws_io_num = PIN_MIC_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = PIN_MIC_SD;

  i2s_driver_install(I2S_MIC_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_MIC_PORT, &pins);
  i2s_zero_dma_buffer(I2S_MIC_PORT);

  Serial.println("INMP441 ready");
}

void wavHeader(uint8_t* h, uint32_t dataSize) {
  uint32_t fileSize = dataSize + 36;
  uint32_t byteRate = MIC_SAMPLE_RATE * 2;
  uint16_t blockAlign = 2;

  memcpy(h, "RIFF", 4);
  memcpy(h + 4, &fileSize, 4);
  memcpy(h + 8, "WAVE", 4);
  memcpy(h + 12, "fmt ", 4);

  uint32_t fmtSize = 16;
  uint16_t audioFormat = 1;
  uint16_t channels = 1;
  uint32_t rate = MIC_SAMPLE_RATE;
  uint16_t bits = 16;

  memcpy(h + 16, &fmtSize, 4);
  memcpy(h + 20, &audioFormat, 2);
  memcpy(h + 22, &channels, 2);
  memcpy(h + 24, &rate, 4);
  memcpy(h + 28, &byteRate, 4);
  memcpy(h + 32, &blockAlign, 2);
  memcpy(h + 34, &bits, 2);
  memcpy(h + 36, "data", 4);
  memcpy(h + 40, &dataSize, 4);
}

uint8_t* recordWav(size_t& wavSize) {
  size_t pcmBytes = MIC_SAMPLES * 2;
  wavSize = pcmBytes + 44;

  uint8_t* wav = (uint8_t*)malloc(wavSize);
  if (!wav) return nullptr;

  wavHeader(wav, pcmBytes);

  int16_t* pcm = (int16_t*)(wav + 44);
  int32_t raw[256];
  int idx = 0;

  Serial.println("Recording voice 2s...");

  while (idx < MIC_SAMPLES) {
    size_t bytesRead = 0;
    i2s_read(I2S_MIC_PORT, raw, sizeof(raw), &bytesRead, portMAX_DELAY);

    int count = bytesRead / 4;

    for (int i = 0; i < count && idx < MIC_SAMPLES; i++) {
      pcm[idx++] = (int16_t)(raw[i] >> 14);
    }

    server.handleClient();
    yield();
  }

  Serial.println("Voice recorded");
  return wav;
}

// ================= HTTP =================

bool httpPostJson(const String& path, const JsonDocument& doc, String* responseOut = nullptr) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  String url = String(BACKEND_URL) + path;

  if (!http.begin(secureClient, url)) return false;

  http.setTimeout(12000);
  http.addHeader("Content-Type", "application/json");

  String body;
  serializeJson(doc, body);

  int code = http.POST(body);
  String res = http.getString();
  http.end();

  if (responseOut) *responseOut = res;

  return code >= 200 && code < 300;
}

bool httpGetJson(const String& path, JsonDocument& doc) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  String url = String(BACKEND_URL) + path;

  if (!http.begin(secureClient, url)) return false;

  http.setTimeout(8000);
  int code = http.GET();

  if (code != 200) {
    http.end();
    return false;
  }

  String payload = http.getString();
  http.end();

  return deserializeJson(doc, payload) == DeserializationError::Ok;
}

bool uploadVoice() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected");
    return false;
  }

  size_t wavSize = 0;
  uint8_t* wav = recordWav(wavSize);

  if (!wav) {
    Serial.println("Record malloc failed");
    return false;
  }

  String boundary = "----SmartLamp";
  String head =
    "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"file\"; filename=\"voice.wav\"\r\n"
    "Content-Type: audio/wav\r\n\r\n";

  String tail = "\r\n--" + boundary + "--\r\n";

  size_t total = head.length() + wavSize + tail.length();
  uint8_t* body = (uint8_t*)malloc(total);

  if (!body) {
    free(wav);
    Serial.println("Multipart malloc failed");
    return false;
  }

  size_t off = 0;
  memcpy(body + off, head.c_str(), head.length()); off += head.length();
  memcpy(body + off, wav, wavSize); off += wavSize;
  memcpy(body + off, tail.c_str(), tail.length());

  free(wav);

  WiFiClientSecure c;
  c.setInsecure();

  HTTPClient http;
  String url = String(BACKEND_URL) + "/voice";

  Serial.print("POST voice: ");
  Serial.println(url);

  if (!http.begin(c, url)) {
    free(body);
    Serial.println("HTTP begin failed");
    return false;
  }

  http.setTimeout(40000);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  int code = http.POST(body, total);
  free(body);

  String response = http.getString();
  http.end();

  Serial.print("Voice HTTP code: ");
  Serial.println(code);
  Serial.println(response);

  if (code < 200 || code >= 300) return false;

  StaticJsonDocument<2048> doc;
  if (deserializeJson(doc, response)) {
    Serial.println("Voice JSON parse failed");
    return false;
  }

  String command = doc["command"] | "none";
  int value = doc["value"] | -1;
  String reply = doc["reply"] | "";
  String audioUrl = doc["audio_url"] | "";

  Serial.print("Heard: ");
  Serial.println((const char*)doc["heard_text"]);
  Serial.print("Reply: ");
  Serial.println(reply);
  Serial.print("Audio: ");
  Serial.println(audioUrl);

  if (command == "lamp_on") {
    if (brightness == 0) brightness = 100;
    setLamp(true);
  } else if (command == "lamp_off") {
    setLamp(false);
  } else if (command == "set_brightness") {
    brightness = constrain(value, 0, 100);
    setLamp(value > 0);
    pwmWritePercent(brightness);
  } else if (command == "brighter") {
    brightness = min(100, brightness + 10);
    setLamp(true);
  } else if (command == "dimmer") {
    brightness = brightness > 10 ? brightness - 10 : 0;
    setLamp(brightness > 0);
  }

  if (audioUrl.length() > 0) playWavUrl(audioUrl);

  return true;
}

void sendAlert(const String& type) {
  if (WiFi.status() != WL_CONNECTED) return;

  StaticJsonDocument<128> req;
  req["type"] = type;

  String res;
  if (!httpPostJson("/device/alert", req, &res)) return;

  StaticJsonDocument<512> doc;
  if (deserializeJson(doc, res)) return;

  String audioUrl = doc["audio_url"] | "";

  if (audioUrl.length() > 0) playWavUrl(audioUrl);
}

// ================= WIFI PORTAL =================

String portalHtml() {
  String html;
  html += "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>";
  html += "<style>body{font-family:Arial;padding:20px}.box{max-width:500px;margin:auto}input,button{width:100%;padding:12px;margin:8px 0}</style></head><body><div class='box'>";
  html += "<h2>Smart Lamp WiFi</h2>";
  html += "<p>AP: " + htmlEscape(apSsid) + "</p>";
  html += "<p>PASS: " + String(CONFIG_AP_PASS) + "</p>";

  if (WiFi.status() == WL_CONNECTED) {
    html += "<p>WiFi OK: " + WiFi.SSID() + " IP: " + WiFi.localIP().toString() + "</p>";
  } else {
    html += "<p>WiFi chưa kết nối</p>";
  }

  html += "<form method='POST' action='/save'>";
  html += "<input name='ssid' placeholder='Ten WiFi' value='" + htmlEscape(savedSsid) + "'>";
  html += "<input name='pass' placeholder='Mat khau WiFi' type='password'>";
  html += "<button>Luu WiFi</button></form>";
  html += "<a href='/voice'>Test voice 2 giay</a><br><br>";
  html += "<a href='/clear'>Xoa WiFi</a>";
  html += "</div></body></html>";
  return html;
}

void startWifiConnect() {
  if (savedSsid.length() == 0) return;

  WiFi.mode(WIFI_AP_STA);
  WiFi.begin(savedSsid.c_str(), savedPass.c_str());

  wifiConnecting = true;
  wifiStartMs = millis();
  lastWifiRetryMs = millis();

  Serial.print("Trying WiFi: ");
  Serial.println(savedSsid);
}

void setupPortal() {
  server.on("/", HTTP_GET, []() {
    server.send(200, "text/html; charset=utf-8", portalHtml());
  });

  server.on("/save", HTTP_POST, []() {
    String ssid = server.arg("ssid");
    String pass = server.arg("pass");

    ssid.trim();

    if (ssid.length() == 0) {
      server.send(400, "text/plain", "SSID empty");
      return;
    }

    if (pass.length() == 0 && ssid == savedSsid) pass = savedPass;

    saveWifi(ssid, pass);
    startWifiConnect();

    server.send(200, "text/html; charset=utf-8", "<h2>Da luu WiFi</h2><a href='/'>Quay lai</a>");
  });

  server.on("/clear", HTTP_GET, []() {
    clearWifi();
    WiFi.disconnect();
    server.send(200, "text/html; charset=utf-8", "<h2>Da xoa WiFi</h2><a href='/'>Quay lai</a>");
  });

  server.on("/voice", HTTP_GET, []() {
    bool ok = uploadVoice();
    server.send(
      ok ? 200 : 500,
      "text/html; charset=utf-8",
      ok ? "<h2>OK</h2><a href='/'>Quay lai</a>" : "<h2>Loi voice</h2><a href='/'>Quay lai</a>"
    );
  });
}

void startAp() {
  apSsid = makeApSsid();

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAPConfig(IPAddress(192,168,4,1), IPAddress(192,168,4,1), IPAddress(255,255,255,0));
  WiFi.softAP(apSsid.c_str(), CONFIG_AP_PASS);

  Serial.println("=== AP STARTED ===");
  Serial.println(apSsid);
  Serial.println("http://192.168.4.1");
}

void maintainWifi() {
  if (savedSsid.length() == 0) return;

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnecting = false;
    return;
  }

  uint32_t now = millis();

  if (wifiConnecting && now - wifiStartMs > 20000) {
    wifiConnecting = false;
    lastWifiRetryMs = now;
  }

  if (!wifiConnecting && now - lastWifiRetryMs > 10000) {
    startWifiConnect();
  }
}

// ================= HCSR04 =================

void updateDistance() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  uint32_t duration = pulseIn(PIN_ECHO, HIGH, 25000);

  if (duration > 0) {
    distanceCm = duration / 58.0;
  }
}

// ================= LOGIC =================

void logicControl() {
  uint32_t now = millis();

  if (!isnan(distanceCm)) {
    if (distanceCm < POSTURE_BAD_CM) {
      if (postureBadSinceMs == 0) postureBadSinceMs = now;

      if (now - postureBadSinceMs >= POSTURE_CONFIRM_MS) {
        postureBad = true;

        if (lastPostureAlertMs == 0 || now - lastPostureAlertMs > POSTURE_ALERT_REPEAT_MS) {
          lastPostureAlertMs = now;
          sendAlert("wrong_posture");
        }
      }
    } else if (distanceCm >= POSTURE_OK_CM) {
      postureBad = false;
      postureBadSinceMs = 0;
    }

    if (distanceCm > ABSENT_CM) {
      if (absentSinceMs == 0) absentSinceMs = now;

      if (now - absentSinceMs >= ABSENT_CONFIRM_MS) {
        present = false;

        if (lampPower && !noPersonAlertSent) {
          setLamp(false);
          sendAlert("no_person");
          noPersonAlertSent = true;
        }
      }
    } else {
      present = true;
      absentSinceMs = 0;
      noPersonAlertSent = false;
    }
  }
}

void readLux() {
  float v = lightMeter.readLightLevel();
  if (v >= 0 && v < 100000) lux = v;
}

void reportStatus() {
  StaticJsonDocument<384> doc;

  doc["power"] = lampPower;
  doc["brightness"] = brightness;

  if (!isnan(lux)) {
    doc["ambient_lux"] = lux;
  }

  if (!isnan(distanceCm)) {
    doc["distance_cm"] = distanceCm;
  }

  doc["present"] = present;
  doc["posture_bad"] = postureBad;

  httpPostJson("/device/report", doc);
}

void pollCommand() {
  StaticJsonDocument<1024> doc;
  if (!httpGetJson("/device/pull", doc)) return;

  String command = doc["command"] | "none";
  int value = doc["value"] | -1;
  String audioUrl = doc["audio_url"] | "";

  if (command == "lamp_on") {
    setLamp(true);
  } else if (command == "lamp_off") {
    setLamp(false);
  } else if (command == "set_brightness") {
    brightness = constrain(value, 0, 100);
    setLamp(value > 0);
    pwmWritePercent(brightness);
  } else if (command == "brighter") {
    brightness = min(100, brightness + 10);
    setLamp(true);
  } else if (command == "dimmer") {
    brightness = brightness > 10 ? brightness - 10 : 0;
    setLamp(brightness > 0);
  }

  if (audioUrl.length() > 0) playWavUrl(audioUrl);
}

void debugPrint() {
  Serial.print("STA=");

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(WiFi.SSID());
    Serial.print("@");
    Serial.print(WiFi.localIP());
  } else {
    Serial.print("NO");
  }

  Serial.print(" AP=");
  Serial.print(apSsid);
  Serial.print(" Power=");
  Serial.print(lampPower);
  Serial.print(" Bright=");
  Serial.print(brightness);
  Serial.print(" Lux=");
  Serial.print(lux);
  Serial.print(" Dist=");
  Serial.print(distanceCm);
  Serial.print(" Present=");
  Serial.print(present);
  Serial.print(" PostureBad=");
  Serial.println(postureBad);
}

void serialCmd() {
  static String line;

  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      line.trim();

      if (line == "voice") uploadVoice();
      else if (line == "on") setLamp(true);
      else if (line == "off") setLamp(false);
      else if (line.startsWith("b")) {
        brightness = constrain(line.substring(1).toInt(), 0, 100);
        setLamp(brightness > 0);
        pwmWritePercent(brightness);
      }

      line = "";
    } else {
      line += c;
    }
  }
}

// ================= SETUP LOOP =================

void setup() {
  Serial.begin(115200);

  pinMode(PIN_RELAY, OUTPUT);
  digitalWrite(PIN_RELAY, relayLevel(false));

  pinMode(PIN_PWM_LED, OUTPUT);
  pwmBegin();
  brightness = 100;
  setLamp(false);

  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  delay(200);

  if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, 0x23, &Wire)) {
    Serial.println("BH1750 ready");
  } else {
    Serial.println("BH1750 failed");
  }

  WiFi.persistent(false);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  secureClient.setInsecure();

  speakerBegin();
  micBegin();

  setupPortal();
  startAp();
  loadWifi();

  if (savedSsid.length() > 0) startWifiConnect();

  server.begin();

  Serial.println("Smart Study Lamp ESP32-S3 VOICE started");
  Serial.println("Commands: voice | on | off | b50");
}

void loop() {
  server.handleClient();

  if (tWifi.due()) maintainWifi();
  if (tDistance.due()) updateDistance();
  if (tLux.due()) readLux();
  if (tLogic.due()) logicControl();
  if (tReport.due()) reportStatus();
  if (tPull.due()) pollCommand();
  if (tDebug.due()) debugPrint();

  serialCmd();
}
