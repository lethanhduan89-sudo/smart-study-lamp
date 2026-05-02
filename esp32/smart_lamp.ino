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

#if __has_include(<esp_arduino_version.h>)
  #include <esp_arduino_version.h>
#else
  #define ESP_ARDUINO_VERSION_MAJOR 2
#endif

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

// ===================== PIN MAP ESP32-S3 44PIN =====================

// BH1750
constexpr uint8_t PIN_I2C_SDA = 8;
constexpr uint8_t PIN_I2C_SCL = 9;

// HC-SR04
constexpr uint8_t PIN_HCSR04_TRIG = 4;
constexpr uint8_t PIN_HCSR04_ECHO = 5;

// PWM MOS + Relay
constexpr uint8_t PIN_PWM_LED = 6;
constexpr uint8_t PIN_RELAY   = 7;

// INMP441
constexpr int PIN_MIC_SCK = 15;
constexpr int PIN_MIC_WS  = 16;
constexpr int PIN_MIC_SD  = 17;

// MAX98357A
constexpr int PIN_SPK_BCLK = 18;
constexpr int PIN_SPK_LRC  = 21;
constexpr int PIN_SPK_DIN  = 47;

constexpr bool RELAY_ACTIVE_LOW = true;

// ===================== PWM =====================

constexpr uint32_t PWM_FREQ = 20000;
constexpr uint8_t PWM_RES_BITS = 10;
constexpr uint16_t PWM_MAX = (1 << PWM_RES_BITS) - 1;

#if ESP_ARDUINO_VERSION_MAJOR < 3
constexpr uint8_t PWM_CHANNEL_LED = 0;
#endif

// ===================== WIFI CONFIG =====================

Preferences preferences;
WebServer configServer(80);

String savedWifiSsid;
String savedWifiPass;
String configApSsid;

bool staConnecting = false;
uint32_t staConnectStartedMs = 0;
uint32_t lastStaRetryMs = 0;

constexpr uint32_t WIFI_CONNECT_TIMEOUT_MS = 20000;
constexpr uint32_t WIFI_RETRY_INTERVAL_MS  = 10000;

IPAddress AP_IP(192, 168, 4, 1);
IPAddress AP_GW(192, 168, 4, 1);
IPAddress AP_SUBNET(255, 255, 255, 0);

// ===================== SENSOR LOGIC =====================

constexpr float POSTURE_BAD_CM = 30.0f;
constexpr float POSTURE_OK_CM  = 35.0f;
constexpr float ABSENT_CM      = 50.0f;

constexpr uint32_t POSTURE_CONFIRM_MS = 2000;
constexpr uint32_t ABSENT_CONFIRM_MS  = 5000;
constexpr uint32_t HCSR04_STALE_MS    = 1500;
constexpr uint32_t POSTURE_ALERT_REPEAT_MS = 15000;

constexpr float TARGET_LUX_AUTO = 450.0f;
constexpr uint8_t MIN_BRIGHTNESS = 5;
constexpr uint8_t MAX_BRIGHTNESS = 100;

// ===================== AUDIO CONFIG =====================

constexpr int MIC_SAMPLE_RATE = 8000;
constexpr int RECORD_SECONDS = 2;
constexpr int MIC_BITS = 16;
constexpr int MIC_SAMPLES = MIC_SAMPLE_RATE * RECORD_SECONDS;

constexpr i2s_port_t I2S_MIC_PORT = I2S_NUM_0;
constexpr i2s_port_t I2S_SPK_PORT = I2S_NUM_1;

// ===================== GLOBAL STATE =====================

BH1750 lightMeter;
WiFiClientSecure secureClient;

volatile uint32_t echoRiseUs = 0;
volatile uint32_t echoPulseUs = 0;
volatile bool echoReady = false;

enum class LampMode : uint8_t {
  AUTO,
  MANUAL
};

struct LampState {
  bool power = false;
  uint8_t brightness = 50;
  LampMode mode = LampMode::AUTO;

  float lux = NAN;
  float distanceCm = NAN;

  bool present = false;
  bool postureBad = false;

  uint32_t absentSinceMs = 0;
  uint32_t postureBadSinceMs = 0;
  uint32_t lastPostureAlertMs = 0;
  uint32_t lastDistanceMs = 0;
};

LampState lamp;

struct TaskTimer {
  uint32_t last = 0;
  uint32_t period = 1000;

  bool due() {
    uint32_t now = millis();
    if (now - last >= period) {
      last = now;
      return true;
    }
    return false;
  }
};

TaskTimer taskWifi     {0, 3000};
TaskTimer taskLux      {0, 1000};
TaskTimer taskTrig     {0, 80};
TaskTimer taskDistance {0, 20};
TaskTimer taskControl  {0, 500};
TaskTimer taskReport   {0, 5000};
TaskTimer taskCommand  {0, 1500};
TaskTimer taskDebug    {0, 2000};

// ===================== FORWARD =====================

void applyCommand(const String& command, int value);
bool uploadVoiceToBackend();
bool playWavUrl(const String& url);

// ===================== HELPERS =====================

bool relayLevel(bool on) {
  return RELAY_ACTIVE_LOW ? !on : on;
}

const char* modeToText(LampMode mode) {
  switch (mode) {
    case LampMode::AUTO:   return "auto";
    case LampMode::MANUAL: return "manual";
    default:               return "auto";
  }
}

String htmlEscape(String s) {
  s.replace("&", "&amp;");
  s.replace("<", "&lt;");
  s.replace(">", "&gt;");
  s.replace("\"", "&quot;");
  s.replace("'", "&#39;");
  return s;
}

String makeApSsid() {
  uint64_t chipId = ESP.getEfuseMac();
  char buf[32];
  snprintf(buf, sizeof(buf), "%s-%04X", CONFIG_AP_SSID_PREFIX, (uint16_t)(chipId & 0xFFFF));
  return String(buf);
}

uint16_t readLe16(const uint8_t* p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

uint32_t readLe32(const uint8_t* p) {
  return (uint32_t)p[0] |
         ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

// ===================== PWM / RELAY =====================

void pwmBegin() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PIN_PWM_LED, PWM_FREQ, PWM_RES_BITS);
#else
  ledcSetup(PWM_CHANNEL_LED, PWM_FREQ, PWM_RES_BITS);
  ledcAttachPin(PIN_PWM_LED, PWM_CHANNEL_LED);
#endif
}

void pwmWriteRaw(uint16_t duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PIN_PWM_LED, duty);
#else
  ledcWrite(PWM_CHANNEL_LED, duty);
#endif
}

void writeLampPwm(uint8_t percent) {
  percent = constrain(percent, 0, 100);

  uint16_t duty = 0;

  if (lamp.power) {
    duty = map(percent, 0, 100, 0, PWM_MAX);
  }

  pwmWriteRaw(duty);
}

void setRelay(bool on) {
  lamp.power = on;
  digitalWrite(PIN_RELAY, relayLevel(on));
  writeLampPwm(lamp.brightness);
}

void setBrightness(uint8_t percent) {
  lamp.brightness = constrain(percent, 0, 100);
  writeLampPwm(lamp.brightness);
}

// ===================== WIFI STORAGE =====================

void loadWifiConfig() {
  preferences.begin("wifi_cfg", true);
  savedWifiSsid = preferences.getString("ssid", "");
  savedWifiPass = preferences.getString("pass", "");
  preferences.end();
}

void saveWifiConfig(const String& ssid, const String& pass) {
  preferences.begin("wifi_cfg", false);
  preferences.putString("ssid", ssid);
  preferences.putString("pass", pass);
  preferences.end();

  savedWifiSsid = ssid;
  savedWifiPass = pass;
}

void clearWifiConfig() {
  preferences.begin("wifi_cfg", false);
  preferences.clear();
  preferences.end();

  savedWifiSsid = "";
  savedWifiPass = "";
}

// ===================== I2S SPEAKER WAV =====================

void speakerBegin() {
  i2s_config_t config = {};
  config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  config.sample_rate = 24000;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  config.dma_buf_count = 8;
  config.dma_buf_len = 512;
  config.use_apll = false;
  config.tx_desc_auto_clear = true;
  config.fixed_mclk = 0;

  i2s_pin_config_t pins = {};
  pins.bck_io_num = PIN_SPK_BCLK;
  pins.ws_io_num = PIN_SPK_LRC;
  pins.data_out_num = PIN_SPK_DIN;
  pins.data_in_num = I2S_PIN_NO_CHANGE;

  i2s_driver_install(I2S_SPK_PORT, &config, 0, NULL);
  i2s_set_pin(I2S_SPK_PORT, &pins);
  i2s_zero_dma_buffer(I2S_SPK_PORT);

  Serial.println("MAX98357A WAV speaker ready");
}

bool playWavUrl(const String& url) {
  if (url.length() == 0) return false;

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, cannot play WAV");
    return false;
  }

  Serial.print("Playing WAV: ");
  Serial.println(url);

  WiFiClientSecure audioClient;
  audioClient.setInsecure();

  HTTPClient http;

  if (!http.begin(audioClient, url)) {
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

  uint8_t header[44];
  size_t got = stream->readBytes(header, 44);

  if (got != 44) {
    Serial.println("WAV header read failed");
    http.end();
    return false;
  }

  if (memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0) {
    Serial.println("Invalid WAV");
    http.end();
    return false;
  }

  uint16_t channels = readLe16(header + 22);
  uint32_t sampleRate = readLe32(header + 24);
  uint16_t bits = readLe16(header + 34);
  uint32_t dataSize = readLe32(header + 40);

  Serial.print("WAV rate=");
  Serial.print(sampleRate);
  Serial.print(" channels=");
  Serial.print(channels);
  Serial.print(" bits=");
  Serial.print(bits);
  Serial.print(" data=");
  Serial.println(dataSize);

  if (bits != 16) {
    Serial.println("Only 16-bit WAV supported");
    http.end();
    return false;
  }

  i2s_set_clk(
    I2S_SPK_PORT,
    sampleRate,
    I2S_BITS_PER_SAMPLE_16BIT,
    I2S_CHANNEL_MONO
  );

  uint8_t buffer[1024];
  uint32_t remaining = dataSize;

  while (http.connected() && remaining > 0) {
    int available = stream->available();

    if (available <= 0) {
      delay(1);
      configServer.handleClient();
      continue;
    }

    int toRead = min((uint32_t)sizeof(buffer), remaining);
    int n = stream->readBytes(buffer, min(toRead, available));

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

    remaining -= n;
    configServer.handleClient();
    yield();
  }

  i2s_zero_dma_buffer(I2S_SPK_PORT);
  http.end();

  Serial.println("WAV finished");
  return true;
}

// ===================== I2S MICROPHONE =====================

void micBegin() {
  i2s_config_t config = {};
  config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  config.sample_rate = MIC_SAMPLE_RATE;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  config.dma_buf_count = 8;
  config.dma_buf_len = 256;
  config.use_apll = false;
  config.tx_desc_auto_clear = false;
  config.fixed_mclk = 0;

  i2s_pin_config_t pins = {};
  pins.bck_io_num = PIN_MIC_SCK;
  pins.ws_io_num = PIN_MIC_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = PIN_MIC_SD;

  i2s_driver_install(I2S_MIC_PORT, &config, 0, NULL);
  i2s_set_pin(I2S_MIC_PORT, &pins);
  i2s_zero_dma_buffer(I2S_MIC_PORT);

  Serial.println("INMP441 mic ready");
}

void writeWavHeader(uint8_t* header, uint32_t dataSize) {
  uint32_t fileSize = dataSize + 36;
  uint32_t byteRate = MIC_SAMPLE_RATE * 1 * MIC_BITS / 8;
  uint16_t blockAlign = 1 * MIC_BITS / 8;

  memcpy(header, "RIFF", 4);
  memcpy(header + 4, &fileSize, 4);
  memcpy(header + 8, "WAVE", 4);
  memcpy(header + 12, "fmt ", 4);

  uint32_t subchunk1Size = 16;
  uint16_t audioFormat = 1;
  uint16_t numChannels = 1;
  uint32_t sampleRate = MIC_SAMPLE_RATE;
  uint16_t bitsPerSample = MIC_BITS;

  memcpy(header + 16, &subchunk1Size, 4);
  memcpy(header + 20, &audioFormat, 2);
  memcpy(header + 22, &numChannels, 2);
  memcpy(header + 24, &sampleRate, 4);
  memcpy(header + 28, &byteRate, 4);
  memcpy(header + 32, &blockAlign, 2);
  memcpy(header + 34, &bitsPerSample, 2);

  memcpy(header + 36, "data", 4);
  memcpy(header + 40, &dataSize, 4);
}

uint8_t* recordWav(size_t& wavSize) {
  const size_t pcmBytes = MIC_SAMPLES * sizeof(int16_t);
  wavSize = 44 + pcmBytes;

  uint8_t* wav = (uint8_t*)malloc(wavSize);

  if (!wav) {
    Serial.println("Audio malloc failed");
    return nullptr;
  }

  writeWavHeader(wav, pcmBytes);

  int16_t* pcm = (int16_t*)(wav + 44);

  int32_t rawBuffer[256];
  int sampleIndex = 0;

  Serial.println("Recording voice for 2 seconds...");

  while (sampleIndex < MIC_SAMPLES) {
    size_t bytesRead = 0;

    i2s_read(
      I2S_MIC_PORT,
      rawBuffer,
      sizeof(rawBuffer),
      &bytesRead,
      portMAX_DELAY
    );

    int count = bytesRead / sizeof(int32_t);

    for (int i = 0; i < count && sampleIndex < MIC_SAMPLES; i++) {
      int32_t raw = rawBuffer[i];
      int16_t sample = (int16_t)(raw >> 14);
      pcm[sampleIndex++] = sample;
    }

    configServer.handleClient();
    yield();
  }

  Serial.println("Voice recorded");
  return wav;
}

// ===================== UPLOAD VOICE =====================

bool uploadVoiceToBackend() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, cannot upload voice");
    return false;
  }

  size_t wavSize = 0;
  uint8_t* wav = recordWav(wavSize);

  if (!wav) {
    Serial.println("Record WAV failed");
    return false;
  }

  String boundary = "----SmartLampBoundary";

  String head =
    "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"file\"; filename=\"voice.wav\"\r\n"
    "Content-Type: audio/wav\r\n\r\n";

  String tail = "\r\n--" + boundary + "--\r\n";

  size_t totalLen = head.length() + wavSize + tail.length();

  uint8_t* body = (uint8_t*)malloc(totalLen);

  if (!body) {
    Serial.println("Multipart malloc failed");
    free(wav);
    return false;
  }

  size_t offset = 0;

  memcpy(body + offset, head.c_str(), head.length());
  offset += head.length();

  memcpy(body + offset, wav, wavSize);
  offset += wavSize;

  memcpy(body + offset, tail.c_str(), tail.length());
  offset += tail.length();

  free(wav);

  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  String url = String(BACKEND_URL) + "/voice";

  Serial.print("POST voice to: ");
  Serial.println(url);

  if (!http.begin(client, url)) {
    Serial.println("HTTP begin failed");
    free(body);
    return false;
  }

  http.setTimeout(40000);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  int code = http.POST(body, totalLen);

  free(body);

  String response = http.getString();
  http.end();

  Serial.print("Voice HTTP code: ");
  Serial.println(code);
  Serial.println(response);

  if (code < 200 || code >= 300) {
    return false;
  }

  StaticJsonDocument<2048> doc;
  DeserializationError err = deserializeJson(doc, response);

  if (err) {
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

  Serial.print("Audio URL: ");
  Serial.println(audioUrl);

  applyCommand(command, value);

  if (audioUrl.length() > 0) {
    playWavUrl(audioUrl);
  }

  return true;
}

// ===================== HTTP JSON =====================

bool httpPostJson(const String& path, const JsonDocument& doc) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  String url = String(BACKEND_URL) + path;

  if (!http.begin(secureClient, url)) return false;

  http.setTimeout(6000);
  http.addHeader("Content-Type", "application/json");

  String body;
  serializeJson(doc, body);

  int code = http.POST(body);
  http.end();

  return code >= 200 && code < 300;
}

bool httpGetJson(const String& path, JsonDocument& doc) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  String url = String(BACKEND_URL) + path;

  if (!http.begin(secureClient, url)) return false;

  http.setTimeout(6000);

  int code = http.GET();

  if (code != 200) {
    http.end();
    return false;
  }

  String payload = http.getString();
  http.end();

  return deserializeJson(doc, payload) == DeserializationError::Ok;
}

// ===================== WIFI PORTAL =====================

String routerStatusText() {
  if (WiFi.status() == WL_CONNECTED) {
    return "Đã kết nối WiFi: " + WiFi.SSID() + " | IP: " + WiFi.localIP().toString();
  }

  if (staConnecting) return "Đang thử kết nối tới WiFi: " + savedWifiSsid;
  if (savedWifiSsid.length() > 0) return "Chưa kết nối được WiFi đã lưu: " + savedWifiSsid;

  return "Chưa có WiFi nào được lưu.";
}

String configPageHtml() {
  String html;
  html.reserve(5200);

  html += "<!DOCTYPE html><html lang='vi'><head>";
  html += "<meta charset='UTF-8'>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<title>Smart Lamp WiFi Setup</title>";
  html += "<style>";
  html += "body{font-family:Arial;background:#f4f6f8;padding:20px;}";
  html += ".box{max-width:640px;margin:auto;background:#fff;padding:20px;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.08);}";
  html += "input{width:100%;padding:12px;margin:8px 0 16px;border:1px solid #ccc;border-radius:8px;box-sizing:border-box;}";
  html += "button,a.btn{display:inline-block;padding:12px 16px;border:none;border-radius:8px;background:#2563eb;color:#fff;text-decoration:none;cursor:pointer;margin:4px 0;}";
  html += ".danger{background:#dc2626;} .green{background:#16a34a;} .line{margin-bottom:10px;} .muted{color:#555;font-size:14px;}";
  html += "</style></head><body><div class='box'>";

  html += "<h2>Cấu hình WiFi cho đèn học thông minh</h2>";
  html += "<div class='line'><b>WiFi ESP32:</b> " + htmlEscape(configApSsid) + "</div>";
  html += "<div class='line'><b>Mật khẩu:</b> " + htmlEscape(String(CONFIG_AP_PASS)) + "</div>";
  html += "<div class='line'><b>IP:</b> 192.168.4.1</div>";
  html += "<div class='line'><b>Trạng thái:</b> " + htmlEscape(routerStatusText()) + "</div>";

  html += "<hr>";
  html += "<form method='POST' action='/save'>";
  html += "<label><b>Tên WiFi nhà bạn</b></label>";
  html += "<input name='ssid' placeholder='Ví dụ: TP-Link_2.4G' value='" + htmlEscape(savedWifiSsid) + "' required>";
  html += "<label><b>Mật khẩu WiFi</b></label>";
  html += "<input name='password' type='password' placeholder='Nhập mật khẩu WiFi'>";
  html += "<button type='submit'>Lưu WiFi</button>";
  html += "</form>";

  html += "<hr>";
  html += "<h3>Test AI Voice WAV</h3>";
  html += "<p class='muted'>ESP32-S3 thu âm 2 giây bằng INMP441, gửi Render, nhận WAV và phát qua MAX98357A.</p>";
  html += "<a class='btn green' href='/voice-test'>Thu âm 2 giây gửi AI</a><br>";
  html += "<a class='btn danger' href='/clear'>Xóa WiFi đã lưu</a>";

  html += "</div></body></html>";

  return html;
}

void handleRootPage() {
  configServer.sendHeader("Cache-Control", "no-store");
  configServer.send(200, "text/html; charset=utf-8", configPageHtml());
}

void startStaConnect() {
  if (savedWifiSsid.length() == 0) {
    staConnecting = false;
    return;
  }

  WiFi.mode(WIFI_AP_STA);
  WiFi.disconnect();
  delay(100);
  WiFi.begin(savedWifiSsid.c_str(), savedWifiPass.c_str());

  staConnecting = true;
  staConnectStartedMs = millis();
  lastStaRetryMs = millis();

  Serial.print("Trying WiFi: ");
  Serial.println(savedWifiSsid);
}

void handleSaveWifi() {
  String ssid = configServer.arg("ssid");
  String password = configServer.arg("password");

  ssid.trim();

  if (ssid.length() == 0) {
    configServer.send(400, "text/plain; charset=utf-8", "SSID không được để trống.");
    return;
  }

  if (password.length() == 0 && ssid == savedWifiSsid && savedWifiPass.length() > 0) {
    password = savedWifiPass;
  }

  saveWifiConfig(ssid, password);
  startStaConnect();

  configServer.send(
    200,
    "text/html; charset=utf-8",
    "<html><body><h2>Đã lưu WiFi</h2><p>Đợi 10-20 giây rồi tải lại 192.168.4.1</p><a href='/'>Quay lại</a></body></html>"
  );
}

void handleClearWifi() {
  clearWifiConfig();
  WiFi.disconnect();
  staConnecting = false;

  configServer.send(
    200,
    "text/html; charset=utf-8",
    "<html><body><h2>Đã xóa WiFi</h2><a href='/'>Quay lại</a></body></html>"
  );
}

void handleVoiceTest() {
  bool ok = uploadVoiceToBackend();

  if (ok) {
    configServer.send(
      200,
      "text/html; charset=utf-8",
      "<html><body><h2>Đã gửi voice lên AI</h2><p>Xem Serial Monitor để biết kết quả.</p><a href='/'>Quay lại</a></body></html>"
    );
  } else {
    configServer.send(
      500,
      "text/html; charset=utf-8",
      "<html><body><h2>Lỗi gửi voice</h2><p>Kiểm tra WiFi, Render, OPENAI_API_KEY, INMP441.</p><a href='/'>Quay lại</a></body></html>"
    );
  }
}

void setupPortalServer() {
  configServer.on("/", HTTP_GET, handleRootPage);
  configServer.on("/save", HTTP_POST, handleSaveWifi);
  configServer.on("/clear", HTTP_GET, handleClearWifi);
  configServer.on("/voice-test", HTTP_GET, handleVoiceTest);
  configServer.onNotFound(handleRootPage);
}

void startConfigAp() {
  configApSsid = makeApSsid();

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAPConfig(AP_IP, AP_GW, AP_SUBNET);
  WiFi.softAP(configApSsid.c_str(), CONFIG_AP_PASS);

  Serial.println();
  Serial.println("=== CONFIG AP STARTED ===");
  Serial.print("AP SSID: ");
  Serial.println(configApSsid);
  Serial.print("AP PASS: ");
  Serial.println(CONFIG_AP_PASS);
  Serial.println("Open: http://192.168.4.1");
}

void maintainWifi() {
  if (savedWifiSsid.length() == 0) return;

  if (WiFi.status() == WL_CONNECTED) {
    staConnecting = false;
    return;
  }

  uint32_t now = millis();

  if (staConnecting) {
    if (now - staConnectStartedMs >= WIFI_CONNECT_TIMEOUT_MS) {
      staConnecting = false;
      lastStaRetryMs = now;
      Serial.println("WiFi connect timeout");
    }

    return;
  }

  if (now - lastStaRetryMs >= WIFI_RETRY_INTERVAL_MS) {
    startStaConnect();
  }
}

// ===================== HC-SR04 =====================

void IRAM_ATTR onEchoChange() {
  uint32_t now = micros();

  if (digitalRead(PIN_HCSR04_ECHO)) {
    echoRiseUs = now;
  } else {
    echoPulseUs = now - echoRiseUs;
    echoReady = true;
  }
}

void triggerUltrasonic() {
  digitalWrite(PIN_HCSR04_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_HCSR04_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_HCSR04_TRIG, LOW);
}

void confirmAbsentCandidate() {
  uint32_t now = millis();

  if (lamp.absentSinceMs == 0) lamp.absentSinceMs = now;

  if (now - lamp.absentSinceMs >= ABSENT_CONFIRM_MS) {
    lamp.present = false;
    lamp.postureBad = false;
    lamp.postureBadSinceMs = 0;
  }
}

void updatePresenceAndPosture(float cm) {
  uint32_t now = millis();

  lamp.distanceCm = cm;
  lamp.lastDistanceMs = now;

  if (cm > ABSENT_CM) {
    confirmAbsentCandidate();
  } else {
    lamp.present = true;
    lamp.absentSinceMs = 0;
  }

  if (!lamp.present) return;

  if (cm < POSTURE_BAD_CM) {
    if (lamp.postureBadSinceMs == 0) lamp.postureBadSinceMs = now;

    if (now - lamp.postureBadSinceMs >= POSTURE_CONFIRM_MS) {
      lamp.postureBad = true;
    }
  } else if (cm >= POSTURE_OK_CM) {
    lamp.postureBad = false;
    lamp.postureBadSinceMs = 0;
  }
}

void updateNoEchoAsAbsent() {
  uint32_t now = millis();

  if (lamp.lastDistanceMs == 0) return;

  if (now - lamp.lastDistanceMs > HCSR04_STALE_MS) {
    lamp.distanceCm = NAN;
    confirmAbsentCandidate();
  }
}

void updateDistance() {
  bool ready = false;
  uint32_t pulse = 0;

  noInterrupts();

  if (echoReady) {
    ready = true;
    pulse = echoPulseUs;
    echoReady = false;
  }

  interrupts();

  if (!ready) {
    updateNoEchoAsAbsent();
    return;
  }

  if (pulse < 100 || pulse > 25000) {
    updateNoEchoAsAbsent();
    return;
  }

  float cm = pulse / 58.0f;
  updatePresenceAndPosture(cm);
}

// ===================== SENSOR + CONTROL =====================

void readLux() {
  float lux = lightMeter.readLightLevel();

  if (lux >= 0.0f && lux < 100000.0f) {
    lamp.lux = lux;
  }
}

void handlePostureAlert() {
  if (!lamp.postureBad) {
    lamp.lastPostureAlertMs = 0;
    return;
  }

  uint32_t now = millis();

  if (lamp.lastPostureAlertMs == 0 || now - lamp.lastPostureAlertMs >= POSTURE_ALERT_REPEAT_MS) {
    lamp.lastPostureAlertMs = now;
    Serial.println("[POSTURE] Sai tu the: ban dang cui qua thap.");
  }
}

void autoControl() {
  if (!lamp.present) {
    if (lamp.power) setRelay(false);
    return;
  }

  if (!lamp.power) setRelay(true);

  handlePostureAlert();

  if (lamp.mode == LampMode::MANUAL) return;
  if (isnan(lamp.lux)) return;

  float error = TARGET_LUX_AUTO - lamp.lux;
  int nextBrightness = lamp.brightness + (int)(error * 0.03f);
  nextBrightness = constrain(nextBrightness, MIN_BRIGHTNESS, MAX_BRIGHTNESS);

  setBrightness((uint8_t)nextBrightness);
}

// ===================== BACKEND SYNC =====================

void sendReport() {
  StaticJsonDocument<512> doc;

  doc["power"] = lamp.power;
  doc["brightness"] = lamp.brightness;
  doc["auto_mode"] = lamp.mode == LampMode::AUTO;
  doc["mode"] = modeToText(lamp.mode);
  doc["present"] = lamp.present;
  doc["posture_bad"] = lamp.postureBad;
  doc["alert"] = lamp.postureBad ? "wrong_posture" : "none";

  if (!isnan(lamp.lux)) doc["ambient_lux"] = lamp.lux;
  if (!isnan(lamp.distanceCm)) doc["distance_cm"] = lamp.distanceCm;

  httpPostJson("/device/report", doc);
}

void applyCommand(const String& command, int value) {
  if (command.length() == 0 || command == "none") return;

  if (command == "lamp_on") {
    lamp.mode = LampMode::MANUAL;
    if (lamp.brightness == 0) setBrightness(50);
    setRelay(true);
  }
  else if (command == "lamp_off") {
    lamp.mode = LampMode::MANUAL;
    setRelay(false);
  }
  else if (command == "brighter") {
    lamp.mode = LampMode::MANUAL;
    setRelay(true);
    setBrightness(constrain((int)lamp.brightness + 10, 0, 100));
  }
  else if (command == "dimmer") {
    lamp.mode = LampMode::MANUAL;
    setBrightness(constrain((int)lamp.brightness - 10, 0, 100));

    if (lamp.brightness == 0) setRelay(false);
  }
  else if (command == "set_brightness") {
    lamp.mode = LampMode::MANUAL;

    int b = constrain(value, 0, 100);
    setBrightness((uint8_t)b);

    if (b == 0) setRelay(false);
    else setRelay(true);
  }
  else if (command == "auto_mode") {
    lamp.mode = LampMode::AUTO;
    setRelay(true);
  }
  else if (command == "manual_mode") {
    lamp.mode = LampMode::MANUAL;
    setRelay(true);
  }
}

void pollCommand() {
  StaticJsonDocument<1536> doc;

  if (!httpGetJson("/device/pull", doc)) return;

  String command = doc["command"] | "";
  int value = doc["value"] | -1;
  String audioUrl = doc["audio_url"] | "";

  applyCommand(command, value);

  if (audioUrl.length() > 0) {
    playWavUrl(audioUrl);
  }
}

// ===================== DEBUG =====================

void printStatus() {
  Serial.print("STA=");

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(WiFi.SSID());
    Serial.print("@");
    Serial.print(WiFi.localIP());
  } else {
    Serial.print("NO");
  }

  Serial.print(" AP=");
  Serial.print(configApSsid);

  Serial.print(" Power=");
  Serial.print(lamp.power);

  Serial.print(" Brightness=");
  Serial.print(lamp.brightness);

  Serial.print(" Mode=");
  Serial.print(modeToText(lamp.mode));

  Serial.print(" Lux=");
  Serial.print(lamp.lux);

  Serial.print(" Distance=");
  Serial.print(lamp.distanceCm);

  Serial.print(" Present=");
  Serial.print(lamp.present);

  Serial.print(" PostureBad=");
  Serial.println(lamp.postureBad);
}

void handleSerialCommand() {
  static String line;

  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      line.trim();

      if (line == "voice") {
        uploadVoiceToBackend();
      }
      else if (line == "on") {
        applyCommand("lamp_on", -1);
      }
      else if (line == "off") {
        applyCommand("lamp_off", -1);
      }
      else if (line == "auto") {
        applyCommand("auto_mode", -1);
      }

      line = "";
    } else {
      line += c;
    }
  }
}

// ===================== SETUP / LOOP =====================

void setup() {
  Serial.begin(115200);

  pinMode(PIN_RELAY, OUTPUT);
  digitalWrite(PIN_RELAY, relayLevel(false));

  pinMode(PIN_PWM_LED, OUTPUT);
  pwmBegin();
  writeLampPwm(0);

  pinMode(PIN_HCSR04_TRIG, OUTPUT);
  pinMode(PIN_HCSR04_ECHO, INPUT);
  digitalWrite(PIN_HCSR04_TRIG, LOW);

  attachInterrupt(digitalPinToInterrupt(PIN_HCSR04_ECHO), onEchoChange, CHANGE);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  delay(200);

  if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, 0x23, &Wire)) {
    Serial.println("BH1750 init failed at 0x23");
  } else {
    Serial.println("BH1750 ready at 0x23");
  }

  WiFi.persistent(false);
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);

  secureClient.setInsecure();

  speakerBegin();
  micBegin();

  setupPortalServer();
  startConfigAp();

  loadWifiConfig();

  if (savedWifiSsid.length() > 0) {
    startStaConnect();
  } else {
    Serial.println("No saved WiFi. Connect to ESP32 AP and open http://192.168.4.1");
  }

  configServer.begin();

  setRelay(false);
  setBrightness(50);

  Serial.println("Smart Study Lamp ESP32-S3 FULL WAV PCM started");
  Serial.println("Serial commands: voice | on | off | auto");
}

void loop() {
  configServer.handleClient();

  if (taskWifi.due()) maintainWifi();
  if (taskTrig.due()) triggerUltrasonic();
  if (taskDistance.due()) updateDistance();
  if (taskLux.due()) readLux();
  if (taskControl.due()) autoControl();
  if (taskReport.due()) sendReport();
  if (taskCommand.due()) pollCommand();
  if (taskDebug.due()) printStatus();

  handleSerialCommand();
}
