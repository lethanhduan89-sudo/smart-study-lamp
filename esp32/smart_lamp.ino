#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <BH1750.h>
#include <Preferences.h>
#include <WebServer.h>

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
  #define BACKEND_URL "https://REPLACE-WITH-YOUR-RENDER-URL.onrender.com"
#endif

#ifndef CONFIG_AP_SSID_PREFIX
  #define CONFIG_AP_SSID_PREFIX "SmartLamp"
#endif

#ifndef CONFIG_AP_PASS
  #define CONFIG_AP_PASS "12345678"
#endif

// ESP32 30P CH340 Type-C
// BH1750 SDA  -> GPIO21
// BH1750 SCL  -> GPIO22
// HC-SR04 TRIG -> GPIO16 / RX2
// HC-SR04 ECHO -> GPIO17 / TX2 qua chia áp 20k/10k
// PWM LED      -> GPIO23
// Relay        -> GPIO19

constexpr uint8_t PIN_I2C_SDA = 21;
constexpr uint8_t PIN_I2C_SCL = 22;

constexpr uint8_t PIN_HCSR04_TRIG = 16;
constexpr uint8_t PIN_HCSR04_ECHO = 17;

constexpr uint8_t PIN_PWM_LED = 23;
constexpr uint8_t PIN_RELAY   = 19;

constexpr bool RELAY_ACTIVE_LOW = true;

constexpr uint32_t PWM_FREQ = 20000;
constexpr uint8_t PWM_RES_BITS = 10;
constexpr uint16_t PWM_MAX = (1 << PWM_RES_BITS) - 1;

#if ESP_ARDUINO_VERSION_MAJOR < 3
constexpr uint8_t PWM_CHANNEL_LED = 0;
#endif

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

// <30cm: sai tư thế
// 30-50cm: có người
// >50cm: không có người, tắt đèn
constexpr float POSTURE_BAD_CM = 30.0f;
constexpr float POSTURE_OK_CM = 35.0f;
constexpr float ABSENT_CM = 50.0f;

constexpr uint32_t POSTURE_CONFIRM_MS = 2000;
constexpr uint32_t ABSENT_CONFIRM_MS  = 5000;
constexpr uint32_t HCSR04_STALE_MS    = 1500;
constexpr uint32_t POSTURE_ALERT_REPEAT_MS = 15000;

constexpr float TARGET_LUX_AUTO = 450.0f;
constexpr uint8_t MIN_BRIGHTNESS = 5;
constexpr uint8_t MAX_BRIGHTNESS = 100;

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

  uint32_t lastPresenceMs = 0;
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
  snprintf(
    buf,
    sizeof(buf),
    "%s-%04X",
    CONFIG_AP_SSID_PREFIX,
    (uint16_t)(chipId & 0xFFFF)
  );
  return String(buf);
}

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

String routerStatusText() {
  if (WiFi.status() == WL_CONNECTED) {
    return "Đã kết nối WiFi: " + WiFi.SSID() + " | IP: " + WiFi.localIP().toString();
  }

  if (staConnecting) {
    return "Đang thử kết nối tới WiFi: " + savedWifiSsid;
  }

  if (savedWifiSsid.length() > 0) {
    return "Chưa kết nối được WiFi đã lưu: " + savedWifiSsid;
  }

  return "Chưa có WiFi nào được lưu.";
}

String configPageHtml() {
  String html;
  html.reserve(3600);

  html += "<!DOCTYPE html><html lang='vi'><head>";
  html += "<meta charset='UTF-8'>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<title>Smart Lamp WiFi Setup</title>";
  html += "<style>";
  html += "body{font-family:Arial;background:#f4f6f8;padding:20px;}";
  html += ".box{max-width:560px;margin:auto;background:#fff;padding:20px;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.08);}";
  html += "h2{margin-top:0;}input{width:100%;padding:12px;margin:8px 0 16px;border:1px solid #ccc;border-radius:8px;box-sizing:border-box;}";
  html += "button,a.btn{display:inline-block;padding:12px 16px;border:none;border-radius:8px;background:#2563eb;color:#fff;text-decoration:none;cursor:pointer;}";
  html += ".muted{color:#555;font-size:14px;} .danger{background:#dc2626;} .line{margin-bottom:10px;}";
  html += "</style></head><body><div class='box'>";
  html += "<h2>Cấu hình WiFi cho đèn học thông minh</h2>";

  html += "<div class='line'><b>WiFi ESP32 đang phát:</b> ";
  html += htmlEscape(configApSsid);
  html += "</div>";

  html += "<div class='line'><b>Mật khẩu WiFi ESP32:</b> ";
  html += htmlEscape(String(CONFIG_AP_PASS));
  html += "</div>";

  html += "<div class='line'><b>IP cấu hình:</b> 192.168.4.1</div>";

  html += "<div class='line'><b>Trạng thái:</b> ";
  html += htmlEscape(routerStatusText());
  html += "</div>";

  html += "<hr>";

  html += "<form method='POST' action='/save'>";
  html += "<label><b>Tên WiFi nhà bạn (SSID)</b></label>";
  html += "<input name='ssid' placeholder='Ví dụ: TP-Link_2.4G' value='";
  html += htmlEscape(savedWifiSsid);
  html += "' required>";

  html += "<label><b>Mật khẩu WiFi</b></label>";
  html += "<input name='password' type='password' placeholder='Nhập mật khẩu WiFi'>";
  html += "<div class='muted'>Nếu giữ nguyên cùng SSID cũ, để trống sẽ giữ mật khẩu cũ.</div><br>";

  html += "<button type='submit'>Lưu và kết nối</button>";
  html += "</form>";

  html += "<br><a class='btn danger' href='/clear'>Xóa WiFi đã lưu</a>";
  html += "<p class='muted'>Sau khi bấm Lưu, chờ khoảng 10-20 giây rồi tải lại trang.</p>";

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

  String html;
  html += "<!DOCTYPE html><html lang='vi'><head><meta charset='UTF-8'>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<title>Đã lưu</title></head>";
  html += "<body style='font-family:Arial;padding:20px;'>";
  html += "<h2>Đã lưu WiFi thành công</h2>";
  html += "<p>ESP32 đang thử kết nối tới WiFi: <b>" + htmlEscape(ssid) + "</b></p>";
  html += "<p>Đợi khoảng 10-20 giây, sau đó mở lại <b>http://192.168.4.1</b> để xem trạng thái.</p>";
  html += "<p><a href='/'>Quay lại</a></p>";
  html += "</body></html>";

  configServer.send(200, "text/html; charset=utf-8", html);
}

void handleClearWifi() {
  clearWifiConfig();
  WiFi.disconnect();
  staConnecting = false;

  String html;
  html += "<!DOCTYPE html><html lang='vi'><head><meta charset='UTF-8'>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<title>Đã xóa</title></head>";
  html += "<body style='font-family:Arial;padding:20px;'>";
  html += "<h2>Đã xóa WiFi đã lưu</h2>";
  html += "<p>ESP32 sẽ tiếp tục phát WiFi cấu hình để bạn nhập WiFi mới.</p>";
  html += "<p><a href='/'>Quay lại</a></p>";
  html += "</body></html>";

  configServer.send(200, "text/html; charset=utf-8", html);
}

void setupPortalServer() {
  configServer.on("/", HTTP_GET, handleRootPage);
  configServer.on("/save", HTTP_POST, handleSaveWifi);
  configServer.on("/clear", HTTP_GET, handleClearWifi);
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
  if (savedWifiSsid.length() == 0) {
    return;
  }

  if (WiFi.status() == WL_CONNECTED) {
    staConnecting = false;
    return;
  }

  uint32_t now = millis();

  if (staConnecting) {
    if (now - staConnectStartedMs >= WIFI_CONNECT_TIMEOUT_MS) {
      staConnecting = false;
      lastStaRetryMs = now;
      Serial.println("WiFi connect timeout. AP is still available for re-config.");
    }
    return;
  }

  if (now - lastStaRetryMs >= WIFI_RETRY_INTERVAL_MS) {
    startStaConnect();
  }
}

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

  if (lamp.absentSinceMs == 0) {
    lamp.absentSinceMs = now;
  }

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
    lamp.lastPresenceMs = now;
    lamp.absentSinceMs = 0;
  }

  if (!lamp.present) {
    return;
  }

  if (cm < POSTURE_BAD_CM) {
    if (lamp.postureBadSinceMs == 0) {
      lamp.postureBadSinceMs = now;
    }

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

  if (lamp.lastDistanceMs == 0) {
    return;
  }

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
    Serial.println("[POSTURE] Sai tu the: ban dang cui qua thap, hay ngoi thang lung.");
  }
}

void autoControl() {
  if (!lamp.present) {
    if (lamp.power) {
      setRelay(false);
    }
    return;
  }

  if (!lamp.power) {
    setRelay(true);
  }

  handlePostureAlert();

  if (lamp.mode == LampMode::MANUAL) {
    return;
  }

  if (isnan(lamp.lux)) {
    return;
  }

  float error = TARGET_LUX_AUTO - lamp.lux;
  int nextBrightness = lamp.brightness + (int)(error * 0.03f);
  nextBrightness = constrain(nextBrightness, MIN_BRIGHTNESS, MAX_BRIGHTNESS);

  setBrightness((uint8_t)nextBrightness);
}

bool httpPostJson(const String& path, const JsonDocument& doc) {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  HTTPClient http;
  String url = String(BACKEND_URL) + path;

  if (!http.begin(secureClient, url)) {
    return false;
  }

  http.setTimeout(3000);
  http.addHeader("Content-Type", "application/json");

  String body;
  serializeJson(doc, body);

  int code = http.POST(body);
  http.end();

  return code >= 200 && code < 300;
}

bool httpGetJson(const String& path, JsonDocument& doc) {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  HTTPClient http;
  String url = String(BACKEND_URL) + path;

  if (!http.begin(secureClient, url)) {
    return false;
  }

  http.setTimeout(3000);

  int code = http.GET();

  if (code != 200) {
    http.end();
    return false;
  }

  String payload = http.getString();
  http.end();

  return deserializeJson(doc, payload) == DeserializationError::Ok;
}

void sendReport() {
  StaticJsonDocument<512> doc;

  doc["power"] = lamp.power;
  doc["brightness"] = lamp.brightness;
  doc["auto_mode"] = lamp.mode == LampMode::AUTO;
  doc["mode"] = modeToText(lamp.mode);
  doc["present"] = lamp.present;
  doc["posture_bad"] = lamp.postureBad;
  doc["alert"] = lamp.postureBad ? "wrong_posture" : "none";

  if (!isnan(lamp.lux)) {
    doc["ambient_lux"] = lamp.lux;
  }

  if (!isnan(lamp.distanceCm)) {
    doc["distance_cm"] = lamp.distanceCm;
  }

  httpPostJson("/device/report", doc);
}

void applyCommand(const String& command, int value) {
  if (command.length() == 0 || command == "none") {
    return;
  }

  if (command == "lamp_on") {
    lamp.mode = LampMode::MANUAL;
    if (lamp.brightness == 0) {
      setBrightness(50);
    }
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

    if (lamp.brightness == 0) {
      setRelay(false);
    }
  }
  else if (command == "set_brightness") {
    lamp.mode = LampMode::MANUAL;

    int b = constrain(value, 0, 100);
    setBrightness((uint8_t)b);

    if (b == 0) {
      setRelay(false);
    } else {
      setRelay(true);
    }
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
  StaticJsonDocument<256> doc;

  if (!httpGetJson("/device/pull", doc)) {
    return;
  }

  String command = doc["command"] | "";
  int value = doc["value"] | -1;

  applyCommand(command, value);
}

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

  if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
    Serial.println("BH1750 init failed");
  } else {
    Serial.println("BH1750 ready");
  }

  WiFi.persistent(false);
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);

  secureClient.setInsecure();

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

  Serial.println("Smart Study Lamp started");
}

void loop() {
  configServer.handleClient();

  if (taskWifi.due()) {
    maintainWifi();
  }

  if (taskTrig.due()) {
    triggerUltrasonic();
  }

  if (taskDistance.due()) {
    updateDistance();
  }

  if (taskLux.due()) {
    readLux();
  }

  if (taskControl.due()) {
    autoControl();
  }

  if (taskReport.due()) {
    sendReport();
  }

  if (taskCommand.due()) {
    pollCommand();
  }

  if (taskDebug.due()) {
    printStatus();
  }
}
