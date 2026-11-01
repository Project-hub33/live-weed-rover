#include <WiFi.h>
#include <WebServer.h>

// ---------- Wi-Fi configuration ----------
const char* WIFI_SSID = "NEW";
const char* WIFI_PASSWORD = "12345678";

// ---------- GPIO assignments ----------
const uint8_t MOTOR_1 = 13;
const uint8_t MOTOR_2 = 14;
const uint8_t MOTOR_3 = 25;
const uint8_t MOTOR_4 = 26;

const uint8_t CUTTER_MOTOR = 18;
const uint8_t WATER_PUMP = 19;

// Cutter and pump are ACTIVE LOW (relay module): LOW = ON, HIGH = OFF.
// If you use a MOSFET/transistor instead, swap these two values.
const uint8_t TOOL_ON  = LOW;
const uint8_t TOOL_OFF = HIGH;

// ---------- HTTP server ----------
WebServer server(80);

// ---------- Rover state ----------
enum RoverState {
  ROVER_STOPPED,
  ROVER_PATROLLING,
  ROVER_CUTTING_WATERING
};
RoverState roverState = ROVER_STOPPED;

// ---------- Weed sequence (non-blocking) ----------
enum WeedStep {
  WEED_IDLE,
  WEED_CUTTING,   // cutter ON
  WEED_PAUSE,     // both OFF
  WEED_WATERING   // pump ON
};
WeedStep weedStep = WEED_IDLE;
unsigned long weedStepStartedAt = 0;

const unsigned long CUTTER_TIME_MS = 5000;
const unsigned long PAUSE_TIME_MS  = 1000;
const unsigned long PUMP_TIME_MS   = 5000;

// ---------- Manual tool safety timeout ----------
const unsigned long TOOL_TIMEOUT_MS = 10000;
unsigned long toolsStartedAt = 0;
bool toolsRunning = false;

// ---------- Wi-Fi reconnect ----------
unsigned long lastWifiAttempt = 0;
const unsigned long WIFI_RETRY_MS = 5000;

// ==================================================
// MOTOR CONTROL
// ==================================================

void stopDriveMotors() {
  digitalWrite(MOTOR_1, LOW);
  digitalWrite(MOTOR_2, LOW);
  digitalWrite(MOTOR_3, LOW);
  digitalWrite(MOTOR_4, LOW);
}

void moveForward() {
  digitalWrite(MOTOR_1, HIGH);
  digitalWrite(MOTOR_2, LOW);
  digitalWrite(MOTOR_3, HIGH);
  digitalWrite(MOTOR_4, LOW);
}

void stopCutter() { digitalWrite(CUTTER_MOTOR, TOOL_OFF); }
void stopPump()   { digitalWrite(WATER_PUMP, TOOL_OFF); }

void stopTools() {
  stopCutter();
  stopPump();
  toolsRunning = false;
  weedStep = WEED_IDLE;   // cancel any running weed sequence
}

void stopAllMotors() {
  stopDriveMotors();
  stopTools();
  roverState = ROVER_STOPPED;
}

// ==================================================
// HTTP RESPONSE
// ==================================================

void sendResponse(int code, const char* message) {
  server.send(code, "text/plain", message);
}

// ==================================================
// COMMANDS
// ==================================================

// /red : patrol forward
void handlePatrol() {
  stopTools();
  moveForward();
  roverState = ROVER_PATROLLING;

  Serial.println("PATROL: Moving forward");
  sendResponse(200, "ROVER MOVING FORWARD");
}

// /green : weed detected. Starts the sequence and returns immediately.
void handleWeedDetected() {
  stopDriveMotors();
  stopTools();

  weedStep = WEED_CUTTING;
  weedStepStartedAt = millis();
  digitalWrite(CUTTER_MOTOR, TOOL_ON);
  roverState = ROVER_CUTTING_WATERING;

  Serial.println("WEED DETECTED: Rover stopped, cutter ON");
  sendResponse(200, "ROVER STOPPED: WEED SEQUENCE STARTED");
}

// /stop : everything OFF (works at any time, even mid-sequence)
void handleStop() {
  stopAllMotors();

  Serial.println("ALL STOP: All outputs OFF");
  sendResponse(200, "ALL MOTORS STOPPED");
}

// /cutter_on
void handleCutterOn() {
  stopDriveMotors();
  stopTools();

  digitalWrite(CUTTER_MOTOR, TOOL_ON);
  toolsRunning = true;
  toolsStartedAt = millis();
  roverState = ROVER_CUTTING_WATERING;

  Serial.println("CUTTER ON");
  sendResponse(200, "CUTTER ON");
}

// /pump_on
void handlePumpOn() {
  stopDriveMotors();
  stopTools();

  digitalWrite(WATER_PUMP, TOOL_ON);
  toolsRunning = true;
  toolsStartedAt = millis();
  roverState = ROVER_CUTTING_WATERING;

  Serial.println("WATER PUMP ON");
  sendResponse(200, "WATER PUMP ON");
}

// /status
void handleStatus() {
  const char* stateText = "STOPPED";
  if (roverState == ROVER_PATROLLING) {
    stateText = "PATROLLING";
  } else if (roverState == ROVER_CUTTING_WATERING) {
    stateText = "CUTTING_WATERING";
  }

  String response = String("State: ") + stateText;
  response += "\nWiFi: ";
  response += (WiFi.status() == WL_CONNECTED) ? "CONNECTED" : "DISCONNECTED";

  server.send(200, "text/plain", response);
}

// ==================================================
// WEED SEQUENCE UPDATE (called from loop)
// ==================================================

void updateWeedSequence() {
  if (weedStep == WEED_IDLE) return;

  unsigned long elapsed = millis() - weedStepStartedAt;

  switch (weedStep) {
    case WEED_CUTTING:
      if (elapsed >= CUTTER_TIME_MS) {
        stopCutter();
        weedStep = WEED_PAUSE;
        weedStepStartedAt = millis();
        Serial.println("Cutter OFF, pausing");
      }
      break;

    case WEED_PAUSE:
      if (elapsed >= PAUSE_TIME_MS) {
        digitalWrite(WATER_PUMP, TOOL_ON);
        weedStep = WEED_WATERING;
        weedStepStartedAt = millis();
        Serial.println("Pump ON");
      }
      break;

    case WEED_WATERING:
      if (elapsed >= PUMP_TIME_MS) {
        stopPump();
        weedStep = WEED_IDLE;
        moveForward();
        roverState = ROVER_PATROLLING;
        Serial.println("Weed sequence complete: Rover moving forward again");
      }
      break;

    default:
      break;
  }
}

// ==================================================
// WEB SERVER ROUTES
// ==================================================

void configureRoutes() {
  server.on("/", HTTP_GET, []() {
    sendResponse(200, "AgriBot ESP32 is online");
  });

  server.on("/red", HTTP_GET, handlePatrol);
  server.on("/green", HTTP_GET, handleWeedDetected);
  server.on("/stop", HTTP_GET, handleStop);
  server.on("/cutter_on", HTTP_GET, handleCutterOn);
  server.on("/pump_on", HTTP_GET, handlePumpOn);
  server.on("/status", HTTP_GET, handleStatus);

  server.onNotFound([]() {
    sendResponse(404, "Unknown command");
  });
}

// ==================================================
// SETUP
// ==================================================

void setup() {
  Serial.begin(115200);

  pinMode(MOTOR_1, OUTPUT);
  pinMode(MOTOR_2, OUTPUT);
  pinMode(MOTOR_3, OUTPUT);
  pinMode(MOTOR_4, OUTPUT);
  pinMode(CUTTER_MOTOR, OUTPUT);
  pinMode(WATER_PUMP, OUTPUT);

  // Safe startup: everything OFF
  stopDriveMotors();
  stopCutter();
  stopPump();

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");
  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 20000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("Wi-Fi connected");
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());

    // System ON: rover starts moving forward
    moveForward();
    roverState = ROVER_PATROLLING;
    Serial.println("Rover moving forward");
  } else {
    Serial.println("Wi-Fi not connected yet. Will keep retrying in loop().");
  }

  // Always start the server so it works as soon as Wi-Fi comes back
  configureRoutes();
  server.begin();

  Serial.println("HTTP server started on port 80");
  Serial.println("/red /green /stop /cutter_on /pump_on /status");
}

// ==================================================
// MAIN LOOP
// ==================================================

void loop() {
  // Fail-safe + retry if Wi-Fi is lost
  if (WiFi.status() != WL_CONNECTED) {
    stopAllMotors();

    if (millis() - lastWifiAttempt >= WIFI_RETRY_MS) {
      lastWifiAttempt = millis();
      Serial.println("Wi-Fi lost. Reconnecting...");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }
    delay(100);
    return;
  }

  server.handleClient();
  updateWeedSequence();

  // Auto-stop manual cutter/pump after timeout
  if (toolsRunning && millis() - toolsStartedAt >= TOOL_TIMEOUT_MS) {
    stopTools();
    stopDriveMotors();
    roverState = ROVER_STOPPED;
    Serial.println("Tool timeout: Cutter and pump OFF");
  }
}
