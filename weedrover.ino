#include <WiFi.h>
#include <WebServer.h>

// ---------- Wi-Fi configuration ----------
const char* WIFI_SSID = "NEW";
const char* WIFI_PASSWORD = "12345678";

// ---------- GPIO assignments ----------
// Connect these GPIOs to motor driver inputs.
// Do NOT connect motors directly to ESP32 pins.

const uint8_t MOTOR_1 = 13;
const uint8_t MOTOR_2 = 14;
const uint8_t MOTOR_3 = 25;
const uint8_t MOTOR_4 = 26;

const uint8_t CUTTER_MOTOR = 18;
const uint8_t WATER_PUMP = 19;

// ---------- HTTP server ----------
WebServer server(80);

// ---------- Rover state ----------
enum RoverState {
  ROVER_STOPPED,
  ROVER_PATROLLING,
  ROVER_CUTTING_WATERING
};

RoverState roverState = ROVER_STOPPED;

// Safety timeout for tool motors
// Cutter and pump automatically stop after 10 seconds.
// Adjust only after testing your hardware.
const unsigned long TOOL_TIMEOUT_MS = 10000;
unsigned long toolsStartedAt = 0;
bool toolsRunning = false;

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
  digitalWrite(MOTOR_2, HIGH);
  digitalWrite(MOTOR_3, HIGH);
  digitalWrite(MOTOR_4, HIGH);
}

void stopCutter() {
  digitalWrite(CUTTER_MOTOR, LOW);
}

void stopPump() {
  digitalWrite(WATER_PUMP, LOW);
}

void stopTools() {
  stopCutter();
  stopPump();
  toolsRunning = false;
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
// COMMAND: /red
// Dashboard: Patrol
// Action: All four drive motors forward
// ==================================================

void handlePatrol() {
  stopTools();
  moveForward();

  roverState = ROVER_PATROLLING;

  Serial.println("PATROL: Moving forward");
  sendResponse(200, "ROVER MOVING FORWARD");
}

// ==================================================
// COMMAND: /green
// Dashboard: Weed detected
// Action: Stop rover, activate cutter and pump
// ==================================================

void handleWeedDetected() {
  stopDriveMotors();

  digitalWrite(CUTTER_MOTOR, HIGH);
  digitalWrite(WATER_PUMP, HIGH);

  toolsRunning = true;
  toolsStartedAt = millis();
  roverState = ROVER_CUTTING_WATERING;

  Serial.println("WEED DETECTED: Rover stopped");
  Serial.println("Cutter ON, water pump ON");

  sendResponse(200, "ROVER STOPPED: CUTTER AND PUMP ON");
}

// ==================================================
// COMMAND: /stop
// Dashboard: ALL STOP
// Action: Turn off every motor
// ==================================================

void handleStop() {
  stopAllMotors();

  Serial.println("EMERGENCY/ALL STOP: All outputs OFF");
  sendResponse(200, "ALL MOTORS STOPPED");
}

// ==================================================
// COMMAND: /cutter_on
// Dashboard: Cutter ON
// Action: Stop rover, activate cutter only
// ==================================================

void handleCutterOn() {
  stopDriveMotors();
  stopTools();

  digitalWrite(CUTTER_MOTOR, HIGH);

  toolsRunning = true;
  toolsStartedAt = millis();
  roverState = ROVER_CUTTING_WATERING;

  Serial.println("CUTTER ON");

  sendResponse(200, "CUTTER ON");
}

// ==================================================
// COMMAND: /pump_on
// Dashboard: Pump ON
// Action: Stop rover, activate pump only
// ==================================================

void handlePumpOn() {
  stopDriveMotors();
  stopTools();

  digitalWrite(WATER_PUMP, HIGH);

  toolsRunning = true;
  toolsStartedAt = millis();
  roverState = ROVER_CUTTING_WATERING;

  Serial.println("WATER PUMP ON");

  sendResponse(200, "WATER PUMP ON");
}

// ==================================================
// COMMAND: /status
// Reports current rover state
// ==================================================

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

  // Configure every motor output
  pinMode(MOTOR_1, OUTPUT);
  pinMode(MOTOR_2, OUTPUT);
  pinMode(MOTOR_3, OUTPUT);
  pinMode(MOTOR_4, OUTPUT);

  pinMode(CUTTER_MOTOR, OUTPUT);
  pinMode(WATER_PUMP, OUTPUT);

  // Ensure all outputs start LOW
  stopAllMotors();

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");

  unsigned long wifiStart = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - wifiStart < 20000) {
    delay(250);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("Wi-Fi connected");
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("Wi-Fi connection failed.");
    Serial.println("Restart after checking credentials/router.");
    return;
  }

  configureRoutes();
  server.begin();

  Serial.println("HTTP server started on port 80");
  Serial.println("Available commands:");
  Serial.println("/red        - Forward patrol");
  Serial.println("/green      - Stop, cutter and pump ON");
  Serial.println("/stop       - All motors OFF");
  Serial.println("/cutter_on  - Cutter ON");
  Serial.println("/pump_on    - Pump ON");
  Serial.println("/status     - Rover status");
}

// ==================================================
// MAIN LOOP
// ==================================================

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    // Fail safe if Wi-Fi is lost
    stopAllMotors();
    delay(100);
    return;
  }

  server.handleClient();

  // Automatically stop the cutter and pump after timeout
  if (toolsRunning &&
      millis() - toolsStartedAt >= TOOL_TIMEOUT_MS) {

    stopTools();
    stopDriveMotors();
    roverState = ROVER_STOPPED;

    Serial.println("Tool timeout: Cutter and pump OFF");
  }
}
