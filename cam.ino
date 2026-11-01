#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ---------------- Wi-Fi configuration ----------------
const char* ssid = "NEW";
const char* password = "12345678";

// ---------------- AI Thinker ESP32-CAM pins ----------------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ---------------- HTTP server ----------------
static httpd_handle_t stream_server = NULL;

// ---------------- MJPEG stream handler ----------------
static esp_err_t stream_handler(httpd_req_t *req)
{
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[96];

  res = httpd_resp_set_type(
    req,
    "multipart/x-mixed-replace;boundary=frame"
  );

  if (res != ESP_OK) {
    return res;
  }

  while (true) {
    fb = esp_camera_fb_get();

    if (fb == NULL) {
      Serial.println("ERROR: Camera capture failed");
      return ESP_FAIL;
    }

    if (fb->format != PIXFORMAT_JPEG) {
      Serial.println("ERROR: Frame is not JPEG");
      esp_camera_fb_return(fb);
      return ESP_FAIL;
    }

    int hlen = snprintf(
      part_buf,
      sizeof(part_buf),
      "--frame\r\n"
      "Content-Type: image/jpeg\r\n"
      "Content-Length: %u\r\n\r\n",
      (unsigned int)fb->len
    );

    if (hlen < 0 || hlen >= (int)sizeof(part_buf)) {
      esp_camera_fb_return(fb);
      return ESP_FAIL;
    }

    res = httpd_resp_send_chunk(req, part_buf, hlen);

    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(
        req,
        (const char *)fb->buf,
        fb->len
      );
    }

    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, "\r\n", 2);
    }

    // Always release the frame buffer
    esp_camera_fb_return(fb);
    fb = NULL;

    if (res != ESP_OK) {
      // Usually the browser/client disconnected
      break;
    }
  }

  return res;
}

// ---------------- Start streaming server ----------------
void startCameraServer()
{
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();

  config.server_port = 81;
  config.ctrl_port = 32769;
  config.max_uri_handlers = 4;
  config.stack_size = 8192;

  httpd_uri_t stream_uri = {};
  stream_uri.uri = "/stream";
  stream_uri.method = HTTP_GET;
  stream_uri.handler = stream_handler;
  stream_uri.user_ctx = NULL;

  esp_err_t err = httpd_start(&stream_server, &config);

  if (err == ESP_OK) {
    err = httpd_register_uri_handler(
      stream_server,
      &stream_uri
    );

    if (err == ESP_OK) {
      Serial.println("SUCCESS: Stream server started");
    } else {
      Serial.printf(
        "ERROR: URI registration failed: 0x%x\n",
        err
      );
      httpd_stop(stream_server);
      stream_server = NULL;
    }
  } else {
    Serial.printf(
      "ERROR: HTTP server failed to start: 0x%x\n",
      err
    );
  }
}

// ---------------- Initialize camera ----------------
bool initializeCamera()
{
  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Use PSRAM when available
  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf(
      "ERROR: Camera initialization failed: 0x%x\n",
      err
    );
    return false;
  }

  Serial.println("SUCCESS: Camera initialized");

  sensor_t *sensor = esp_camera_sensor_get();

  if (sensor != NULL) {
    sensor->set_brightness(sensor, 0);
    sensor->set_contrast(sensor, 0);
    sensor->set_saturation(sensor, 0);
  }

  return true;
}

// ---------------- Connect to Wi-Fi ----------------
bool connectWiFi()
{
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to Wi-Fi");

  unsigned long startTime = millis();

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");

    if (millis() - startTime > 30000) {
      Serial.println("\nERROR: Wi-Fi connection timed out");
      return false;
    }
  }

  Serial.println("\nSUCCESS: Wi-Fi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  return true;
}

// ---------------- Setup ----------------
void setup()
{
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("==============================");
  Serial.println("AI THINKER ESP32-CAM STREAM");
  Serial.println("==============================");

  if (!initializeCamera()) {
    Serial.println("Check camera model, pins and power.");
    return;
  }

  if (!connectWiFi()) {
    Serial.println("Check Wi-Fi SSID and password.");
    return;
  }

  startCameraServer();

  if (stream_server != NULL) {
    Serial.println();
    Serial.println("Open this URL in your browser:");
    Serial.print("http://");
    Serial.print(WiFi.localIP());
    Serial.println(":81/stream");
  }
}

// ---------------- Main loop ----------------
void loop()
{
  // The HTTP server handles video streaming asynchronously.
  // Restart Wi-Fi if the connection is lost.
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi disconnected. Reconnecting...");
    WiFi.reconnect();

    unsigned long startTime = millis();

    while (WiFi.status() != WL_CONNECTED &&
           millis() - startTime < 10000) {
      delay(250);
    }

    if (WiFi.status() == WL_CONNECTED) {
      Serial.print("Reconnected. IP: ");
      Serial.println(WiFi.localIP());
    }
  }

  delay(1000);
}
