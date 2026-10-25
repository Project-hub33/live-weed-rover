#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ---------------- Wi-Fi ----------------
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

httpd_handle_t stream_server = NULL;

// ---------------- Video stream handler ----------------
static esp_err_t stream_handler(httpd_req_t *req)
{
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;

  char part_buf[64];

  res = httpd_resp_set_type(
    req,
    "multipart/x-mixed-replace;boundary=frame"
  );

  if (res != ESP_OK)
    return res;

  while (true)
  {
    fb = esp_camera_fb_get();

    if (!fb)
    {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    size_t hlen = snprintf(
      part_buf,
      sizeof(part_buf),
      "--frame\r\n"
      "Content-Type: image/jpeg\r\n"
      "Content-Length: %u\r\n\r\n",
      fb->len
    );

    res = httpd_resp_send_chunk(req, part_buf, hlen);

    if (res == ESP_OK)
      res = httpd_resp_send_chunk(
        req,
        (const char *)fb->buf,
        fb->len
      );

    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, "\r\n", 2);

    esp_camera_fb_return(fb);
    fb = NULL;

    if (res != ESP_OK)
      break;
  }

  return res;
}

// ---------------- Start streaming server ----------------
void startCameraServer()
{
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();

  config.server_port = 81;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_server, &config) == ESP_OK)
  {
    httpd_register_uri_handler(stream_server, &stream_uri);
    Serial.println("Camera stream server started");
  }
  else
  {
    Serial.println("Failed to start camera stream server");
  }
}

// ---------------- Setup ----------------
void setup()
{
  Serial.begin(115200);
  Serial.println();

  // Configure camera
  camera_config_t config;

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

  // Use lower resolution to reduce streaming load
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK)
  {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return;
  }

  Serial.println("Camera initialized");

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected");

  Serial.print("ESP32-CAM IP address: ");
  Serial.println(WiFi.localIP());

  // Start video streaming
  startCameraServer();

  Serial.println("Open this URL in your browser:");
  Serial.print("http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
}

// ---------------- Main loop ----------------
void loop()
{
  delay(1000);
}