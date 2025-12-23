#include "esp_camera.h"
#include <WiFi.h>

// ===================
// Select camera model
// ===================
#define CAMERA_MODEL_ESP32S3_EYE // Has PSRAM
#include "camera_pins.h"

// ===========================
// Enter your WiFi credentials
// ===========================
const char *ssid = "Cash";
const char *password = "kayesmah";

void startCameraServer();
void setupLedFlash(int pin);

// --------------------------------------------------------------------
// HAWKEYE: LOCKED DEFAULTS (from your /status JSON)
// http://172.20.10.3/status
// --------------------------------------------------------------------
static const int HAWKEYE_XCLK_MHZ = 25;

// From JSON:
static const int HAWKEYE_FRAMESIZE_VAL = 10;   // framesize:10 (from your /status)
static const int HAWKEYE_QUALITY      = 10;
static const int HAWKEYE_BRIGHTNESS   = 1;
static const int HAWKEYE_CONTRAST     = -1;
static const int HAWKEYE_SATURATION   = 0;
static const int HAWKEYE_SPECIAL_EFFECT = 0;

static const int HAWKEYE_WB_MODE      = 0;
static const int HAWKEYE_AWB          = 1;
static const int HAWKEYE_AWB_GAIN     = 1;

static const int HAWKEYE_AEC          = 1;
static const int HAWKEYE_AEC2         = 0;
static const int HAWKEYE_AE_LEVEL     = 0;

static const int HAWKEYE_AGC          = 1;
static const int HAWKEYE_GAINCEILING  = 0;

static const int HAWKEYE_BPC          = 0;
static const int HAWKEYE_WPC          = 1;
static const int HAWKEYE_RAW_GMA      = 1;
static const int HAWKEYE_LENC         = 1;

static const int HAWKEYE_HMIRROR      = 1;
static const int HAWKEYE_VFLIP        = 0;   // not present in your JSON, but your UI shows OFF

static const int HAWKEYE_DCW          = 1;
static const int HAWKEYE_COLORBAR     = 0;

static const int HAWKEYE_LED_INTENSITY = 0;

// --------------------------------------------------------------------
// Apply your exact sensor defaults every boot
// --------------------------------------------------------------------
static void apply_hawkeye_defaults() {
  sensor_t *s = esp_camera_sensor_get();
  if (!s) return;

  // Framesize: set using numeric value from /status
  // If you later want VGA explicitly, replace next line with:
  // s->set_framesize(s, FRAMESIZE_VGA);
  s->set_framesize(s, (framesize_t)HAWKEYE_FRAMESIZE_VAL);

  s->set_quality(s, HAWKEYE_QUALITY);
  s->set_brightness(s, HAWKEYE_BRIGHTNESS);
  s->set_contrast(s, HAWKEYE_CONTRAST);
  s->set_saturation(s, HAWKEYE_SATURATION);
  s->set_special_effect(s, HAWKEYE_SPECIAL_EFFECT);

  s->set_wb_mode(s, HAWKEYE_WB_MODE);
  s->set_whitebal(s, HAWKEYE_AWB);
  s->set_awb_gain(s, HAWKEYE_AWB_GAIN);

  s->set_exposure_ctrl(s, HAWKEYE_AEC);
  s->set_aec2(s, HAWKEYE_AEC2);
  s->set_ae_level(s, HAWKEYE_AE_LEVEL);

  s->set_gain_ctrl(s, HAWKEYE_AGC);
  s->set_gainceiling(s, (gainceiling_t)HAWKEYE_GAINCEILING);

  s->set_bpc(s, HAWKEYE_BPC);
  s->set_wpc(s, HAWKEYE_WPC);
  s->set_raw_gma(s, HAWKEYE_RAW_GMA);
  s->set_lenc(s, HAWKEYE_LENC);

  s->set_hmirror(s, HAWKEYE_HMIRROR);
  s->set_vflip(s, HAWKEYE_VFLIP);

  s->set_dcw(s, HAWKEYE_DCW);
  s->set_colorbar(s, HAWKEYE_COLORBAR);

  // LED intensity is handled by webserver in many builds; we keep default 0.
}

// --------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;

  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  // LOCK XCLK to 25MHz as per your settings
  config.xclk_freq_hz = HAWKEYE_XCLK_MHZ * 1000000;

  // IMPORTANT:
  // Initialize with a safe size; then we apply exact framesize right after init.
  // This prevents init instability on some boards.
  config.frame_size   = FRAMESIZE_VGA;

  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  config.grab_mode    = CAMERA_GRAB_LATEST;
  config.fb_location  = CAMERA_FB_IN_PSRAM;

  // Match your JSON quality
  config.jpeg_quality = HAWKEYE_QUALITY;

  // Use 2 framebuffers if PSRAM is available for smoother stream
  config.fb_count     = psramFound() ? 2 : 1;

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  // APPLY YOUR EXACT SETTINGS (this overrides the sketch's defaults)
  apply_hawkeye_defaults();

  // REMOVE old sketch overrides:
  // - no forced vflip for ESP32S3_EYE
  // - no forced framesize QVGA
  // - no OV3660 special handling overriding your values

#if defined(LED_GPIO_NUM)
  setupLedFlash(LED_GPIO_NUM);
#endif

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi connected");

  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
}

void loop() {
  delay(10000);
}
