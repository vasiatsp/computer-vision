#include "esp_camera.h"
#include <WiFi.h>

const char* ssid = "mojo";
const char* password = "tiganitespatates";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
}

void loop() {
  Serial.println("Hello from ESP32-CAM!");
  delay(1000);
}
