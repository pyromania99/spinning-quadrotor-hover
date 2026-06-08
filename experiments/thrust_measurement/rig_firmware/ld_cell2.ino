#include "HX711.h"

// HX711 circuit wiring
const int LOADCELL1_DOUT_PIN = 2;
const int LOADCELL1_SCK_PIN = 3;
const int LOADCELL2_DOUT_PIN = 18;
const int LOADCELL2_SCK_PIN = 19;

HX711 scale1;
HX711 scale2;

// Array of known weights (in your chosen units)
const float known_weights[] = {50.0, 100.0, 200.0}; // Example weights in grams
const int num_weights = sizeof(known_weights) / sizeof(known_weights[0]);

void setup() {
  Serial.begin(57600);

  // Initialize both HX711 scales
  scale1.begin(LOADCELL1_DOUT_PIN, LOADCELL1_SCK_PIN);
  scale2.begin(LOADCELL2_DOUT_PIN, LOADCELL2_SCK_PIN);
}

void loop() {
  if (scale1.is_ready() && scale2.is_ready()){
    scale1.set_scale();
    scale2.set_scale();

    Serial.println("Tare... remove any weights from the scales.");
    delay(5000);
    scale1.tare();
    scale2.tare();
    Serial.println("Tare done...");

    for (int i = 0; i < num_weights; i++) {
      Serial.print("Place a known weight of ");
      Serial.print(known_weights[i]);
      Serial.println(" on the scales...");
      delay(5000); // Give time to place the weight

      // long reading1 = scale1.get_units(10);
      // long reading2 = scale2.get_units(10);
      long reading1 = scale1.read_average(10);
      long reading2 = scale2.read_average(10);
      long combined_reading = (reading1 + reading2);

      Serial.print("Reading for load cell 1: ");
      Serial.println(reading1);
      Serial.print("Reading for load cell 2: ");
      Serial.println(reading2);
      Serial.print("Combined reading for ");
      Serial.print(known_weights[i]);
      Serial.print(" grams: ");
      Serial.println(combined_reading);

      while (Serial.available() == 0){}
    }
    
  } else {
    if (!scale1.is_ready()) {
      Serial.println("HX711 for load cell 1 not found.");
    }
    if (!scale2.is_ready()) {
      Serial.println("HX711 for load cell 2 not found.");
    }
  }

  delay(1000);
}
