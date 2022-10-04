/*************************************************** 
  This is an example for our Adafruit 12-channel PWM/LED driver

  Pick one up today in the adafruit shop!
  ------> http://www.adafruit.com/products/

  These drivers uses SPI to communicate, 2 pins are required to  
  interface: Data and Clock. The boards are chainable

  Adafruit invests time and resources providing this open source code, 
  please support Adafruit and open-source hardware by purchasing 
  products from Adafruit!

  Written by Limor Fried/Ladyada for Adafruit Industries.  
  BSD license, all text above must be included in any redistribution
 ****************************************************/

#include "Adafruit_TLC59711.h"
#include <SPI.h>

// How many boards do you have chained?
#define NUM_TLC59711 1

#define data   11
#define clock  13

Adafruit_TLC59711 tlc = Adafruit_TLC59711(NUM_TLC59711, clock, data);
//Adafruit_TLC59711 tlc = Adafruit_TLC59711(NUM_TLC59711);

void setup() {
  Serial.begin(9600);
  
  Serial.println("TLC59711 test");
  pinMode(10, OUTPUT);
  tlc.begin();
  tlc.write();
}

void loop() {
      tlc.setLED(1, 48000, 000, 48000);
      tlc.write();

      tlc.setLED(2, 48000, 000, 48000);
      tlc.write();

      tlc.setLED(3, 48000, 000, 48000);
      tlc.write();

      tlc.setLED(0, 48000, 000, 48000);
      tlc.write();
      delay(300);
  
}


