#include <Wire.h>  // Include the Wire library for I2C communication

// ADXL345 I2C address
#define ADXL345_ADDR 0x53

// ADXL345 register addresses
#define POWER_CTL 0x2D
#define DATA_FORMAT 0x31
#define DATAX0 0x32
#define DATAX1 0x33
#define DATAY0 0x34
#define DATAY1 0x35
#define DATAZ0 0x36
#define DATAZ1 0x37

// Threshold for detecting motion (adjust as necessary)
#define MOTION_THRESHOLD 0.02  // 0.02g (2% of 1g)

// Sleep detection parameters
#define SLEEP_TIMEOUT 900000  // 15 minutes in milliseconds

unsigned long lastMovementTime = 0;  // Time of last movement detection
bool personIsSleeping = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  // Initialize I2C communication
  Wire.begin();

  // Wake up the ADXL345
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(POWER_CTL);  // Access the POWER_CTL register
  Wire.write(0x08);  // Set the measure bit to 1 (wake up)
  Wire.endTransmission();

  // Set data format to full resolution
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(DATA_FORMAT);  // Access the DATA_FORMAT register
  Wire.write(0x08);  // Set the range to ± 4g, full resolution
  Wire.endTransmission();
}

void loop() {
  // Read acceleration data from the ADXL345
  int x = read16BitData(DATAX0);
  int y = read16BitData(DATAY0);
  int z = read16BitData(DATAZ0);

  // Convert to "g"
  float x_g = (float)x / 256.0;
  float y_g = (float)y / 256.0;
  float z_g = (float)z / 256.0;

  // Calculate the total acceleration magnitude (distance from origin)
  float totalAcceleration = sqrt(x_g * x_g + y_g * y_g + z_g * z_g);

  // Print the acceleration values
  Serial.print("X: ");
  Serial.print(x_g);
  Serial.print("g, Y: ");
  Serial.print(y_g);
  Serial.print("g, Z: ");
  Serial.print(z_g);
  Serial.print("g, Total Acceleration: ");
  Serial.print(totalAcceleration);
  Serial.println("g");

  // Check if the person is still (no significant motion)
  if (totalAcceleration < MOTION_THRESHOLD) {
    // If no significant motion, check if 15 minutes have passed
    if (millis() - lastMovementTime >= SLEEP_TIMEOUT && !personIsSleeping) {
      personIsSleeping = true;
      Serial.println("Person is sleeping.");
    }
  } else {
    // Reset last movement time if there's significant motion
    lastMovementTime = millis();
    personIsSleeping = false;  // Not sleeping
  }

  delay(500);  // Delay for readability
}

// Function to read 16-bit data from the accelerometer
int read16BitData(byte addr) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(addr);  // Write the register address to start reading
  Wire.endTransmission();
  Wire.requestFrom(ADXL345_ADDR, 2);  // Request 2 bytes of data

  byte lowByte = Wire.read();
  byte highByte = Wire.read();

  int value = (highByte << 8) | lowByte;  // Combine the 2 bytes
  return value;
}

