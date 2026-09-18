#include <ESP32Servo.h>

Servo servoH;   // Horizontal servo
Servo servoV;   // Vertical servo

int posH = 90;  // Start from middle
int posV = 90;

// Added variables for sweep control
enum Mode { SWEEP, TRACK };
Mode currentMode = SWEEP;
int direction = 1; // sweep direction

void setup() {
  Serial.begin(9600);
  servoH.attach(13);  // Horizontal servo pin
  servoV.attach(12);  // Vertical servo pin
  servoH.write(posH);
  servoV.write(posV);
  Serial.println(" Starting sweep mode... waiting for 'F' to stop and tracking to begin.");
}

void loop() {
  
  // 1) Check serial for commands or mode switch
  if (Serial.available()) {
    char cmd = Serial.read();

    // Handle mode switch from ML
    if (cmd == 'I') {        // Keep sweeping until detection
      currentMode = SWEEP;
    } 
    else if (cmd == 'F') {   // Stop sweeping — switch to face tracking
      currentMode = TRACK;
      Serial.println("✅ Face detected — switching to tracking mode.");
      
      servoH.write(posH);
      delay(20);
    } 
    else {
      // Normal tracking commands (active only in TRACK mode)
      if (currentMode == TRACK) {
        digitalWrite(27,HIGH);
        switch (cmd) {
          case 'L':  // move left
            posH += 2;
            break;
          case 'R':  // move right
            posH -= 2;
            break;
          case 'U':  // move up
            posV += 2;
            break;
          case 'D':  // move down
            posV -= 2;
            break;
          default:
            return;
        }

        // Clamp angles
        posH = constrain(posH, 0, 180);
        posV = constrain(posV, 0, 180);

        servoH.write(posH);
        servoV.write(posV);

        Serial.print("H:");
        Serial.print(posH);
        Serial.print(" V:");
        Serial.println(posV);

        delay(15);
      }
    }
  }

  // 2) Perform sweep if in SWEEP mode
  if (currentMode == SWEEP) {
    posH += direction;
    if (posH >= 180 || posH <= 0) direction = -direction;  // reverse direction
    servoH.write(posH);
    delay(30); // slow continuous sweep
  }
}
