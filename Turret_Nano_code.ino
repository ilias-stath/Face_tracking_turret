#include <Arduino.h>
#include <Servo.h>
#include <A4988.h>


Servo laserServo;

#define MOTOR_STEPS 200       // 1.8° per step = 200 steps/rev
#define stepPin 4  // D4 -> STEP
#define direPin 3  // D3 -> DIR
#define RPM 10


float servoPos = 0;
float servoAngle = 40;
float servoGetter = 0;
float stepperPos = 0;
float stepperPosPrev = 0;
int seperate = 0;
String coord;
float Kpx = 0.1; // P like controller, delay the move to reduce jittering
float Kpy = 0.4; // P like controller, delay the move to reduce jittering


A4988 stepper(MOTOR_STEPS, direPin, stepPin);

void setup() {
  laserServo.attach(2); // use D2
  Serial.begin(9600);
  laserServo.write(140); // center position


  stepper.begin(RPM);
  stepper.setMicrostep(1);  // Set to 1 for full-step mode (adjust if using MS1–MS3)


}

void loop() {

  if(Serial.available()){
    // Seperating the Serial input
    coord = Serial.readStringUntil('\n');
    seperate = coord.indexOf(',');

    if(seperate > 0){
      stepperPos = coord.substring(0,seperate).toInt();
      servoGetter = coord.substring(seperate + 1).toInt();

      servoAngle -= Kpy*servoGetter; // Apply the P controller logic

      servoAngle = max(0, min(75, servoAngle)); // constrain the angle betqeen 0 and 75 deg

      servoPos = 180 - servoAngle; // Shift 180 deg, because of the servo orientation

      // This check isn't neccesairy, because we contrained it before
      if(servoPos <= 180 && servoPos >= 0){
        laserServo.write(servoPos); // Move the servo
      }

      // It only does steps between 4 and 50, because of the jittering
      // Furthermore by experiments we conclude that deviding the input by 1.8 to get steps has liitle impact so we leave it as it is
      if(abs(stepperPos) > 4 && abs(stepperPos) <= 50){
        stepperPos = Kpx*stepperPos; // Apply the P controller logic
        stepper.move(stepperPos); // Move the servo
      }
    }
  }
}
