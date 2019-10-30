int nTriggerPin = 2;

void setup() {
	pinMode(nTriggerPin, OUTPUT);
	// put your setup code here, to run once:
	Serial.begin(9600);
	while (!Serial) {
		; // wait for serial port to connect. Needed for native USB
	}
}

void loop() {
	int16_t nData;
	int nRecvBytes = Serial.readBytes((char*)&nData, sizeof(nData));
	if (nRecvBytes > 0) {
		if (nData >= 0) {
			digitalWrite(nTriggerPin, HIGH);   // sets the trigger on
			delay(nData);                  // waits for a milisec
			digitalWrite(nTriggerPin, LOW);    // sets the trigger off
		} else {
			digitalWrite(nTriggerPin, LOW);   // sets the trigger on
			delay(nData);                  // waits for a milisec
			digitalWrite(nTriggerPin, HIGH);    // sets the trigger off
		}
	}
}
