"""
Minimal serial test — just prints whatever the Arduino sends.
Run this to confirm real data is arriving before troubleshooting the plot.
"""

import serial

SERIAL_PORT = "/dev/tty.usbmodem101"
BAUD_RATE = 230400

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print("Connected. Printing raw lines (Ctrl+C to stop)...\n")

try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(line)
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    ser.close()