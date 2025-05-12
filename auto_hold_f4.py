import time
import pyautogui

# Space tuşunu baştan basılı tut
pyautogui.keyDown('space')

try:
    # Ctrl+C ile durdurana kadar döngü devam eder
    while True:
        pyautogui.press('f4')  # F4’e bas
        time.sleep(2)          # 2 saniye bekle
except KeyboardInterrupt:
    # Ctrl+C ile yakalandığında Space’i bırak
    pyautogui.keyUp('space')
    print('\nStopped, space released.')
