--- Scroll down for English ---

# Gauss-zaj szűrése

A projektben olyan különböző algoritmusokat valósítottam meg, melyekkel az additív Gauss-zaj szűrését mutathatjuk be a képtérben.
A projekt elsődleges nyelve magyar, ezért a magyarázó kommentek magyarul íródtak.

A demóalkalmazásban hat algoritmust implementáltam:
- Kuwahara-szűrő
- Szigma-szűrő
- Gradiens inverz súlyozott metódus
- A gradiens inverz súlyozott metódus továbbfejlesztett változata
- Bilaterális szűrő
- A bilaterális szűrő továbbfejlesztett változata

### A futtatáshoz szükséges csomagok:
- math
- statistics
- sys
- numpy
- PyQt5
- opencv-python
- time

### A demóprogram elindítása

* A GUI.py az alkalmazás belépési pontja, a futtatást követően megjelenik a felhasználói felület.
* A felhasználói felületen ki kell választani a tesztelni kívánt algoritmust és ennek paramétereit, majd lenyomni a `Run algorithm` gombot.
* A kimeneti kép mentéséhez használható a kikommentezett kódrész a `call_algoritm` függvény végén.


# Gaussian noise reduction

This project is implementing different algorithms to achieve reduction of additive Gaussian noise in the spatial domain.
The primary language is Hungarian.

Six algorithms are included in the demo:
- Kuwahara algorithm
- Sigma algorithm
- Gradient inverse weighted method
- A possible upgrade of gradient inverse weighted method
- Bilateral filter
- A possible upgrade of bilateral filter

### Required modules to install:
- math
- statistics
- sys
- numpy
- PyQt5
- opencv-python
- time

### How to start the demo

* GUI.py is the entry point of the application, after running it the user interface of the demo shows up.
* Choose the desired options on the window to test and then click on `Run algorithm`.
* To save your output image, use the commented code in `call_algoritm` function.
