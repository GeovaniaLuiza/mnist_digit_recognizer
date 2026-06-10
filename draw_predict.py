import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import torch
from model import MLP

# carregar modelo
model = MLP()
model.load_state_dict(torch.load("models/mnist_pytorch.pth", map_location="cpu"))
model.eval()

# canvas
W, H = 200, 200

window = tk.Tk()
window.title("MNIST Draw Predictor")

canvas = tk.Canvas(window, width=W, height=H, bg="black")
canvas.pack()

image = Image.new("L", (W, H), "black")
draw = ImageDraw.Draw(image)

last_x, last_y = None, None

def paint(event):
    global last_x, last_y

    x, y = event.x, event.y

    if last_x is not None:
        canvas.create_line(last_x, last_y, x, y, fill="white", width=10)
        draw.line([last_x, last_y, x, y], fill=255, width=10)

    last_x, last_y = x, y

def reset(event):
    global last_x, last_y
    last_x, last_y = None, None

canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", reset)

def predict():
    img = image.resize((28, 28))
    img = np.array(img)

    # 🔥 CORREÇÃO MAIS IMPORTANTE
    img = 255 - img  # inverter cores

    img = img / 255.0
    img = img.reshape(1, 28*28)

    x = torch.tensor(img, dtype=torch.float32)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    label.config(text=f"Predição: {pred}")

def clear():
    global image, draw
    canvas.delete("all")
    image = Image.new("L", (W, H), "black")
    draw = ImageDraw.Draw(image)

btn = tk.Button(window, text="Predict", command=predict)
btn.pack()

btn2 = tk.Button(window, text="Clear", command=clear)
btn2.pack()

label = tk.Label(window, text="Desenhe um dígito")
label.pack()

window.mainloop()