import tkinter as tk
from tkinter import messagebox
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 📊 Modeli yükleyelim ve eğitelim
xgb_model = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1)

data = pd.DataFrame({
    "Yaş": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    "Nabız": [80, 85, 90, 95, 100, 105, 110, 78, 82, 88, 92, 97, 102, 108, 112, 115, 118],
    "Sistolik": [120, 125, 130, 135, 140, 145, 150, 122, 128, 133, 138, 142, 147, 152, 156, 160, 165],
    "Diyastolik": [80, 82, 85, 88, 90, 92, 95, 81, 84, 87, 89, 91, 94, 96, 98, 100, 102],
    "Ateş": [36.5, 37.0, 38.0, 38.5, 39.0, 39.5, 40.0, 36.2, 36.8, 37.4, 37.9, 38.3, 38.7, 39.1, 39.6, 40.0, 40.2],
    "Hasta_mı": [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
})

# 📊 Yeni özellikler ekleyelim
data["Kan Basıncı Farkı"] = data["Sistolik"] - data["Diyastolik"]
data["Kan Basıncı Oranı"] = data["Sistolik"] / data["Diyastolik"]
data["Ateş Değişim Hızı"] = (data["Ateş"] - data["Ateş"].shift(1)).fillna(0)

X = data[["Yaş", "Nabız", "Sistolik", "Diyastolik", "Kan Basıncı Farkı", "Kan Basıncı Oranı", "Ateş Değişim Hızı"]]
y = data["Hasta_mı"]

xgb_model.fit(X, y)

# 📊 GUI Penceresi
root = tk.Tk()
root.title("Hasta Tahmin Sistemi")
root.configure(bg="#2E4053")  # Koyu tema

# 📊 Tam ekran modu
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

FONT = ("Arial", 14)
ENTRY_BG = "#D6EAF8"
BTN_BG = "#1F618D"
BTN_FG = "white"

labels = ["Yaş:", "Nabız:", "Sistolik:", "Diyastolik:", "Ateş:"]
entries = {}

for i, label in enumerate(labels):
    tk.Label(root, text=label, font=FONT, bg="#2E4053", fg="white").grid(row=i, column=0, padx=20, pady=10, sticky="w")
    entry = tk.Entry(root, font=FONT, bg=ENTRY_BG, relief="solid", bd=2, width=10)
    entry.grid(row=i, column=1, padx=20, pady=10)
    entries[label[:-1]] = entry

# 📊 Girişlerin yalnızca sayı olmasını sağlayan fonksiyon
def validate_float(value):
    try:
        return float(value)
    except ValueError:
        return None

# 📊 Tahmin Fonksiyonu
def predict():
    user_data = {}
    for key, entry in entries.items():
        value = validate_float(entry.get())
        if value is None:
            messagebox.showerror("Hata", f"Lütfen geçerli bir sayı girin: {key}")
            return
        user_data[key] = value

    yeni_veri = pd.DataFrame({
        "Yaş": [user_data["Yaş"]],
        "Nabız": [user_data["Nabız"]],
        "Sistolik": [user_data["Sistolik"]],
        "Diyastolik": [user_data["Diyastolik"]],
        "Kan Basıncı Farkı": [user_data["Sistolik"] - user_data["Diyastolik"]],
        "Kan Basıncı Oranı": [user_data["Sistolik"] / user_data["Diyastolik"]],
        "Ateş Değişim Hızı": [0]
    })
    
    tahmin = xgb_model.predict(yeni_veri)
    result = "Hasta" if tahmin[0] == 1 else "Sağlıklı"
    messagebox.showinfo("Tahmin Sonucu", f"Sonuç: {result}")

# 📊 Grafik Çizme Fonksiyonu
def plot_data():
    user_data = {}
    for key, entry in entries.items():
        value = validate_float(entry.get())
        if value is None:
            messagebox.showerror("Hata", f"Lütfen geçerli bir sayı girin: {key}")
            return
        user_data[key] = value

    categories = list(user_data.keys())
    values = list(user_data.values())

    plt.figure(figsize=(8, 5))
    plt.style.use("ggplot")
    plt.bar(categories, values, color=["blue", "red", "green", "orange", "purple"])
    plt.title("Girilen Sağlık Verileri", fontsize=14, fontweight="bold")
    plt.xlabel("Özellikler", fontsize=12)
    plt.ylabel("Değerler", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

# 📊 Butonlar
predict_button = tk.Button(root, text="Tahmin Yap", command=predict, font=FONT, bg=BTN_BG, fg=BTN_FG, width=15, height=2)
predict_button.grid(row=5, column=1, pady=10)

plot_button = tk.Button(root, text="Grafik Göster", command=plot_data, font=FONT, bg=BTN_BG, fg=BTN_FG, width=15, height=2)
plot_button.grid(row=6, column=1, pady=10)

exit_button = tk.Button(root, text="Çıkış (ESC)", command=lambda: root.attributes("-fullscreen", False), font=FONT, bg="red", fg="white", width=15, height=2)
exit_button.grid(row=7, column=1, pady=10)

root.mainloop()