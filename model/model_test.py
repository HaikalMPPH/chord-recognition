from ultralytics import YOLO
import os

dirs = os.listdir("../dataset/decorative-plants/test/")
dirs.sort()

model = YOLO("./model-trained.pt")
#results = model("../Dataset/DecorativePlants/test/Calathea/17.jpg")
results = model("https://nouveauraw.com/wp-content/uploads/2020/01/Pothos-Golden-Pothos-plant-800-great-coloring.png")
for result in results:
    print(f"\nPredict -> {dirs[result.probs.top1]}")
