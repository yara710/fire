from flask import Flask, render_template, request, send_from_directory
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO("bestt.pt")

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Perform YOLO detection
        results = model(filepath)
        for r in results:
            output_path = os.path.join(OUTPUT_FOLDER, "output.jpg")
            r.save(output_path)  # Save detected image
        
        return render_template("index.html", original=filepath, output=output_path)

    return render_template("index.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)
