from ultralytics import YOLO
def main():
    model = YOLO("yolov8l.pt")

    model.train(
        data="../dataset/data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        lr0=0.001,
        workers=4,
        patience=20
    )


if __name__ == "__main__":
    main()

