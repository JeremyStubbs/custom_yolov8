from ultralytics import YOLO

def main():
    model = YOLO('yolov8l.pt')
    results = model.track(source=0, show=True)

if __name__ == '__main__':
    main()