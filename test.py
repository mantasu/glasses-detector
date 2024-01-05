from src.glasses_detector import GlassesDetector

if __name__ == "__main__":
    detector = GlassesDetector(size="medium", pretrained=False)
    print(detector.model)
