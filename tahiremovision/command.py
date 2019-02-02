from .detector import Detector

def command(input_file, model_file, input_size):
    detector = Detector(model_file, input_size)
    result = detector.predict(input_file)
    return result
