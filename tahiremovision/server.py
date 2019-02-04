import os

from pathlib import Path
import subprocess

import fire
import responder

from .detector import Detector

LABELS = ['off', 'on']
api = responder.API()


def init_api(camera: dict, model: dict):
    camera_cmd = str(
        Path(os.path.dirname(__file__)) /
        '..' /
        'bin' /
        'camera.sh'
    )
    data_path = Path(camera['data_dir'])
    detector = Detector(**model)

    def camera_func() -> str:
        res = subprocess.run(
            [camera_cmd, str(data_path)],
            stdout=subprocess.PIPE
        )
        return res.stdout.decode()

    def detect_func(filename: str) -> int:
        target_file = str(data_path / filename)
        label, probs = detector.predict(target_file)
        return label

    @api.route('/camera')
    def camera_api(req, resp):
        filename = camera_func()
        resp.text = filename

    @api.route('/detect')
    def detect_api(req, resp):
        label = detect_func(camera_func())
        resp.media = {"status": LABELS[label]}


def server(camera: dict, model: dict):
    init_api(camera, model)
    api.run()
