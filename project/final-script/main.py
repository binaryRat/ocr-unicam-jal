"""Main Script"""
import os.path

import parser
import denoising
import ocr
import utils

images = []
denoised = []
transcriptions = []

args = parser.get_args()
input_dir = args.input_dir
output_dir = args.output_dir
denoising_technique = "none"
ocr_technique = "none"


if __name__ == '__main__':
    images = utils.load_images(input_dir)
    den_path = None
    # denoising
    if args.tresholding:
        denoising_technique = "adaptive tresholding"
        denoised = denoising.adaptive_treshold(images)
        den_path = os.path.join(output_dir, "adaptive-tresholding-results")
    if args.edgedetection:
        denoising_technique = "edge detection"
        denoised = denoising.edge_detection(images)
        utils.save_images(denoised, output_dir + "/denoised/edge_detection")
        den_path = os.path.join(output_dir, "edge-detection-results")
    if not os.path.exists(den_path) and den_path is not None:
        os.mkdir(den_path)
    utils.save_images(denoised, den_path)

    # ocr
    if args.ocr:
        transcription_path = os.path.join(output_dir, "ocr-transcriptions")
        if not os.path.exists(transcription_path):
            os.mkdir(transcription_path)
        ocr_technique = "easy-ocr whit retrained model"
        for img in images:
            transcriptions.append(ocr.img_to_text(img))
        utils.save_ocr_result(transcriptions, transcription_path, True)

    print("Input dir: " + input_dir)
    print("Output dir: " + output_dir)
    print("Denoising: " + denoising_technique, ",", "Ocr: " + ocr_technique)
    if args.ocr:
        print("Transcription numbers: " + str(len(transcriptions)))
