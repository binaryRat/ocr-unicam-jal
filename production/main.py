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
denoising_flag = False
ocr_technique = "none"
ocr_flag = False


if __name__ == '__main__':
    images = utils.load_images(input_dir)
    denoising_path = None
    # denoising
    if args.tresholding:
        denoising_flag = True
        denoising_technique = "adaptive tresholding"
        denoised = map(denoising.adaptive_treshold, images)
        denoising_path = os.path.join(output_dir, "adaptive-tresholding-results")
    if args.edgedetection:
        denoising_flag = True
        denoising_technique = "edge detection"
        denoised = map(denoising.edge_detection, images)
        denoising_path = os.path.join(output_dir, "edge-detection-results")

    if denoising_flag:
        if not os.path.exists(denoising_path) and denoising_path is not None:
            os.mkdir(denoising_path)
        utils.save_images(denoised, denoising_path)

    # ocr
    transcriptions_path = None
    if args.standardmodel:
        transcriptions_path = os.path.join(output_dir, "ocr-transcriptions-standard-model")
        ocr_flag = True
        ocr_technique = "easy-ocr whit the EasyOCR standard model"
        for img in images:
            transcriptions.append(ocr.easy_ocr_standard_model(img))
    elif args.machinewrittenmodel:
        transcriptions_path = os.path.join(output_dir, "ocr-transcriptions-machine-model")
        ocr_flag = True
        ocr_technique = "easy-ocr whit retrained model for machine written documents"
        for img in images:
            transcriptions.append(ocr.custom_model_machine_written(img))
    elif args.handwrittenmodel:
        transcriptions_path = os.path.join(output_dir, "ocr-transcriptions-hand-model")
        ocr_flag = True
        ocr_technique = "easy-ocr whit retrained model for hand written documents"
        for img in images:
            transcriptions.append(ocr.custom_model_hand_written(img))

    if ocr_flag:
        if not os.path.exists(transcriptions_path):
            os.mkdir(transcriptions_path)
        utils.save_ocr_result(transcriptions, transcriptions_path, unified=True)

    print("Input dir: " + input_dir)
    print("Output dir: " + output_dir)
    print("Denoising: " + denoising_technique, ",", "Ocr: " + ocr_technique)
    if ocr_flag:
        print("Transcription numbers: " + str(len(transcriptions)))
