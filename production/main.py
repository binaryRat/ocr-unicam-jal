"""Main script to be executed"""
import os.path
import cv2

import denoising
import ocr
import parser
import utils

args = parser.get_args()
input_dir = args.input_dir
output_dir = args.output_dir
denoising_technique = "none"
denoising_flag = False
ocr_technique = "none"
ocr_flag = False

if __name__ == '__main__':
    print("Input dir: " + input_dir)
    print("Output dir: " + output_dir)

    for filename in os.listdir(input_dir):
        # denoising
        denoising_path = output_dir
        image = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if args.tresholding:
            denoising_flag = True
            denoising_technique = "adaptive tresholding"
            image = denoising.adaptive_treshold(image)
            filename = utils.concatenate_filename(filename, "_adaptiveTresholding")
            denoising_path = os.path.join(denoising_path, "adaptive-tresholding-results")
        elif args.edgedetection:
            denoising_flag = True
            denoising_technique = "edge detection"
            image = denoising.edge_detection(image)
            filename = utils.concatenate_filename(filename, "_edgeDetection")
            denoising_path = os.path.join(denoising_path, "edge-detection-results")
        if denoising_flag:
            if not os.path.exists(denoising_path) and denoising_path is not None:
                os.mkdir(denoising_path)
            denoising_path = os.path.join(denoising_path, filename)
            utils.save_image(image, denoising_path)
        # ocr
        transcriptions_path = output_dir
        transcription = None
        if args.standardmodel:
            transcriptions_path = os.path.join(output_dir, "ocr-transcriptions-standard-model")
            ocr_flag = True
            ocr_technique = "easy-ocr whit the EasyOCR standard model"
            transcription = ocr.easy_ocr_standard_model(image)
            filename = utils.concatenate_filename(filename, "_standardModel")
        elif args.machinewrittenmodel:
            transcriptions_path = os.path.join(output_dir, "ocr-transcriptions-machine-model")
            ocr_flag = True
            ocr_technique = "easy-ocr whit retrained model for machine written documents"
            transcription = ocr.custom_model_machine_written(image)
            filename = utils.concatenate_filename(filename, "_machineWrittenModel")
        elif args.handwrittenmodel:
            transcriptions_path = os.path.join(output_dir, "ocr-transcriptions-hand-model")
            ocr_flag = True
            ocr_technique = "easy-ocr whit retrained model for hand written documents"
            transcription = ocr.custom_model_hand_written(image)
            filename = utils.concatenate_filename(filename, "_handWrittenModel")
        if ocr_flag:
            if not os.path.exists(transcriptions_path):
                os.mkdir(transcriptions_path)
            transcriptions_path = os.path.join(transcriptions_path, utils.get_filename(filename) + ".txt")
            ocr.save_ocr_result(transcription, transcriptions_path, united=False)
    print("Denoising: " + denoising_technique)
    print("Ocr: " + ocr_technique)
