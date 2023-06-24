"""Main Script"""
import sys
import numpy
import parser
import denoising
import ocr
import utils

denoising_flag = False
ocr_flag = False
images = []
denoised = []
transcriptions = []

args = parser.get_args()
input_dir = args.input_dir
output_dir = args.output_dir
denoising_flag = args.denoising
ocr_flag = args.ocr


if __name__ == '__main__':
    print("Input dir: " + input_dir)
    print("Output dir: " + output_dir)
    print("Denoising: " + str(denoising_flag), ",", "Ocr: " + str(ocr_flag))
    images = utils.load_images(input_dir)
    if(denoising_flag == True):
        denoised = denoising.denoise(images)
        utils.save_images(denoised, output_dir+"/denoised")
    if(ocr_flag == True):
        for img in images:
            transcriptions.append(ocr.img_to_text(img))
    print("Transcription numbers: " + str(len(transcriptions)))
    utils.save_ocr_result(transcriptions, "output/transcriptions", True)



