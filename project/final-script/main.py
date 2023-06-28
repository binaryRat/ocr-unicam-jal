"""Main Script"""
import parser
import denoising
import ocr
import utils

tresholding = False
edge_detection = False
ocr_flag = False
images = []
denoised = []
transcriptions = []

args = parser.get_args()
input_dir = args.input_dir
output_dir = args.output_dir
tresholding = args.tresholding
edge_detection = args.edgedetection
ocr_flag = args.ocr
denoising_technique = ""
ocr_technique = ""


if __name__ == '__main__':
    images = utils.load_images(input_dir)
    if(tresholding):
        denoising_technique = "adaptive tresholding"
        denoised = denoising.adaptive_treshold(images)
        utils.save_images(denoised, output_dir+"/denoised/adaptive_treshold")
    if(edge_detection):
        denoising_technique = "edge detection"
        denoised = denoising.edge_detection(images)
        utils.save_images(denoised, output_dir + "/denoised/edge_detection")
    if(ocr_flag):
        ocr_technique = "easy-ocr whit retrained model"
        for img in images:
            transcriptions.append(ocr.img_to_text(img))
        utils.save_ocr_result(transcriptions, "output/transcriptions", True)
    print("Input dir: " + input_dir)
    print("Output dir: " + output_dir)
    print("Denoising: " + denoising_technique, ",", "Ocr: " + ocr_technique)
    if ocr_flag:
        print("Transcription numbers: " + str(len(transcriptions)))




