import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog='OldDocumentTranscriptor',
        description='Unicam Group Project JAL',
        epilog=''
    )
    parser.add_argument('input_dir', metavar='input_dir', type=str)
    parser.add_argument('output_dir', metavar='output_dir', type=str)
    parser.add_argument('-t', '--tresholding', action='store_true', help="Tresholding denoising", required=False)
    parser.add_argument('-e', '--edgedetection', action='store_true', help="Edge Detection Denoising", required=False)
    parser.add_argument('-s', '--standardmodel', action='store_true', help="Ocr whit standard EasyOCR model", required=False)
    parser.add_argument('-w', '--handwrittenmodel', action='store_true', help="Ocr whit model for handwritten documents", required=False)
    parser.add_argument('-m', '--machinewrittenmodel', action='store_true', help="Ocr whit model for machinewritten documents", required=False)
    args = parser.parse_args()
    if args.tresholding and args.edgedetection:
        parser.error('Only one denoising technique can be selected')
    if args.standardmodel + args.handwrittenmodel + args.machinewrittenmodel >= 2:
        parser.error('Only one ocr model can be selected')
    return args
