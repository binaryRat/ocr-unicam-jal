import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog='OldDocumentTranscriptor',
        description='Unicam Group Project JAL',
        epilog='Text at the bottom of help'
    )
    parser.add_argument('input_dir', metavar='input_dir', type=str)
    parser.add_argument('output_dir', metavar='output_dir', type=str)
    parser.add_argument('-t', '--tresholding', action='store_true', help="Tresholding flag", required=False)
    parser.add_argument('-e', '--edgedetection', action='store_true', help="Denoising flag", required=False)
    parser.add_argument('-o', '--ocr', action='store_true', help="OCR flag", required=False)
    args = parser.parse_args()
    if args.tresholding and args.edgedetection:
        parser.error('Only one denoising technique can be selected')
    return args
