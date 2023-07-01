import services
import ocr

hand_written = services.load_img('input/hand')
machine_written = services.load_img('input/machine')

hand_written_by_char = []
machine_written_by_char = []
hand_written_by_words = []
machine_written_by_words = []

for img in hand_written:
    hand_written_by_char.append(ocr.img_to_img_by_char(img))
    hand_written_by_words.append(ocr.img_to_img_by_words(img))

for img in machine_written:
    machine_written_by_char.append(ocr.img_to_img_by_char(img))
    machine_written_by_words.append(ocr.img_to_img_by_words(img))

services.save_images(hand_written_by_char, "output/hand/by-char")
services.save_images(hand_written_by_words, "output/hand/by-words")
services.save_images(machine_written_by_char, "output/machine/by-char")
services.save_images(machine_written_by_words, "output/machine/by-words")


