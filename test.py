from vllama.core import *
from vllama.functions.object_detection_video.object_detection_video import object_detection_video

if __name__ == "__main__":
    # text_to_speech("Hello world this is Manvith building the text to speech module with in built library of python")
    # text_to_speech()
    # text_to_speech("exit")

    # list_downloads()
    # translation = translate_fast("hello world this is your boy from bangalore india building vllama which helps in translation. I have built a framework called vllama which helps all in translation")
    # print(translation)

    object_detection_video("outputs/test_video.mp4", output_dir="outputs")