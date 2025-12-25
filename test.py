from vllama.core import *
from vllama.functions.object_detection_video.object_detection_video import object_detection_video
from vllama.functions.image3d.image3dRemote import run_kaggle_image_to_3d

if __name__ == "__main__":
    # text_to_speech("Hello world this is Manvith building the text to speech module with in built library of python")
    # text_to_speech()
    # text_to_speech("exit")

    # list_downloads()
    # translation = translate_fast("hello world this is your boy from bangalore india building vllama which helps in translation. I have built a framework called vllama which helps all in translation")
    # print(translation)

    # object_detection_video("outputs/test_video.mp4", output_dir="outputs")
    run_kaggle_image_to_3d(
        image_path="outputs/room_4.jpeg",
        output_dir="outputs/3d_models"
    )