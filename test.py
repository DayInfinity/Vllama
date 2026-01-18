from vllama.core import *
from vllama.functions.object_detection_video.object_detection_video import object_detection_video
from vllama.functions.image3d.image3dRemote import run_kaggle_image_to_3d
from vllama.functions.viewer3d.viewer3d import view_3d_model
from vllama.functions.video3d.video3dRemote import run_kaggle_video_to_3d


def main():
    
    try:
        result_ply = run_kaggle_video_to_3d(
            video_path= "input_videos/input_1.mp4",
            output_dir="outputs/video_3d_models",
            frame_interval=10,
            max_frames_per_batch=25,
            total_frames_limit=100
        )
        
        print(f"\n🎉 Success! Your 3D model is ready at: {result_ply}")
        print("\nYou can view it using:")
        print("  - MeshLab")
        print("  - CloudCompare")
        print("  - Blender")
        print("  - Any PLY viewer")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    # text_to_speech("Hello world this is Manvith building the text to speech module with in built library of python")
    # text_to_speech()
    # text_to_speech("exit")

    # list_downloads()
    # translation = translate_fast("hello world this is your boy from bangalore india building vllama which helps in translation. I have built a framework called vllama which helps all in translation")
    # print(translation)

    # object_detection_video("outputs/test_video.mp4", output_dir="outputs")
    # run_kaggle_image_to_3d(
    #     image_path="outputs/room_4.jpeg",
    #     output_dir="outputs/3d_models"
    # )
    # view_3d_model(model_path="outputs/3d_models/outputs/image_to_3d_20251225_170416.ply")
    main()
