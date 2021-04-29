import argparse
import jetson.utils
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import time

TRUE_TYPE = 'ttf/univers-medium-5871d3c7acb28.ttf'
FONT_SIZE = 14


def parse_args():
    parser = argparse.ArgumentParser(description="Demo real-time detection-recognition-tracking system", 
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    epilog=jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage())
    parser.add_argument("input_URI", type=str, default="/dev/video0", nargs='?', help="URI of the input stream")
    # parser.add_argument("output_URI", type=str, default="display://0", nargs='?', help="URI of the output stream")
    parser.add_argument("output_URI", type=str, default="troubleshoot.mp4", nargs='?', help="URI of the output stream")
    return parser.parse_known_args()[0]


def get_streams(opt):
    print(opt.input_URI)
    print(opt.output_URI)

    input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
    output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
    # output = jetson.utils.videoOutput(opt.output_URI, argv="--width=1920 --height=1080")

    return (input, output)


if __name__ == "__main__":
    opt = parse_args()
    input, output = get_streams(opt)

    # For some reason the streams need to be created before importing these libraries
    import cv2
    from model_pipeline import ModelPipeline

    # Create model pipeline
    model = ModelPipeline()

    # Enroll based on Video
    # test_vid = mmcv.VideoReader('enroll_michael4.mp4')
    # frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in test_vid]
    # model.enroll_user(frames, "Michael")

    # test_vid = mmcv.VideoReader('enroll_shane.mp4')
    # frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in test_vid]
    # model.enroll_user(frames, "Shane")

    model.load_user_database("users.txt")

    zeros = np.zeros(10)
    recent_times = np.ascontiguousarray(zeros)

    frame_count = 0
    recent_avg_time = 0
    cummulative_process_time = []
    t_begin = time.time()
    # process frames until the user exits
    while True:
        # capture the next image
        img = input.Capture()
        frame = Image.fromarray(cv2.cvtColor(jetson.utils.cudaToNumpy(img), cv2.COLOR_BGR2RGB))
        frame_count = frame_count + 1

        t_1 = time.time()
        annotated = model.annotate_apply_tracking(frame)#.resize((640, 480))
        # annotated = annotated.resize((1920, 1080))
        t_2 = time.time()

        cummulative_process_time.append(t_2 - t_1)
        recent_times[frame_count % 10] = t_2 - t_1
        rolling_avg = np.sum(recent_times) / 10

        # Add Processing time to image
        draw = ImageDraw.Draw(annotated)
        draw.text(
            (10, 10),
            str(rolling_avg),
            # str(t_2 - t_1),
            font=ImageFont.truetype(TRUE_TYPE, FONT_SIZE), fill=(0, 0, 0))

        img = jetson.utils.cudaFromNumpy(np.array(annotated))

        # print("Process time: {}".format(t_2 - t_1))
        # print()
        output.Render(img)

        # exit on input/output EOS
        if not input.IsStreaming() or not output.IsStreaming():
            t_end = time.time()
            process_time = sum(cummulative_process_time)
            total_time = t_end - t_begin
            # print("Processed {} frames in {} seconds".format(frame_count, total_time))
            print("Frames: {}".format(frame_count))
            print("Total end-to-end time: {}".format(total_time))
            print("Cummulative process time: {}".format(process_time))
            print("End-to-end speed: {} FPS".format(frame_count / total_time))
            print("Process speed: {} FPS".format(frame_count / process_time))
            break
