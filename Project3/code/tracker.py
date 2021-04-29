from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import time
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw, ImageFont
from IPython import display
from sort import Sort


class ModelPipeline:
    def __init__(self, recognition_threshhold = 0.7):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detect_model = MTCNN(keep_all=True, device=self.device, post_process=False)
        self.rec_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.users_embeddings = []
        self.unknown_idx = 0
        self.rec_thresh = recognition_threshhold

    # Detect faces and extract feature embeddings for each face in each image
    def extract_embeddings(self, images):
        embeddings = []

        for im in images:
            faces = self.detect_model(im)
            if faces is not None:
                embedding = self.rec_model(faces.to(self.device)).detach()

                for e in embedding:
                    embeddings.append(e)

        if (embeddings == []):
            return None

        return torch.stack(embeddings)

    # Compute distance between feature embeddings
    def embedding_dist(self, em1, em2):
        return (em1 - em2).norm()

    # Assumes all images contain only a single face, of the target user
    def enroll_user(self, images, name):
        embeddings = self.extract_embeddings(images)

        if (embeddings is None):
            return

        avg_embedding = torch.mean(embeddings, dim=0)
        self.users_embeddings.append((name, avg_embedding))

    # Returns name of the user for the given embedding
    def identify_user(self, embedding):
        # Find most similar user
        best_match = 1e6
        match_name = "N/A"
        for u in self.users_embeddings:
            dist = self.embedding_dist(u[1], embedding)
            if ((dist < self.rec_thresh) and (dist < best_match)):
                best_match = dist
                match_name = u[0]

        # If unknown user, auto enroll as new user from this embedding
        if (best_match == 1e6):
            match_name = "Unknown_"+str(self.unknown_idx)
            self.unknown_idx += 1
            self.users_embeddings.append((match_name, embedding))

        return match_name

    # Annotate single frame
    def annotate_frame(self, frame, font_size = 14, box_size = 1, output_image=(256, 144)):
        boxes, _ = self.detect_model.detect(frame)
        embeddings = self.extract_embeddings([frame])

        names = []
        if (embeddings is not None):
            for i in range(embeddings.shape[0]):
                name = self.identify_user(embeddings[i])
                names.append(name)

        annotated = frame.copy()
        draw = ImageDraw.Draw(annotated)

        if (boxes is not None):
            for i in range(len(boxes)):
                draw.rectangle(boxes[i].tolist(), outline=(0, 255, 0), width=box_size)
                draw.text((boxes[i][0], boxes[i][3]), names[i], font=ImageFont.truetype('./univers-medium-5871d3c7acb28.ttf', font_size), fill=(0, 255, 0))

        return annotated.resize(output_image, Image.BILINEAR)

    # Rather than annotating the frame, return the boxes and the associated names (first list is boxes, second list is names)
    def get_boxes(self, frame):
        boxes, _ = self.detect_model.detect(frame)
        embeddings = self.extract_embeddings([frame])

        names = []
        if (embeddings is not None):
            for i in range(embeddings.shape[0]):
                name = self.identify_user(embeddings[i])
                names.append(name)

        return boxes, names

    # Perform normal detection on first frame, then tracking on all subsequent frames
    # Returns list of boxes+ids, and names 
    def detect_and_track(self, frames):
        boxes, names = self.get_boxes(frames[0])

        if boxes is None:
            return [], []

        tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.3)

        # Initalize tracker
        inital_detections = []

        for i in range(len(boxes)):
            inital_detections.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], 1])

        inital_detections = np.asarray(inital_detections)
        tracker.update(inital_detections)

        # Apply tracking on future frames
        detections = [inital_detections]
        for i in range(1, len(frames)):
            boxes, score = self.detect_model.detect(frames[i])

            dets = []
            if (boxes is not None):
                for i in range(len(boxes)):
                    dets.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], score])
            else:
                dets = np.empty((0, 5))

            dets = np.asarray(dets)
            detections.append(tracker.update(dets))

        return detections, names

    # Runs detect and track, but returns the annotated frames instead
    def annotate_detect_and_track(self, frames, font_size = 14, box_size = 1, output_image=(256, 144)):
        detections, names = self.detect_and_track(frames)

        annotated_frames = []
        for i in range(len(detections)):
            annotated = frames[i].copy()
            draw = ImageDraw.Draw(annotated)

            for j in range(len(detections[i])):
                draw.rectangle([detections[i][j][0], detections[i][j][1], detections[i][j][2], detections[i][j][3]], outline=(0, 255, 0), width=box_size)
                draw.text((detections[i][j][0], detections[i][j][3]), names[min(int(detections[i][j][4])-1, len(names)-1)], font=ImageFont.truetype('./UniversCondensed.ttf', font_size), fill=(0, 255, 0))

            annotated_frames.append(annotated.resize(output_image, Image.BILINEAR)) 

        return annotated_frames

    # Begins tracking from this frame using then normal detection + recognition pipeline
    # Returns boxes and names for each detection
    def begin_track(self, frame):
        boxes, names = self.get_boxes(frames[0])
        self.tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.3)
        self.tracked_names = {}

        if boxes is None:
            return [], []

        for i in range(len(names)):
            self.tracked_names[i] = names[i]

        inital_detections = []
        for i in range(len(boxes)):
            inital_detections.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], 1])

        inital_detections = np.asarray(inital_detections)
        self.tracker.update(inital_detections)

        return boxes, names

    # Runs begin track, but returns the annotated frame instead
    def annotate_begin_track(self, frame, font_size = 14, box_size = 1, output_image=(256, 144)):
        boxes, names = self.begin_track(frame)

        annotated = frame.copy()
        draw = ImageDraw.Draw(annotated)

        if (boxes is not None):
            for i in range(len(boxes)):
                draw.rectangle(boxes[i].tolist(), outline=(0, 255, 0), width=box_size)
                draw.text((boxes[i][0], boxes[i][3]), names[i], font=ImageFont.truetype('./UniversCondensed.ttf', font_size), fill=(0, 255, 0))

        return annotated.resize(output_image, Image.BILINEAR)

    # Continue tracking with this as next frame (must have previously called begin_track)
    # Returns list of boxes+ids that are detected
    def apply_tracking(self, frame):
        boxes, score = self.detect_model.detect(frame)

        dets = []
        if (boxes is not None):
            for i in range(len(boxes)):
                dets.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], score])
        else:
            dets = np.empty((0, 5))

        dets = np.asarray(dets)
        return self.tracker.update(dets)

    # Runs apply tracking, but returns the annotated frame instead
    def annotate_apply_tracking(self, frame, font_size = 14, box_size = 1, output_image=(256, 144)):
        detections = self.apply_tracking(frame)

        annotated = frame.copy()
        draw = ImageDraw.Draw(annotated)

        for j in range(len(detections)):
            draw.rectangle([detections[j][0], detections[j][1], detections[j][2], detections[j][3]], outline=(0, 255, 0), width=box_size)
            draw.text((detections[j][0], detections[j][3]), self.tracked_names[min(int(detections[j][4])-1, len(self.tracked_names)-1)], font=ImageFont.truetype('./UniversCondensed.ttf', font_size), fill=(0, 255, 0))

        return annotated.resize(output_image, Image.BILINEAR)


if __name__ == "__main__":
    # Create model pipeline
    model = ModelPipeline()

    # Enroll Shane based on Video
    test_vid = mmcv.VideoReader('enroll_shane.mp4')
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in test_vid]
    model.enroll_user(frames, "Shane")

    # Load raw video for annotation
    test_vid = mmcv.VideoReader('test_shane.mp4')
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in test_vid]
    
    # Annotate video
    print("Starting: ", len(frames))
    annotated_frames = []
    cnt = 0
    run_time = 0
    print("Processed: 0", end="")
    for frame in frames:
        if (frame is None):
            break
        s = time.time()

        # Annotate the frame
        annotated = model.annotate_frame(frame)

        # Save stats and append frame to list
        t = time.time() - s
        annotated_frames.append(annotated)
        cnt += 1
        run_time += t
        print("\rProcessed: ", cnt, " ", t, end="")
    print("\nFinished in ", run_time)

    # Save annotated video
    dim = annotated_frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
    video_tracked = cv2.VideoWriter('test_annotated.mp4', fourcc, 25.0, dim)
    for frame in annotated_frames:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()
    

    # Repeat annotation process using tracker for all frames after first
    s = time.time()
    annotated_frames = model.annotate_detect_and_track(frames)
    print("----------------  Annotated via tracking in: ", (time.time() - s))

    # Save annotated video
    dim = annotated_frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
    video_tracked = cv2.VideoWriter('test_annotated_tracking.mp4', fourcc, 25.0, dim)
    for frame in annotated_frames:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()
    
    # Repeat annotation process, doing normal detection then tracking for 5 frames and repeating. 
    # (1 frame detection, 4 frames tracking, 1 frame dec, 4 frames track, ...)
    
    s = time.time()
    annotated_frames = []
    for i in range(len(frames)):
        frame = frames[i]
        if (frame is None):
            break

        # First of 5 frames is detection
        if (i%5 == 0):
            annotated = model.annotate_begin_track(frame)
            annotated_frames.append(annotated)
        
        # Remaining 4 frames is tracking
        else:
            annotated = model.annotate_apply_tracking(frame)
            annotated_frames.append(annotated)
    print("----------------  Annotated via 1 Frame detect, 4 Frames tracking in: ", (time.time() - s))

    # Save annotated video
    dim = annotated_frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
    video_tracked = cv2.VideoWriter('test_annotated_spaced_tracking.mp4', fourcc, 25.0, dim)
    for frame in annotated_frames:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()


