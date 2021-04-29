from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
# from IPython import display
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sort import Sort
import time
import torch
from torchvision import transforms

TRUE_TYPE = 'ttf/univers-medium-5871d3c7acb28.ttf'
FONT_SIZE = 14

# INPUT_SIZE must match command-line argument
# INPUT_SIZE = (640, 480)
INPUT_SIZE = (800, 600)
# INPUT_SIZE = (1920, 1080)
# DETECTION_SIZE = (600, 450)
DETECTION_SIZE = (400, 300)
# DETECTION_SIZE = (160, 120)

X_SCALE = INPUT_SIZE[0] / DETECTION_SIZE[0]
Y_SCALE = INPUT_SIZE[1] / DETECTION_SIZE[1]
BUFFER = 0.0
RESIZE_FACTOR = np.array([X_SCALE - BUFFER, Y_SCALE - BUFFER, X_SCALE + BUFFER, Y_SCALE + BUFFER])


class ModelPipeline:
    def __init__(self, recognition_threshhold=0.35):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detect_model = MTCNN(keep_all=True, device=self.device, post_process=False, min_face_size=20)
        self.face_labels = {}
        self.frames_per_track = {}
        self.prev_dets = None
        self.rec_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.rec_thresh = recognition_threshhold
        self.stride = 0
        self.track_embeddings = {}
        self.tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.3)
        self.users_embeddings = []
        self.unknown_idx = 0

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

    # old way
    # def create_enroll_entry(self, images, name):
    #     embeddings = self.extract_embeddings(images)

    #     if embeddings is None:
    #         return None
        
    #     avg_embedding = torch.mean(embeddings, dim=0).cpu().numpy().tolist()
    #     return {name: avg_embedding}

    # new way
    def create_enroll_entry(self, images, name):
        embeddings = []
        detect_model = MTCNN(keep_all=False, device=self.device, post_process=False, min_face_size=20)

        for im in images:
            # face = self.detect_model(im)
            face = detect_model(im)

            if face is not None:
                # embedding = self.rec_model(face.to(self.device)).detach()
                embedding = self.rec_model(face[None, :, :, :].to(self.device)).detach()
                embeddings.append(embedding)

        if (embeddings == []):
            return None

        avg_embedding = torch.mean(torch.stack(embeddings), dim=0).cpu().numpy().tolist()
        return {name: avg_embedding}

    def load_user_database(self, file):
        with open(file, 'r') as file:
            data = file.read()
        users = json.loads(data)
        for user, val in users.items():
            embedding = torch.from_numpy(np.array(val)).to(self.device)
            self.users_embeddings.append((user, embedding))

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
            # match_name = "Unknown_"+str(self.unknown_idx)
            match_name = "Imposter"
            # self.unknown_idx += 1
            # self.users_embeddings.append((match_name, embedding))
        # print(best_match)
        return match_name

    # Rather than annotating the frame, return the boxes and the associated names 
    # (first list is boxes, second list is names)
    def get_boxes(self, frame):
        boxes, _ = self.detect_model.detect(frame)
        embeddings = self.extract_embeddings([frame])

        names = []
        if (embeddings is not None):
            for i in range(embeddings.shape[0]):
                name = self.identify_user(embeddings[i])
                names.append(name)

        return boxes, names

    # Begins tracking from this frame using then normal detection + recognition pipeline
    # Returns boxes and names for each detection
    def begin_track(self, frame):
        boxes, names = self.get_boxes(frame)
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
    def annotate_begin_track(self, frame, font_size=FONT_SIZE, box_size=1):
        boxes, names = self.begin_track(frame)

        annotated = frame.copy()
        draw = ImageDraw.Draw(annotated)

        if (boxes is not None):
            for i in range(len(boxes)):
                draw.rectangle(boxes[i].tolist(), outline=(0, 255, 0), width=box_size)
                draw.text((boxes[i][0], boxes[i][3]), names[i], font=ImageFont.truetype(TRUE_TYPE, font_size), fill=(0, 255, 0))

        # return annotated.resize(output_image, Image.BILINEAR)
        return annotated

    # Continue tracking with this as next frame
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
    def annotate_apply_tracking(self, frame, font_size=FONT_SIZE, box_size=1):
        detections = self.prev_dets

        if (self.stride % 4 == 0):
            detections = self.apply_tracking(frame.resize(DETECTION_SIZE))
            #detections = self.apply_tracking(frame)
            self.prev_dets = detections
            self.stride = self.stride + 1

        self.stride = self.stride + 1
        annotated = frame.copy()
        draw = ImageDraw.Draw(annotated)

        for j in range(len(detections)):
            detection_box = np.multiply(detections[j][:4], RESIZE_FACTOR)
            #detection_box = detections[j][:4]
            track_id = detections[j][4]

            num_frames = self.frames_per_track.get(track_id, 0) + 1
            self.frames_per_track[track_id] = num_frames
            label = self.face_labels.get(track_id)

            if label is None or num_frames % 7 == 0:
                embedding_list = self.track_embeddings.get(track_id, [])
                if len(embedding_list) < 5:
                    face = extract_face(frame, detection_box)
                    embedding = self.rec_model(face[None, :, :, :].to(self.device)).detach()
                    if embedding is not None:
                        embedding_list.append(embedding)
                        self.track_embeddings[track_id] = embedding_list
                        avg_embedding = torch.mean(torch.stack(embedding_list), dim=0)
                        name = self.identify_user(avg_embedding)
                        self.face_labels.update({track_id: name})

            draw.rectangle(
                [detection_box[0], detection_box[1], detection_box[2], detection_box[3]],
                outline=(0, 255, 0), width=box_size)

            draw.text(
                (detection_box[0], detection_box[3]),
                str(self.face_labels.get(track_id, "bad_embedding") + "_" + str(track_id)),
                font=ImageFont.truetype(TRUE_TYPE, font_size), fill=(0, 255, 0))

        return annotated
