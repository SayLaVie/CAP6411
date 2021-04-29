import cv2
# from facenet_pytorch import MTCNN
import json
import mmcv
from model_pipeline import ModelPipeline
from PIL import Image

USER_DATABASE_FILE = "users.txt"

# load existing database file
with open(USER_DATABASE_FILE, 'r') as file:
    data = file.read()
users = json.loads(data)

# Create model pipeline
model = ModelPipeline()

# Enroll based on Video
test_vid = mmcv.VideoReader('enroll_shane3.mp4')
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in test_vid]
entry1 = model.create_enroll_entry(frames, "Shane")

# Enroll based on Video
#test_vid = mmcv.VideoReader('enroll_michael8.mp4')
#frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in test_vid]
#entry2 = model.create_enroll_entry(frames, "Michael")

users.update(entry1)
#users.update(entry2)

print(users.keys())

with open(USER_DATABASE_FILE, 'w') as file:
    file.write(json.dumps(users))
