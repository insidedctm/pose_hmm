import cv2

class VideoWriter:
  def __init__(self, video, output_path):
    self.cv2_writer = None
    self.output_path = output_path
    self.fps = video.get(cv2.CAP_PROP_FPS)

  def __call__(self, frame):
    if self.output_path:
      if not self.cv2_writer:
        h, w = frame.shape[0], frame.shape[1]
        self.cv2_writer = create_cv2_writer(
                self.output_path, 
                self.fps,
                w,
                h
        )
      self.cv2_writer.write(frame)

  def release(self):
    if self.cv2_writer:
      self.cv2_writer.release()

def create_cv2_writer(path, fps, w, h):
  print(f'create VideoWriter for {path}')
  return cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))
