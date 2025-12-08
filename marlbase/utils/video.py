import numpy as np
import imageio
import cv2


class VideoRecorder:
    def __init__(self, fps=30, resolution=(128, 128), use_opencv=False):
        self.fps = fps
        self.resolution = resolution
        self.frames = []
        self.use_opencv = use_opencv
        self.video_writer = None

    def reset(self):
        self.frames = []
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def record_frame(self, env):
        frame = None
        try:
            frame = env.unwrapped.render()
        except Exception as e:
            print(f"[WARNING] env.render() failed: {e}")

        # Fallback: solid white frame if invalid
        if isinstance(frame, bool) or frame is None:
            frame = np.full((*self.resolution, 3), 255, dtype=np.uint8)

        elif isinstance(frame, np.ndarray):
            # Convert bool → uint8 image
            if frame.dtype == np.bool_:
                frame = frame.astype(np.uint8) * 255
            elif frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            # Convert grayscale to RGB
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.ndim == 3 and frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)

            # Resize if needed
            frame = cv2.resize(frame, self.resolution)
 

        else:
            print(f"[WARNING] Unsupported frame type: {type(frame)}. Using blank.")
            frame = np.full((*self.resolution, 3), 255, dtype=np.uint8)

        self.frames.append(frame)

    def save(self, filename="out.mp4"):
        if not self.frames:
            print("[WARNING] No frames to save.")
            return

        if self.use_opencv:
        
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, self.fps, self.resolution)
            for frame in self.frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"[INFO] Saved video via OpenCV: {filename}")
        else:
            imageio.mimsave(filename, self.frames, fps=self.fps)
            print(f"[INFO] Saved video via imageio: {filename}")
