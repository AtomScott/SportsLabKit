import threading
from queue import Queue
from time import sleep

import cv2
import numpy as np


class VideoReader:
    """Pythonic wrapper around OpenCV's VideoCapture().

    This class provides a convenient way to access and manipulate video files using OpenCV's VideoCapture object. It implements several convenient methods and properties to make it easy to work with video files, including slicing and indexing into the video file, iteration through the video frames, and more.

    Args:
        filename (str): The path to the video file.
        threaded (bool): Whether to run the video reading in a separate thread.
        queue_size (int): The size of the queue for storing video frames.

    Properties:
        frame_width (int): The width of the video frames.
        frame_height (int): The height of the video frames.
        frame_channels (int): The number of channels in the video frames.
        frame_rate (float): The frame rate of the video.
        frame_shape (tuple): The shape of the video frames (height, width, channels).
        number_of_frames (int): The total number of frames in the video.
        fourcc (int): The fourcc code of the video.
        current_frame_pos (int): The current position of the video frame.

    Methods:
        read(frame_number=None): Read the next frame or a specified frame from the video.
        close(): Close the video file.
    """

    def __init__(self, filename: str, threaded=False, queue_size=10):
        """Open video in filename."""
        self._filename = filename
        self._vc = cv2.VideoCapture(str(self._filename))
        self.threaded = threaded

        self.stopped = False
        self.q: Queue = Queue(maxsize=queue_size)

        if threaded:
            t = threading.Thread(target=self.read_thread)
            t.daemon = True
            t.start()

    def __del__(self):
        try:
            self._vr.release()
        except AttributeError:  # if file does not exist this will be raised since _vr does not exist
            pass

    def __len__(self):
        """Length is number of frames."""
        return self.number_of_frames

    def __getitem__(self, index):
        # numpy-like slice imaging into arbitrary dims of the video
        # ugly.hacky but works
        frames = None
        if isinstance(index, int):  # single frame
            ret, frames = self.read(index)
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
        elif isinstance(index, slice):  # slice of frames
            frames = np.stack([self[ii] for ii in range(*index.indices(len(self)))])
        elif isinstance(index, range):  # range of frames
            frames = np.stack([self[ii] for ii in index])
        elif isinstance(index, tuple):  # unpack tuple of indices
            if isinstance(index[0], slice):
                indices = range(*index[0].indices(len(self)))
            elif isinstance(index[0], (np.integer, int)):
                indices = int(index[0])
            else:
                indices = None
            if indices is not None:
                frames = self[indices]

                # index into pixels and channels
                for cnt, idx in enumerate(index[1:]):
                    if isinstance(idx, slice):
                        ix = range(*idx.indices(self.shape[cnt + 1]))
                    elif isinstance(idx, int):
                        ix = range(idx - 1, idx)
                    else:
                        continue

                    if frames.ndim == 4:  # ugly indexing from the back (-1,-2 etc)
                        cnt = cnt + 1
                    frames = np.take(frames, ix, axis=cnt)

        if self.remove_leading_singleton and frames is not None:
            if frames.shape[0] == 1:
                frames = frames[0]
        return frames

    def __repr__(self):
        return f"{self._filename} with {len(self)} frames of size {self.frame_shape} at {self.frame_rate:1.2f} fps"

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.read()
        if ret:
            return frame
        raise StopIteration

    def __enter__(self):
        return self

    def __exit__(self):
        """Release video file."""
        del self

    def close(self):
        """Close video file."""
        self._vc.release()

    def read_thread(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self._vc.read()
                if not ret:
                    self.stopped = True
                self.q.put(frame)

    def read(self, frame_number=None):
        """Read next frame or frame specified by `frame_number`."""
        if not self.stopped and self.threaded:
            sleep(10**-6)  # wait for frame to be read?
            frame = self.q.get(0.1)
            return True, frame

        is_current_frame = frame_number == self.current_frame_pos
        # no need to seek if we are at the right position
        # - greatly speeds up reading sunbsequent frames
        if frame_number is not None and not is_current_frame:
            self._seek(frame_number)
        ret, frame = self._vc.read()
        return ret, frame

    def _reset(self):
        """Re-initialize object."""
        self.__init__(self._filename)

    def _seek(self, frame_number):
        """Go to frame."""
        self._vc.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    @property
    def number_of_frames(self):
        return int(self._vc.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_rate(self):
        return self._vc.get(cv2.CAP_PROP_FPS)

    @property
    def frame_height(self):
        return int(self._vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_width(self):
        return int(self._vc.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_channels(self):
        n_channels = int(self._vc.get(cv2.CAP_PROP_CHANNEL))
        if n_channels == 0:
            # if channel is 0, backend is not supported
            self._reset()
            n_channels = self.read(0)[1].shape[-1]

        return n_channels

    @property
    def fourcc(self):
        return int(self._vc.get(cv2.CAP_PROP_FOURCC))

    @property
    def frame_format(self):
        return int(self._vc.get(cv2.CAP_PROP_FORMAT))

    @property
    def current_frame_pos(self):
        return int(self._vc.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def frame_shape(self):
        return (self.frame_height, self.frame_width, self.frame_channels)

    @property
    def shape(self):
        return (
            self.number_of_frames,
            self.frame_height,
            self.frame_width,
            self.frame_channels,
        )
