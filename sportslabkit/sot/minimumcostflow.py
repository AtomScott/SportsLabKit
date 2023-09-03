from sportslabkit.sot.base import SingleObjectTracker


class MinimumCostFlowTracker(SingleObjectTracker):
    def __init__(
        self,
        target,
        initial_frame,
        detection_model=None,
        image_model=None,
        motion_model=None,
        matching_fn=None,
        window_size=10,
    ):
        super().__init__(
            target,
            window_size=window_size,
            pre_init_args={
                "initial_frame": initial_frame,
                "detection_model": detection_model,
                "image_model": image_model,
                "motion_model": motion_model,
                "matching_fn": matching_fn,
            },
        )

    def pre_initialize(self, initial_frame, detection_model, image_model, motion_model, matching_fn):
        self.detections = []
        self.detection_model = detection_model
        self.image_model = image_model
        self.matching_fn = matching_fn
        self.motion_model = motion_model

        if self.image_model is not None:
            self.target["feature"] = self.image_model.embed_detections([self.target], initial_frame)[0]

    def update(self, sequence):
        # Initialize an empty graph
        list_of_detections = []
        for frame in sequence:
            # Extract the new detections from the current frame
            detections = self.detection_model(frame)

            # extract features from the detections
            detections = detections[0].to_list()

            if len(detections) > 0 and self.image_model is not None:
                embeds = self.image_model.embed_detections(detections, frame)
                for i, det in enumerate(detections):
                    det.feature = embeds[i]

            list_of_detections.append(detections)

        # Must be batch matching function
        path = self.matching_fn([self.tracklet], list_of_detections)
        new_states = []
        for frame_idx, det_idx in enumerate(path):
            if det_idx >= 0:  # if there is a match
                new_state = {
                    "box": list_of_detections[frame_idx][det_idx].box,
                    "score": list_of_detections[frame_idx][det_idx].score,
                    "feature": list_of_detections[frame_idx][det_idx].feature,
                }
            else:  # if there is no match
                new_state = {
                    "box": self.tracklet.box,
                    "score": 0.5,
                    "feature": self.tracklet.feature,
                }
                # print(f'no match found @ frame_idx={frame_idx}')

            if self.motion_model is not None:
                self.motion_model.update(new_state)
            new_states.append(new_state)

        return new_states

    @property
    def required_keys(self):
        return ["box"]
