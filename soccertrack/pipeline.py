class PipelineObject:
    def __init__(self):
        pass
    
    def predict(*args, **kwargs):
        pass

    def transform(*args, **kwargs):
        pass
    
    def active_tracks(*args, **kwargs):
        pass
    
    def step(*args, **kwargs):
        pass

def get_fe_pipe(*args, **kwargs):
    return PipelineObject()


def get_matching_fn(*args, **kwargs):
    return PipelineObject()

def get_tracking_model(*args, **kwargs):
    return PipelineObject()

    
def get_detection_model(*args, **kwargs):
    return PipelineObject()