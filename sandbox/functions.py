def cut_video_file(video_file_name: str, start_time: int, end_time: int) -> None:
    """
    Cut a video from start_time to end_time.

    Parameters
    ----------
    video_file_name : str
        Path to the video file to cut.
    start_time : int
        Start time of the video.
    end_time : int
        End time of the video.

    Returns
    -------
    video_file : str #
    """
    pass

def cut_gps_file(gps_file_name: str, start_time: int, end_time: int) -> None:
    """
    Cut a gps file from start_time to end_time.

    Parameters
    ----------
    gps_file_name : str
        Path to the gps file to cut.
    start_time : int
        Start time of the gps file.
    end_time : int
        End time of the gps file.

    Returns
    -------
    cut_gps_file : str #Should I set return to None or the trimmed DataFrame?
    """
    pass

def visualization_gps(gps_file_name: str) -> None:
    """
    Visualize the gps file.

    Parameters
    ----------
    gps_file_name :
        Path to the gps file to visualize.

    Returns
    -------
    None #How should I set the return for the visualization function?
    """
    pass

def visualization_annotations(annotations_file_name: str) -> None:
    """
    Visualize the annotations file.

    Parameters
    ----------
    annotations_file_name : str
        Path to the annotations file to visualize.

    Returns
    -------
    None
    """
    pass

def upload2s3(integration_key: str, bucket_name: str, file_name: str) -> None: #Not sure how to integrate with S3, but probably need to fill in some kind of key or bucket name
    """
    Upload a file to S3.

    Parameters
    ----------
    integration_key : str
        Integration key for the S3 bucket.
    bucket_name : str
        Name of the S3 bucket.
    file_name : str
        Name of the file to upload.

    Returns
    -------
    None
    """
    pass

def download_from_s3(integration_key: str, bucket_name: str, file_name: str) -> None: #Not sure how to integrate with S3, but probably need to fill in some kind of key or bucket name
    """
    Download a file from S3.

    Parameters
    ----------
    integration_key : str
        Integration key for the S3 bucket.
    bucket_name : str
        Name of the S3 bucket.
    file_name : str
        Name of the file to download.

    Returns
    -------
    None
    """
    pass

def upload_annotation2labelbox(annotations_file_name: str, labelbox_api_key: str, labelbox_project_id: str) -> None: #Probably some kind of Labelbox access key is needed.
    """
    Upload annotations to Labelbox.

    Parameters
    ----------
    annotations_file_name : str
        Path to the annotations file to upload.
    labelbox_api_key : str
        Labelbox API key.
    labelbox_project_id : str
        Labelbox project ID.

    Returns
    -------
    None
    """
    pass

def upload_video2labelbox(video_file_name: str, labelbox_api_key: str, labelbox_project_id: str) -> None: #Probably some kind of Labelbox access key is needed.
    """
    Upload video to Labelbox.

    Parameters
    ----------
    video_file_name : str
        Path to the video file to upload.
    labelbox_api_key : str
        Labelbox API key.
    labelbox_project_id : str
        Labelbox project ID.

    Returns
    -------
    None
    """
    pass


