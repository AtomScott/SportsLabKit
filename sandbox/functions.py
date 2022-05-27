def cut_video_file(video_file_name: str, start_time: int, end_time: int, save_dir: str) -> None:
    """Cut a video from start_time to end_time.
    Args:
        video_file_name (str) : Path to the video file to cut.
        start_time (int) : Start time of the video.
        end_time (int) : End time of the video.
        save_dir (str) : Path to save the video.
    """
    pass

def cut_gps_file(gps_file_name: str, start_time: int, end_time: int, save_dir: str) -> None:
    """Cut a gps file from start_time to end_time.

    Args:
        gps_file_name (str) : Path to the gps file to cut.
        start_time (int) : Start time of the gps file.
        end_time (int) : End time of the gps file.
        save_dir (str) : Path to save the gps file.
    """
    pass

def visualization_gps(gps_file_name: str, save_path: str) -> None:
    """Visualize the gps file.

    Args:
        gps_file_name (str) : Path to the gps file to visualize.
        save_path (str) : Path to save the gps file.
    """
    pass

def visualization_annotations(annotations_file_name: str, save_path: str) -> None:
    """Visualize the annotations file.

    Args:
        annotations_file_name (str) : Path to the annotations file to visualize.
        save_path (str) : Path to save the annotations file.
    """
    pass

def upload2s3(integration_key: str, bucket_name: str, file_name: str) -> bool: #Not sure how to integrate with S3, but probably need to fill in some kind of key or bucket name
    """Upload a file to S3.

    Args:
        integration_key (str) : Integration key for the S3 bucket.
        bucket_name (str) : Name of the S3 bucket.
        file_name (str) : Name of the file to upload.

    Returns:
        bool (bool) : True if the upload was successful, False otherwise.

    """
    pass

def download_from_s3(integration_key: str, bucket_name: str, download_dir: str, save_path: str) -> bool:
    """Download a file from S3.

    Args:
        integration_key (str) : Integration key for the S3 bucket.
        bucket_name (str) : Name of the S3 bucket.
        download_dir (str) :Path of the directory to download from S3.
        save_path (str) : Path to save the file.

    Returns:
        bool (bool) : True if the download was successful, False otherwise.

    Note:
        Save S3 directory as a zip file
    """
    pass

def upload_annotation2labelbox(annotations_file_name: str, labelbox_api_key: str, labelbox_project_id: str) -> bool: #Probably some kind of Labelbox access key is needed.
    """Upload annotations to Labelbox.

    Args:
        annotations_file_name (str) : Path to the annotations file to upload.
        labelbox_api_key (str) : Labelbox API key.
        labelbox_project_id (str) : Labelbox project ID.

    Returns:
        bool (bool) : True if the download was successful, False otherwise.
    """
    pass

def upload_video2labelbox(video_file_name: str, labelbox_api_key: str, labelbox_project_id: str) -> bool: #Probably some kind of Labelbox access key is needed.
    """Upload video to Labelbox.

    Args:
        video_file_name (str) : Path to the video file to upload.
        labelbox_api_key (str) : Labelbox API key.
        labelbox_project_id (str) : Labelbox project ID.
    Returns:
        bool (bool) : True if the upload was successful, False otherwise.
    """
    pass

def create_annotation_df_from_s3(integration_key: str, bucket_name: str, root_dir: str ,dir_name_list: list[str], save_path: str) -> None:
    """Create a dataframe(csv file) from the annotations file.

    Args:
        integration_key (str) : Integration key for the S3 bucket.
        bucket_name (str) : Name of the S3 bucket.
        root_dir (str) : Root directory of the S3 bucket.
        dir_name_list (list[str]) :List of data types to be stored in the df columns. The element of each list contains the directory name.
        save_path (str) : Path to save the csv_file.
    """

    pass