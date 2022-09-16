from __future__ import annotations

import os
from tqdm import tqdm

import sys
sys.path.append('/home/guest/dev_repo/SoccerTrack')


from soccertrack.utils.utils import MovieIterator, make_video
from soccertrack import load_df

def visualize_bbox(bboxdf, movie_iterator, save_dir):
    idx_list = []
    frame_list = []
    for frame_idx ,frame in tqdm(enumerate(movie_iterator)):
        idx_list.append(frame_idx)
        frame_list.append(frame)

    img_list = []
    for idx in tqdm(range(0,len(idx_list)-1,1)):
        img_ = bboxdf.visualize_frame(idx_list[idx], frame_list[idx])
        img_list.append(img_)
    
    save_path = os.path.join(save_dir, 'bbox.mp4')
    make_video(img_list, save_path)


# def main():
#     test_movie_path  = '/home/guest/dev_repo/SoccerTrack/x_ignore/test_data/F_20200220_1_0000_0030.mp4'
#     data_path  = '/home/guest/dev_repo/SoccerTrack/x_ignore/test_data/F_20200220_1_0000_0030.csv'
#     save_dir = '/home/guest/dev_repo/SoccerTrack/x_ignore/test_data/'

#     bboxdf = load_df(data_path)
#     movie_iterator = MovieIterator(test_movie_path)
#     visualize_bbox(bboxdf, movie_iterator, save_dir)

# if __name__ == '__main__':
#     main()