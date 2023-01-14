import cv2
import os

def video_process(
    src_path, 
    frame_num=1,     
    save_path=None
):
    '''
        src_dir:   directory of videos to be processed
        frame_num: the number of images expected to be extracted from a given video, default `1`
                   interval of image extraction is #frame_count/(frame_num+1)
                   
        save_path: directory where extracted audio and image will be saved
                   default: <current_dir>/<image>;  <current_dir>/<audio>
                   
                   `image`:
                   extracted images are saved as <save_path>/<image>/<origin_video_name>_<frame_index>.jpg
                   note. extracted frames are indexed from `0`
                   `audio`:
                   extracted audio track is saved as <save_path>/<audio>/<origin_video_name>.wav
    '''
    
    if save_path and not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    if save_path is None: save_path = os.getcwd()
    
    _image_save_path = os.path.join(save_path, 'image')
    if not os.path.isdir(_image_save_path):
        os.mkdir(_image_save_path)
    
    _audio_save_path = os.path.join(save_path, 'audio')
    if not os.path.isdir(_audio_save_path):
        os.mkdir(_audio_save_path)
    
    
    _name = os.path.split(src_path)[-1].split('.')[0]
    
    
    # extract image 
    video = cv2.VideoCapture(src_path)
    
    fc = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_interval = int(fc // (frame_num+1))
    
    count = 1; success = True
    while success:
        success, image = video.read()
        if count % frame_interval == 0:
            frame_index = (count-1) // frame_interval
            _save_path = os.path.join(_image_save_path, '{}_{}.jpg'.format(_name, frame_index))
            cv2.imencode('.jpg', image)[1].tofile(_save_path)
            
            if (frame_index == (frame_num-1)): break

        count += 1
        
    # extract audio track with command line tool: `ffmpeg`
    _save_path = os.path.join(_audio_save_path, '{}.wav'.format(_name))
    cmd = f'ffmpeg -loglevel error -i {src_path} {_save_path}'
    
    os.system(cmd)


            
            
    
    
    
