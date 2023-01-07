import cv2
import os
from collections import Counter
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(prog = 'Japanese Subtitles & Video to HTML Story Book',
                    description = 'Video + Japanese Subs = HTML Picture Book',
                    epilog = '...')

parser.add_argument('-w', '--width', default=200, type=int, help='Width of screenshots in HTML file.') 
parser.add_argument('-i', '--input', default='', help='Input Directory. Put *.SRT & Video here.') 
parser.add_argument('-o', '--output', default='', help='Output Directory. HTML exports here.') 
parser.add_argument('-d', '--disable_furigana', action='store_true') 

args = parser.parse_args()
WIDTH = args.width

if args.input == '':
    src_dir = 'Input'
else:
    src_dir = args.input
    
if args.output == '':
    out_dir = 'Output'
else:
    out_dir = args.output
    
if not(os.path.exists(out_dir)):
    os.mkdir(out_dir)
    
USE_FURIGANA = False if args.disable_furigana else True
if USE_FURIGANA:
    from furigana.furigana import split_furigana

files = os.listdir(src_dir)

projects = [k for k, v in Counter([os.path.splitext(f)[0] for f in files]).items() if v >= 2]
matches = {p: [f for f in files if p in f] for p in projects}

def file_to_line_list(filename, encoding='utf-8-sig'):
        line_list = []
        with open(filename, 'r', encoding=encoding) as file:
            for line in file:
                line_list += [line.replace('\n', '')]
        return line_list

def chunk_sub_idx_to_list(sub_line_list):
        """
            Pass in a list where each line is a line in the subtitle file
            Example:
            ['1', '00:00:00,000 --> 00:00:04,430', 'おはようございます', '2', ...]

            return a list where each list item is another list where each item is specific to its index
            Example:
            [['1', '00:00:00,000 --> 00:00:04,430', 'おはようございます'], ['2', ...], ...]
        """
        lines_indexed = []
        tmp = []
        for i, line in enumerate(sub_line_list):
            if line == '':
                continue

            tmp += [line]
            if len(tmp) > 3:
                digit, timestamp = tmp[-2:]
                if digit.strip().isdigit() and '-->' in timestamp:
                    lines_indexed += [tmp[:-2]]
                    tmp = tmp[-2:]
        return lines_indexed
    
def srt_time_to_seconds(time_line):
    def timestr_to_sec(time_str):
        h, m, s_str = time_str.split(':')
        s, ms = s_str.split(',')
        return int(h)*60*60 + int(m)*60 + int(s) + int(ms)/1000 

    start_time_str, stop_time_str = time_line.split(' --> ')
    start_time = timestr_to_sec(start_time_str)
    stop_time = timestr_to_sec(stop_time_str)

    return start_time, stop_time

def add_image(image, w=200, h=200):
    basename = os.path.splitext(os.path.basename(image))[0]
    return f'<img src="{image}" alt="{basename}" width="{w}" height="{h}" class="center">'

def add_sub(sub):
    return f'<div>{sub}</div>'    

def create_html(name, content):
    return \
    f"""<!DOCTYPE html>
    <html>
    <style>
    div {{text-align: center;}}
    .center {{
      display: block;
      margin-left: auto;
      margin-right: auto;
    }}
    </style>
    <title>{name}</title>
    <head>
    </head>
    <body>
    <h1>{name}</h1>
    {content}
    </body>
    </html>
    """
def add_furigana(text):

    w_furigana = ''
    for pair in split_furigana(text):
        if len(pair)==2:
            kanji,hira = pair
            w_furigana +=  f"<ruby><rb>{kanji}</rb><rt>{hira}</rt></ruby>"
        else:
            w_furigana += pair[0]
    return w_furigana

def create_html_list(prj_name, vid_file, lines_indexed):

    video_capture = cv2.VideoCapture(vid_file)

    html_list = []

    for r in tqdm(lines_indexed):
        line_idx, time_str = r[:2]
        sub_list = r[2:]
        
        if USE_FURIGANA:
            tmp = []
            for s in sub_list:
                try:
                    tmp += [add_furigana(s)]
                except:
                    tmp += [s]
                
            sub_list = tmp

        start, stop = srt_time_to_seconds(time_str)
        time_ms = int(1000*((stop - start)/2 + start))

        video_capture.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        success, image = video_capture.read()

        if success:
            new_filename = prj_name + '_' + str(time_ms) + '.jpg'
            path = os.path.join(prj_dir, new_filename)
            if not(os.path.exists(path)):
                cv2.imwrite(path, image)

            h, w = image.shape[:-1]
            ratio = h/w

            html_list += [add_image(new_filename, w=WIDTH, h=int(WIDTH*ratio))]
            html_list += [add_sub(s) for s in sub_list]
            
    return html_list

for prj, prj_files in matches.items():
    srt_files = [os.path.join(src_dir, f) for f in prj_files if '.srt' in f]
    vid_files = [os.path.join(src_dir, f) for f in prj_files if '.avi' in f]
    
    if len(srt_files) != 1:
        print(f'{len(srt_files)} srt files found. Skipping {prj}')
        continue
    if len(vid_files) != 1:
        print(f'{len(vid_files)} video files found. Skipping {prj}')
        continue
    
    srt_file = srt_files[0]
    vid_file = vid_files[0]
    
    prj_name = prj.replace(' ','_')
    
    prj_dir = os.path.join(out_dir, prj_name)
    if not(os.path.exists(prj_dir)):
        os.mkdir(prj_dir)
        
    line_list = file_to_line_list(srt_file)
    lines_indexed = chunk_sub_idx_to_list(line_list)

    html_list = create_html_list(prj_name, vid_file, lines_indexed)
    
    html = create_html(prj_name, '\n'.join(html_list))

    save_path = os.path.join(prj_dir, prj_name + '.html')

    f = open(save_path, 'w')
    f.write(html)
    f.close()