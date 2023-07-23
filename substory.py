import cv2
import os
from collections import Counter
from tqdm import tqdm
import whisper
import torch
import moviepy.editor as mp
import subprocess
from datetime import timedelta
from yattag import Doc, indent
import time
import pykakasi
import openai
import json
import tiktoken
import math
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


class SubStory:
    vid_exts=['.mp4','.avi','.ogv','.mkv','.webm']
    aud_exts=['.mp3']
    sub_exts=['.srt']
    
    def __init__(self, src_dir='Input', out_dir='Output', add_furigana=True, export_width=200, track_number=0, verbose=True, language=None, 
                 use_gpt=False, gpt_model='gpt-3.5-turbo-0613'):
        self.src_dir = src_dir
        self.out_dir = out_dir
        self.add_furigana = add_furigana
        self.width = export_width
        self.track_number = track_number
        self.language = language
        self.use_gpt = use_gpt
        self.gpt_model = gpt_model
        self.files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
        self.proj_files = [os.path.splitext(f)[0] for f in self.files if os.path.splitext(f)[-1] in self.vid_exts]
        self.proj_files += [os.path.splitext(f)[0] for f in self.files if os.path.splitext(f)[-1] in self.aud_exts and os.path.splitext(f)[0] not in self.proj_files]
        self.device = whisper.torch.device('cuda' if whisper.torch.cuda.is_available() else 'cpu')            

    def __file_to_line_list(self, filename, encoding='utf-8-sig'):
        line_list = []
        with open(filename, 'r', encoding=encoding) as file:
            for line in file:
                line_list += [line.replace('\n', '')]
        return line_list

    def __chunk_sub_idx_to_list(self, sub_line_list):
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

    def __srt_time_to_seconds(self, time_line):
        def timestr_to_sec(time_str):
            h, m, s_str = time_str.split(':')
            s, ms = s_str.split(',')
            return int(h)*60*60 + int(m)*60 + int(s) + int(ms)/1000 

        start_time_str, stop_time_str = time_line.split(' --> ')
        start_time = timestr_to_sec(start_time_str)
        stop_time = timestr_to_sec(stop_time_str)

        return start_time, stop_time

    def _add_image(self, doc, image, idx, w=200, h=200):
        basename = os.path.splitext(os.path.basename(image))[0]
        with doc.tag('img', id=f'image_{idx}', src=image, alt=basename, width=w, height=h, klass="center"):
            pass # No content within this tag

    def _add_audio_clip(self, doc, audio_file):
        with doc.tag('audio', controls=True, klass="center"):
            doc.stag('source', src=os.path.basename(audio_file), type="audio/mpeg")
            doc.text('Your browser does not support the audio element.')

    def _add_sub(self, doc, sub):
        with doc.tag('div'):
            doc.asis(sub) 

    def _add_furigana(self, text):
        w_furigana = ''
        
        kks = pykakasi.kakasi()
        result = kks.convert(text)       
        for item in result:
            hira, orig = item['hira'].strip(), item['orig'].strip()
            
            if hira != orig:
                w_furigana +=  f"<ruby><rb>{orig}</rb><rt>{hira}</rt></ruby>"
            else:
                w_furigana += orig
        
        return w_furigana

    def _build_html_doc(self, prj_dir, vid_file, aud_file, sub_file, base_filename, line_dict, audio_mode='normal', pad=0.5, line_sep=True):
        prj_name = os.path.basename(prj_dir)
        if vid_file:
            video_capture = cv2.VideoCapture(vid_file)
            if audio_mode == 'normal':
                mp_video = mp.VideoFileClip(vid_file)
        else:
            if aud_file:
                audio_mode = 'only'
                mp_audio = mp.AudioFileClip(aud_file)

        doc, tag, text = Doc().tagtext()
        with tag('html'):
            doc.asis('<style>')
            doc.text('div {text-align: center;} .center {display: block; margin-left: auto; margin-right: auto;}')
            doc.text('''
                body { background-color: #D8DFEE; }
                h1, h2, h3 { color: #ABA8A9; }
                .highlight { color: #CBF83E; }
                div {text-align: center;} 
                .center {display: block; margin-left: auto; margin-right: auto;}
            ''')
            doc.asis('</style>')
            
            with tag('head'):
                with tag('title'):
                    text(prj_name)
                    
                # Image Size Control
                with tag('script'):
                    doc.asis("""
                    function updateImageSize() {
                        var slider = document.getElementById("slider");
                        var images = document.getElementsByTagName("img");
                        for (var i = 0; i < images.length; i++) {
                            images[i].style.width = slider.value + "px";
                            images[i].style.height = "auto";
                        }
                    }
                    """)
                
            with tag('body'):
                with doc.tag('div'):
                    with tag('label', ('for', 'slider')):
                        text('Adjust Image Size')
                    with doc.tag('input', ('type', 'range'), ('min', '50'), ('max', '500'), ('value', '200'), 
                                ('id', 'slider'), ('oninput', 'updateImageSize()')):
                        pass

                for idx, line_idx in enumerate(line_dict):
                    time_str, *sub_list = line_dict[line_idx]
                    #line_idx, time_str = r[:2]
                    #sub_list = r[2:]
                    if self.add_furigana:
                        tmp = []
                        for s in sub_list:
                            try:
                                tmp += [self._add_furigana(s)]
                            except:
                                tmp += [s]
                        sub_list = tmp

                    start, stop = self.__srt_time_to_seconds(time_str)
                    time_ms = int(1000*((stop - start)/2 + start))
                    if audio_mode != 'only':
                        video_capture.set(cv2.CAP_PROP_POS_MSEC, time_ms)
                        success, image = video_capture.read()
                        if success:
                            new_filename = prj_name + '_' + str(time_ms) + '.jpg'
                            path = os.path.join(prj_dir, new_filename)
                            if not(os.path.exists(path)):
                                h, w = image.shape[:-1]
                                ratio = h/w
                                new_height = int(self.width*ratio)
                                resized_image = cv2.resize(image, (self.width, new_height))
                                cv2.imwrite(path, resized_image)
      
                                self._add_image(doc, new_filename, idx, w=self.width, h=int(self.width*ratio))
                    for s in sub_list:
                        self._add_sub(doc, s)
                    if audio_mode != 'off':                    
                        new_filename = prj_name + '_' + str(time_ms) + '.mp3'
                        path = os.path.join(prj_dir, new_filename)
                        if audio_mode == 'only':
                            mp_audio.subclip(max(0, start - pad), stop+pad).write_audiofile(path, verbose=False, logger=None)
                        else:
                            mp_video.subclip(max(0, start - pad), stop+pad).audio.write_audiofile(path, verbose=False, logger=None)
                        self._add_audio_clip(doc, path)
                    if line_sep:
                        doc.stag('hr')  # Add a horizontal line
        return indent(doc.getvalue())

    def extract_audio(self, vid_file, track_number=0): 
        def extract_ffmpeg(input_file, output_file, audio_track):
            command = f'ffmpeg -i "{input_file}" -map 0:a:{audio_track} "{output_file}"'
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            
        base_filename = os.path.splitext(vid_file)[0] # Remove ext

        savename = base_filename + '.mp3'

        # Default to using MoviePy
        if track_number == 0:
            v = mp.VideoFileClip(vid_file)
            v.audio.write_audiofile(savename)
            
        # Extract different track, need ffmpeg
        else:
            try:
                extract_ffmpeg(vid_file, savename, track_number)
            except Exception as e:
                print(f'FFMPEG Error: {e}')
                
        return savename

    def transcribe_audio(self, transcription, base_filename):
        savepath = base_filename + '.srt'
        segments = transcription['segments']

        for segment in segments:
            startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
            endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
            text = segment['text']
            if len(text) == 0:
                continue
            
            segmentId = segment['id']+1
            segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"

            with open(savepath, 'a', encoding='utf-8') as srtFile:
                srtFile.write(segment)
                
        return savepath
    
    def __flatten_dict(self, d, flat=''):
            for k, v in d.items():
                if type(v) == dict:
                    flat += self.__flatten_dict(v, flat=flat)
                elif type(v) == list:
                    flat += ''.join([l for l in v])
                else:
                    flat += str(v)
                    
            return flat
    
    def __split_dict_on_token_len(self, line_dict, encoding=None, encoder='cl100k_base', max_tkn=4096):
            if encoding is None:
                encoding = tiktoken.get_encoding(encoder)

            split_list = []
            cur_dict = OrderedDict()
            cur_tkn_count = 0

            for index, text in sorted(line_dict.items(), key=lambda item: int(item[0])):  # Sorting by integer keys
                enc_list = encoding.encode(text)
                
                if cur_tkn_count + len(enc_list) <= max_tkn:
                    cur_dict[index] = text
                    cur_tkn_count += len(enc_list)
                else:
                    # Append current dictionary to split list and start a new one
                    split_list.append(dict(cur_dict))
                    cur_dict = OrderedDict({index: text})
                    cur_tkn_count = len(enc_list)

            # Don't forget to append the last dictionary
            if cur_dict:
                split_list.append(cur_dict)
            
            return split_list
    
    def clean_with_gpt(self, orig_line_dict, prj, encoder='cl100k_base'):

        def get_token_len(text):
            encoding = tiktoken.get_encoding(encoder)
            return len(encoding.encode(text))

        line_dict = {k: ''.join(v[1:]) for k, v in orig_line_dict.items()}

        openai.api_key = os.getenv('OPENAI_API_KEY')

        system_message = (
        'You are a multilingual audio transcription support agent. You will take in an audio transcription that likely originated'
        ' as audio from a movie, tv show, podcast, or similar type of multimedia. Correct any transcription'
        ' mistakes. You will have a much larger context to base the transcriptions on and can more accurately transcribe'
        f' each word based on this context. This transcript is from {os.path.basename(prj)}.'
        )

        # Used to force returned data to adhere to a dictionary format
        functions = [
            {
                "name": "update_corrected_transcript",
                "description": "Update transcript with corrected transcriptions from input dictionary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "transcript_dict": {
                            "type": "object",
                            "description": "Dictionary of index/transcript pairs",
                            "properties": {
                                "index": {
                                    "type": "integer",
                                    "description": "Line Index"
                                },
                                "line": {
                                    "type": "string",
                                    "description": "Single line of the transcript. Can contain multiple sentences."
                                }
                            },
                            "required": ["index", "line"]
                        }
                    },
                    "required": ["transcript_dict"]
                }
            }
        ]
        
        if openai.api_key != None:

            print('Cleaning transcriptions with GPT API...')
        
            # Estimate tokens of function call
            flat = self.__flatten_dict(functions[0]) 
            encoding = tiktoken.get_encoding(encoder)
            prefix_len = len(encoding.encode(system_message)) + get_token_len(flat)

            tkn_cnt_dict = {'gpt-3.5-turbo-0613': 4096, 'gpt-4-0613': 8192} # Must support function calling

            max_tokens = int(tkn_cnt_dict[self.gpt_model]/4) - prefix_len
            split_list = self.__split_dict_on_token_len(line_dict, max_tkn = max_tokens, encoder=encoder)

            for split_dict in split_list:
                cmp_context = str(split_dict)

                tlen = get_token_len(cmp_context)
                print(f'GPT Token Lengths: Total={tlen + prefix_len}, Prefix={prefix_len}, Message={tlen}')

                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": cmp_context}
                ]

                response = openai.ChatCompletion.create(
                    model=self.gpt_model, 
                    messages=messages,
                    functions=functions,
                    function_call={'name': 'update_corrected_transcript'}, # Always use function
                )

                try:
                    reply_content = response.choices[0].message

                    funcs = reply_content.to_dict()['function_call']['arguments']
                    funcs = json.loads(funcs)

                    gpt_dict = funcs['transcript_dict']

                    for k, v in gpt_dict.items():
                        if k in orig_line_dict:
                            line = orig_line_dict[k]
                            line[1:] = v if type(v) == list else [v]
                            orig_line_dict[k] = line

                except Exception as e:
                    print(f'{e} Failed on gpt call. Skipping...')
                    continue
        else:
            print('No OpenAI API KEY found. Please add to environmental variables.')
                
        return orig_line_dict          


    def _get_prj_media(self, base_filename):
        has_ext = lambda exts: [f for f in self.files if f.startswith(base_filename) and os.path.splitext(f)[-1].lower() in exts]
        
        vid_files = has_ext(self.vid_exts)
        aud_files = has_ext(self.aud_exts)
        sub_files = has_ext(self.sub_exts)
        
        assert len(vid_files) <= 1, "Multiple video extensions found w/ same base name."
        assert len(aud_files) <= 1, "Multiple audio extensions found w/ same base name."
        
        # 1 Video Found    
        vid_file = vid_files[0] if len(vid_files) == 1 else None
        aud_file = aud_files[0] if len(aud_files) == 1 else None
        sub_file = sub_files[0] if len(sub_files) == 1 else None
        
        return vid_file, aud_file, sub_file
        
    def _get_language(self, model, audio):
        sample = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(sample).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        
        best_guess = max(probs, key=probs.get)
        
        return best_guess
        
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def process(self):
        start_time = time.time()
        for prj in self.proj_files:    
            vid_file, aud_file, sub_file = self._get_prj_media(prj)
            

            print(f'Video File: {vid_file}')
            print(f'Audio File: {aud_file}')
            print(f'Subtitle File: {sub_file}')
            
            # If audio doesnt exist extract it from video
            if aud_file == None:
                if vid_file:
                    print(f'Extracting mp3 from {prj}...')
                    
                    aud_file = self.extract_audio(vid_file, track_number=self.track_number)
                    print(f'Audio File Extracted: {aud_file}')
                    print('Finished extracting audio.')
            
            # If subtitle file doesn't exists create it
            if sub_file == None:
                if os.path.exists(aud_file):
                    print(f'Creating transcript using OpenAIs Whisper for {aud_file}.')
                    
                    model = whisper.load_model("base", device=self.device)
                    audio = whisper.load_audio(aud_file)
                    lang = self.language if self.language else self._get_language(model, audio)
                    if self.add_furigana and lang not in ['ja', 'jp']:
                        self.add_furigana = False
                        print('Set add furigana False')
                        
                    print(f'Add Furigana: {self.add_furigana}')
                        
                    transcription = model.transcribe(audio, language=lang)
                    
                    print('Exporting subtitle.')
                    sub_file = self.transcribe_audio(transcription, prj)
                    print(f'Sub File Extracted: {sub_file}')
                else:
                    print(f'Error. Audio file {aud_file} not found.')
                
            # Check if subtitle exists
            if os.path.exists(sub_file):
                
                print('Generating Audio Visual HTML Page')
                
                prj_name = os.path.basename(prj).replace(' ','_')
                prj_dir = os.path.join(self.out_dir, prj_name)
                if not(os.path.exists(prj_dir)):
                    os.mkdir(prj_dir)

                line_list = self.__file_to_line_list(sub_file)
                lines_indexed = self.__chunk_sub_idx_to_list(line_list)
                line_dict = {l[0]:l[1:] for l in lines_indexed}

                if self.use_gpt:
                    line_dict = self.clean_with_gpt(line_dict, prj)

                html = self._build_html_doc(prj_dir, vid_file, aud_file, sub_file, prj, line_dict)

                save_path = os.path.join(prj_dir, prj_name + '.html')
                
                with open(save_path, 'w', encoding='utf-8') as html_file:
                    html_file.write(html)

                print(f'Finished processing {prj}')

            else:
                print(f'Error. Subtitle file {sub_file} not found.')

        minutes = round((time.time() - start_time)/60, 2)
        print(f'Total processing time: {minutes} minutes.')

