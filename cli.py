import argparse
from substory import SubStory

def main(args):    
    ss = SubStory(src_dir=args.source_dir, out_dir=args.output_dir, add_furigana=args.add_furigana, export_width=args.width, track_number=args.track, verbose=args.verbose, language=args.language)
    ss.process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line interface for converting video to study book.')
    parser.add_argument('--source-dir', type=str, help='Source directory', default='Input')
    parser.add_argument('--output-dir', type=str, help='Output directory', default='Output')
    parser.add_argument('--add-furigana', action='store_true', help='Whether to add Furigana')
    parser.add_argument('--width', type=int, help='Image width', default=200)
    parser.add_argument('--track', type=int, help="Video's Audio Track Number", default=0)
    parser.add_argument('--verbose', action='store_true', help='Print on or off.')
    parser.add_argument('--language', type=str, help='Language abbreviation (e.g., "ja" for Japanese)', default=None)
    args = parser.parse_args()

    main(args)
