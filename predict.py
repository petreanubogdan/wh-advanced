import os
import shutil
import zipfile

from typing import Optional

from cog import BasePredictor, Input, Path, BaseModel

from transformers import WhisperProcessor, pipeline
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

from typing import Iterable
from typing import Any

import torch

# os.environ['COG_WEIGHTS'] = 'https://storage.googleapis.com/dan-scratch-public/whisper_trained.zip'

class Output(BaseModel):
    segments: Any
    #preview: str
    #srt_file: Path

class Predictor(BasePredictor):

    def setup(self):
        
        #weights_url = 'https://replicate.delivery/pbxt/ufuFFMl7MWzNUqcsHVJ7eYam8rLSaQyeAq3X3IeVGFThhFSIB/training_output.zip'
        weights_url = 'https://replicate.delivery/pbxt/2PUQhupGH0ZYIlX40RaW4tjNVEdcCtVhD7dAX0p9YgJkIlkE/training_output.zip'
        local_path = "/src/weights.zip"
        os.system(f"pget {weights_url} {local_path}")
        out = "/src/weights_dir"
        if os.path.exists(out):
            shutil.rmtree(out)
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            zip_ref.extractall(out)
            #model_name = weights = os.path.join(out, "model")
            #TransformersConverter(model_name, copy_files=[model_name + '/tokenizer_config.json', model_name + '/preprocessor_config.json']).convert(model_name + "/fast_model")
            weights_path = os.path.join(out)

        processor = WhisperProcessor.from_pretrained(weights_path, task="transcribe")
        self.model = pipeline(
            task="automatic-speech-recognition",
            model=weights_path,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=torch.device("cuda:0"),
            chunk_length_s=30,
            generate_kwargs={"num_beams": 5,  
                            "word_timestamps": True, 
                            },
            torch_dtype="float16",
    
        ) 
    
        
    def predict(
        self,
        audio_path: Path = Input(description="audio to transcribe"),
        language: str = Input(
            choices=sorted(LANGUAGES.keys()),
            default=None,
            description="language spoken in the audio, specify None to perform language detection",
        ),

    ) -> Output:
        
        transcription = self.model(str(audio_path), return_timestamps="word", generate_kwargs={"language": language})

        print(transcription )


        #segments = []
        #text = []
#
        #for segment in transcription:
        #    text_segment = dict()
        #    text_segment["end"] = segment.end
        #    text_segment["start"] = segment.start
        #    text_segment["text"] = segment.text
        #    text_segment["words"] = []
        #    
        #    for word in segment.words:
        #        words_segment = dict()
        #        words_segment["start"] = word.start            	
        #        words_segment["end"] = word.end
        #        words_segment["word"] = word.word.strip()
        #        text_segment["words"].append(words_segment)
        #    
        #    segments.append(segment)
        #    text.append(text_segment)
#
        #audio_basename = os.path.basename(str(audio_path)).rsplit(".", 1)[0]
#
        #out_path_srt = f"/tmp/{audio_basename}.{language}.srt"
        #with open(out_path_srt, "w", encoding="utf-8") as srt:
        #    srt.write(generate_srt(segments))
#
        #preview = " ".join([segment.text.strip() for segment in segments[:5]])
        #if len(preview) > 5:
        #    preview += f"... (only the first 5 segments are shown, {len(segments) - 5} more segments in subtitles)"
#
        return Output(
            segments='transcription',
        #    preview=preview,
        #    srt_file=Path(out_path_srt),
        )


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def generate_srt(result):
    srt = ""
    for i, segment in enumerate(result, start=1):
        srt += f"{i}\n"
        srt += f"{format_timestamp(segment.start, always_include_hours=True)} --> {format_timestamp(segment.end, always_include_hours=True)}\n"
        srt += f"{segment.text.strip().replace('-->', '->')}\n"
    return srt
