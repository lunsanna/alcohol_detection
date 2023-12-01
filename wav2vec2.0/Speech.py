import pandas as pd 
import torchaudio

# for typing 
from typing import Dict 
from torch import Tensor 

class Speech(object):
    def __init__(self, df: pd.DataFrame, 
                 target_sr: int = 16000, 
                 target_dur: int = 10, 
                 path_col_name: str = "resampled_path",
                 label_col_name: str = "alc_mapped",
                 path_prefix: str = "",
                 text_to_label: Dict[str, int] = {"Intoxicated": 1, "Sober": -1},
                ):
        """
        df: df with a column named file path containing the paths of the speech files
        target_sr (int): target sampling rate 
        target_dur (int): max speech duration in seconds
        path_col_name (str): the name of the df column that contains the speech paths
        path_prefix (str): prefix added to all the paths, useful for relative paths
        """
        
        self.df = df
        self.target_sr = target_sr
        self.target_dur = target_dur
        self.path_col_name = path_col_name
        self.label_col_name = label_col_name
        self.path_prefix = path_prefix
        self.text_to_label = text_to_label
        self.label_to_text = {value:key for key, value in self.text_to_label.items()}
        
    def _get_speech(self, path:str) -> Tensor:
        "Given speech path, return speech tensor"
        # load speech from path 
        speech, sr = torchaudio.load(path)

        # wav2vec2 expects 1D waveform
        try:
            speech = speech.squeeze()
        except RuntimeError as e:
            print(e)
            print(f"If the paths are relative, be sure to define path_prefix when creating the object.")

        # ensure that the sr and duration are as expected
        assert sr == self.target_sr, f"Expect sr={self.target_sr}, got {sr}"
        assert len(speech) <= sr * 10, f"Expect duration <= {self.target_dur}, got {len(speech)/sr}"
        
        return speech
    
    def get_label(self, text_label:str) -> int:
        """Given label text, return number labels"""
        return self.text_to_label[text_label]
    
    def get_label_text(self, label:int) -> str:
        "Given the label, return lable text"
        return self.label_to_text[label]
    
    def __iter__(self):
        for i, row in self.df.iterrows():
            
            # get speech path, then get speech 
            path =  self.path_prefix + row[self.path_col_name]
            speech = self._get_speech(path)
            
            # get label
            label = self.get_label(row[self.label_col_name])
            
            yield {"index": i, 
                   "path": path,
                   "speech": speech, 
                   "label": label}
