import pandas as pd
import torch
import pickle, time
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from Speech import Speech

def extract_features(speech: Speech, 
                     processor: Wav2Vec2Processor,
                     model: Wav2Vec2Model,
                     device:torch.device, 
                     layer_out_features_path:str, 
                     layer_12_features_path:str, 
                     n_samples:int):
    
    speech_features_layer_out_mean = {}
    speech_features_layer_12_mean = {}
    
    speech_features_layer_out_max = {}
    speech_features_layer_12_max = {}
    
    i = 0
    start = time.time()
    
    for s in speech:
        # preprocess
        inputs = processor(s["speech"].to(device), return_tensors="pt", sampling_rate=16000,  padding="longest")
        
        # extract features
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # hidden features (seq_len, 1024)
        hidden_layer_out = outputs.hidden_states[-1].cpu().numpy()[0] 
        hidden_layer_12 = outputs.hidden_states[-12].cpu().numpy()[0]
        
        # mean
        speech_features_layer_out_mean[s["index"]] = hidden_layer_out.mean(axis=0) # (1024,)
        speech_features_layer_12_mean[s["index"]] = hidden_layer_12.mean(axis=0) # (1024,)
        
        # max 
        speech_features_layer_out_max[s["index"]] = hidden_layer_out.max(axis=0) # (1024,)
        speech_features_layer_12_max[s["index"]] = hidden_layer_12.max(axis=0) # (1024,)
        
        if i % 500 == 0:
            s = time.time()-start
            print(f"{i}/{n_samples} done. Time elapsed: {int(s//3600)}:{int((s//60)%60):02d}:{int(s%60):02d}")
        i += 1
    # save mean 
    with open(layer_out_features_path + "_mean.pkl", "wb") as file:
        pickle.dump(speech_features_layer_out_mean, file)
    with open(layer_12_features_path + "_mean.pkl", "wb") as file:
        pickle.dump(speech_features_layer_12_mean, file)
    # save max
    with open(layer_out_features_path + "_max.pkl", "wb") as file:
        pickle.dump(speech_features_layer_out_max, file)
    with open(layer_12_features_path + "_max.pkl", "wb") as file:
        pickle.dump(speech_features_layer_12_max, file)

if __name__=="__main__":
    # Config
    pre_trained_path = "facebook/wav2vec2-large-xlsr-53-german"
    target_sr = 16000
    path_col_name = "resampled_path"
    df_train_path = "../train.csv"
    df_test_path = "../test.csv"
    
    layer_out_features_path = "speech_features_layer_out_mean"
    layer_12_features_path = "speech_features_layer_12_mean"
    
    layer_out_features_path_test = "test_speech_features_layer_out"
    layer_12_features_path_test = "test_speech_featurs_layer_12"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RUNNING ON {device}")
    
    # Load dfs and print stats
    df = pd.read_csv(df_train_path)
    df = df[df.alc_mapped != "Control"]
    print("---Training set---")
    print("Intoxicated: ", len(df[df.alc_mapped=="Intoxicated"]))
    print("Sober: ", len(df[df.alc_mapped=="Sober"]))
    print("Control: ", len(df[df.alc_mapped=="Control"]))
    
    df_test = pd.read_csv(df_test_path)
    df_test = df_test[df_test.alc_mapped != "Control"]
    print("---Test set---")
    print("Intoxicated: ", len(df_test[df_test.alc_mapped=="Intoxicated"]))
    print("Sober: ", len(df_test[df_test.alc_mapped=="Sober"]))
    print("Control: ", len(df_test[df_test.alc_mapped=="Control"]))
    
    # Get speech for training set 
    speech = Speech(df, path_prefix="../")
    
    # Get speech for test set 
    speech_test = Speech(df_test, path_prefix="../",)

    # Load wav2vec2 processor and model from pre-trained 
    processor = Wav2Vec2Processor.from_pretrained(pre_trained_path)
    model = Wav2Vec2Model.from_pretrained(pre_trained_path)
    
    # Extract hidden features for training set 
    extract_features(speech, processor, model, device, layer_out_features_path, 
                     layer_12_features_path, len(df))
    
    # Extract hidden features for test set
    extract_features(speech_test, processor, model,device, layer_out_features_path_test, 
                     layer_12_features_path_test, len(df_test))
