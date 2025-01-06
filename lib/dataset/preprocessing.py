import numpy as np
import mne
class PreprocessingPipeline():
    def __init__(self, config):
        self.config = config
    def process(self, mne_data):
        raise NotImplemented
    
    
class PreprocessingController():
    @classmethod
    def preprocessing(cls, mne_raw, preprocessing_name, kwargs):
        if "asr" == preprocessing_name:
            cls.asr(mne_raw, kwargs)
        elif "drop_channels" == preprocessing_name:
            cls.drop_channels(mne_raw, kwargs)
        elif "average_reference" == preprocessing_name:
            cls.average_reference(mne_raw, kwargs)
        elif "z_nor" == preprocessing_name:
            cls.z_nor(mne_raw, kwargs)
        elif "min_max_nor" == preprocessing_name:
            cls.min_max_nor(mne_raw, kwargs)    
        elif "prep" == preprocessing_name:
            cls.prep(mne_raw, kwargs)
            
    @classmethod
    def prep(cls, mne_raw, kwargs):
        import pyprep
        montage = mne.channels.make_standard_montage(kwargs.get("montage", "standard_1020"))
        pipeline = pyprep.PrepPipeline(mne_raw, kwargs.get("prep_params", {}), montage,
                                       ransac=kwargs.get("runsac", True))
        pipeline.fit(kwargs.get("reference_args", {
            "correlation_secs": 1.0, 
            "correlation_threshold": 0.4, 
            "frac_bad": 0.01
        }))
        
    @classmethod
    def asr(cls, mne_raw, kwargs):
        import asrpy
        asr = asrpy.ASR(sfreq=mne_raw.info["sfreq"], cutoff=kwargs.get("cutoff", 30))
        mne_raw.load_data()
        asr.fit(mne_raw)
        mne_raw = asr.transform(mne_raw)
        return mne_raw
        
    @classmethod
    def drop_channels(cls, mne_raw, kwargs):
        on_missing = kwargs.get("on_missing", "warn")
        channels = kwargs.get("channels", [])
        mne_raw.drop_channels(**kwargs)
        return mne_raw
    
    @classmethod
    def average_reference(cls, mne_raw, kwargs):
        mne_raw.set_eeg_reference("average")
        return mne_raw
    
    @classmethod
    def z_nor(cls, mne_raw, kwargs):
        data = mne_raw.get_data()
        mean_data = np.mean(data, axis=1, keepdims=True)
        std_data = np.std(data, axis=1, keepdims=True)
        zscore_data = (data - mean_data) / std_data
        mne_raw = mne.io.RawArray(zscore_data, mne_raw.info)
        return mne_raw

    @classmethod
    def min_max_nor(cls, mne_raw, kwargs):
        data = mne_raw.get_data()
        min_data = np.min(data, axis=1, keepdims=True)
        max_data = np.max(data, axis=1, keepdims=True)
        print(min_data.shape)
        minmax_data = (data - min_data) / (max_data - min_data)
        mne_raw = mne.io.RawArray(minmax_data, mne_raw.info)
        return mne_raw

