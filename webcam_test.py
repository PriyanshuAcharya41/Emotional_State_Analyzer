import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        return frame

st.title("🎥 Webcam Test")
webrtc_streamer(key="test", video_transformer_factory=VideoTransformer)
