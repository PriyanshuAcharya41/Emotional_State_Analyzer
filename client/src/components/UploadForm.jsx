import React, { useState, useRef } from "react";

const UploadForm = ({ setResult }) => {
  const [loading, setLoading] = useState(false);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const captureImage = () => {
    const canvas = document.createElement("canvas");
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg"));
  };

  const startRecording = async () => {
    setLoading(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      videoRef.current.srcObject = stream;
      videoRef.current.play();

      // Start recording audio
      const audioStream = new MediaStream(stream.getAudioTracks());
      mediaRecorderRef.current = new MediaRecorder(audioStream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        const imageBlob = await captureImage();

        const formData = new FormData();
        formData.append("audio", audioBlob, "voice.wav");
        formData.append("image", imageBlob, "face.jpg");

        const res = await fetch("http://localhost:5000/analyze", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        setResult(data);
        setLoading(false);
      };

      mediaRecorderRef.current.start();

      // Stop after 5 seconds
      setTimeout(() => {
        mediaRecorderRef.current.stop();
        stream.getTracks().forEach((track) => track.stop());
      }, 5000);
    } catch (err) {
      console.error("Error accessing media devices:", err);
      setLoading(false);
    }
  };

  return (
    <div>
      <video ref={videoRef} style={{ display: "none" }} />
      <button onClick={startRecording} disabled={loading}>
        {loading ? "Analyzing..." : "Capture & Analyze"}
      </button>
    </div>
  );
};

export default UploadForm;