import React from "react";

const ResultCard = ({ result }) => {
  if (!result) return null;

  return (
    <div>
      <h2>Emotion Analysis Result</h2>
      <p>Face: {result.face.label} ({(result.face.confidence * 100).toFixed(1)}%)</p>
      <p>Voice: {result.voice.label} ({(result.voice.confidence * 100).toFixed(1)}%)</p>
      <p>Text: {result.text.sentiment}</p>
      <p>Transcript: "{result.text.transcript}"</p>
    </div>
  );
};

export default ResultCard;
