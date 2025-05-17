import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import ResultCard from "./components/ResultCard";

function App() {
  const [result, setResult] = useState(null);

  return (
    <div>
      <h1>Real-Time Emotional State Analyzer</h1>
      <UploadForm setResult={setResult} />
      <ResultCard result={result} />
    </div>
  );
}

export default App;
