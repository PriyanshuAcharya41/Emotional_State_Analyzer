import path from 'path';
import { execFile } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const analyzeEmotion = (req, res) => {
  const imageFile = req.files?.image?.[0]?.path;
  const audioFile = req.files?.audio?.[0]?.path;

  if (!imageFile || !audioFile) {
    return res.status(400).json({ error: "Missing image or audio file." });
  }

  const fusionScript = path.join(__dirname, '..', '..', 'ESA_PROJECT', 'fusion', 'fusion_predict.py');

  execFile('python', [fusionScript, audioFile, imageFile], (err, stdout, stderr) => {
    if (err) {
      console.error("❌ Python error:", stderr);
      return res.status(500).json({ error: "Fusion script failed." });
    }

    try {
      const result = JSON.parse(stdout);
      res.json(result);
    } catch (parseErr) {
      console.error("❌ Failed to parse output:", stdout);
      res.status(500).json({ error: "Invalid JSON from Python." });
    }
  });
};
