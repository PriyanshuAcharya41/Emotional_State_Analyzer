import express from 'express';
import { analyzeEmotion } from '../controllers/analyzeController.js';
import multer from 'multer';

const router = express.Router();

const storage = multer.diskStorage({
  destination: 'uploads/',
  filename: (req, file, cb) => {
    const suffix = file.fieldname === 'audio' ? 'voice.wav' : 'face.jpg';
    cb(null, Date.now() + '-' + suffix);
  }
});

const upload = multer({ storage });

router.post(
  '/analyze',
  upload.fields([{ name: 'audio', maxCount: 1 }, { name: 'image', maxCount: 1 }]),
  analyzeEmotion
);

export default router;
