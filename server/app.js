import express from 'express';
import cors from 'cors';
import emotionRoutes from './routes/emotion.js';

const app = express();
app.use(cors());
app.use('/uploads', express.static('uploads'));
app.use('/analyze', emotionRoutes);

app.listen(5000, () => {
  console.log("🚀 Server running on port 5000");
});
