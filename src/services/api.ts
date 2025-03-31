import { API_URL, API_ENDPOINTS } from '../config';

interface ImageFilters {
  brightness: number;
  contrast: number;
  saturation: number;
}

export async function predictImage(
  imageData: string,
  model: 'cnn' | 'xception',
  filters: ImageFilters,
  isCameraInput: boolean
) {
  try {
    const backendModel = model === 'cnn' ? 'CustomCNN' : 'Xception71';
    const base64Image = imageData.split(',')[1] || imageData;

    const response = await fetch(`${API_URL}${API_ENDPOINTS.predict}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: base64Image,
        model: backendModel,
        filters,
        isCameraInput // Add this to the payload
      }),
    });

    if (!response.ok) {
      const errorDetails = await response.text();
      throw new Error(`Prediction failed: ${response.statusText}. Details: ${errorDetails}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error during image prediction:', error);
    throw new Error('Failed to process the image. Please try again.');
  }
}

export async function flagPrediction(predictionId: string, feedback: string) {
  try {
    const response = await fetch(`${API_URL}${API_ENDPOINTS.flag}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        id: predictionId,
        feedback,
      }),
    });

    if (!response.ok) {
      const errorDetails = await response.text();
      throw new Error(`Flagging failed: ${response.statusText}. Details: ${errorDetails}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error during flagging:', error);
    throw new Error('Failed to flag the result. Please try again.');
  }
}