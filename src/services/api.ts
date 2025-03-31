import { API_URL, API_ENDPOINTS } from '../config';

interface ImageFilters {
  brightness: number;
  contrast: number;
  saturation: number;
}

export async function predictImage(
  imageData: string,
  model: 'cnn' | 'xception',
  filters: ImageFilters | null, // Allow filters to be null
  isCameraInput: boolean
) {
  try {
    // Map frontend model names to backend model names
    const backendModel = model === 'cnn' ? 'CustomCNN' : 'Xception71';

    // Ensure the base64 image is in the correct format
    let base64Image = imageData;
    if (imageData.startsWith('data:image')) {
      base64Image = imageData.split(',')[1];
    }

    // Log the request payload for debugging
    const payload = {
      image: base64Image.substring(0, 50) + '...', // Log a snippet of the base64 string
      model: backendModel,
      filters,
      isCameraInput,
    };
    console.log('Sending prediction request:', payload);

    const response = await fetch(`${API_URL}${API_ENDPOINTS.predict}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: base64Image,
        model: backendModel,
        filters: filters || null, // Ensure filters is null if not provided
        isCameraInput,
      }),
    });

    if (!response.ok) {
      const errorDetails = await response.text();
      console.error(`Prediction failed: ${response.status} - ${response.statusText}. Details: ${errorDetails}`);
      throw new Error(errorDetails || 'Failed to process the image. Please try again.');
    }

    const result = await response.json();
    console.log('Prediction result:', result);
    return result;
  } catch (error) {
    console.error('Error during image prediction:', error);
    throw new Error(error.message || 'An error occurred while processing the image. Please try again.');
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
