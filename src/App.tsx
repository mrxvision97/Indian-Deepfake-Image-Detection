import React, { useState, useRef, useEffect } from 'react';
import {
  Upload,
  Camera,
  Sliders,
  AlertTriangle,
  Check,
  X,
  RefreshCw,
  Shield,
  Eye,
  FileWarning,
  Info,
  AlertCircle,
  Zap,
  Lock,
  Scan,
} from 'lucide-react';
import { predictImage, flagPrediction } from './services/api';

// Define types for filters and prediction results
interface ImageFilters {
  brightness: number;
  contrast: number;
  saturation: number;
}

interface PredictionResult {
  isReal: boolean;
  probability: number;
  model: string;
}

interface InfoTip {
  id: number;
  text: string;
  icon: React.ReactNode;
}

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<ImageFilters>({
    brightness: 100,
    contrast: 100,
    saturation: 100,
  });
  const [selectedModel, setSelectedModel] = useState<'cnn' | 'xception'>('cnn');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCameraInput, setIsCameraInput] = useState(false);
  const [showWarningBanner, setShowWarningBanner] = useState(true);
  const [currentInfoTip, setCurrentInfoTip] = useState<number>(0);
  const [imageOrientation, setImageOrientation] = useState<'landscape' | 'portrait'>('landscape');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // Deepfake facts and info tips
  const infoTips: InfoTip[] = [
    {
      id: 1,
      text: "Deepfakes use AI to create convincing fake images or videos of real people.",
      icon: <Zap size={20} className="text-orange-400" />,
    },
    {
      id: 2,
      text: "In 2024, over 85% of Indian social media users encountered deepfakes.",
      icon: <AlertCircle size={20} className="text-orange-400" />,
    },
    {
      id: 3,
      text: "Our AI models can detect 94% of current deepfake technologies.",
      icon: <Shield size={20} className="text-orange-400" />,
    },
    {
      id: 4,
      text: "Always verify media sources before sharing sensitive content.",
      icon: <Lock size={20} className="text-orange-400" />,
    },
    {
      id: 5,
      text: "Report suspected deepfakes to help improve our detection systems.",
      icon: <Info size={20} className="text-orange-400" />,
    },
  ];

  // Cycle through info tips
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentInfoTip((prev) => (prev + 1) % infoTips.length);
    }, 5000);
    return () => clearInterval(interval);
  }, [infoTips.length]);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
        setIsCameraInput(false);
        setResult(null);
        
        // Create an image element to determine orientation
        const img = new Image();
        img.onload = () => {
          setImageOrientation(img.width >= img.height ? 'landscape' : 'portrait');
        };
        img.src = reader.result as string;
      };
      reader.readAsDataURL(file);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      alert('Failed to access the camera. Please check your permissions.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      if (!context) return;

      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      context.drawImage(videoRef.current, 0, 0);

      const capturedImage = canvasRef.current.toDataURL('image/jpeg');
      setSelectedImage(capturedImage);
      setIsCameraInput(true);
      setResult(null);
      
      // Determine orientation of captured image
      const img = new Image();
      img.onload = () => {
        setImageOrientation(img.width >= img.height ? 'landscape' : 'portrait');
      };
      img.src = capturedImage;
      
      stopCamera();
    }
  };

  const handleFilterChange = (filter: keyof ImageFilters, value: number) => {
    setFilters((prev) => ({ ...prev, [filter]: value }));
  };

  const processImage = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    try {
      const backendModel = selectedModel === 'cnn' ? 'CustomCNN' : 'Xception71';
      const result = await predictImage(
        selectedImage,
        selectedModel,
        filters,
        isCameraInput
      );
      setResult(result);
    } catch (error) {
      console.error('Error processing image:', error);
      alert('An error occurred while processing the image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const flagResult = async () => {
    if (!result) return;

    try {
      await flagPrediction('temp-id', 'Incorrect prediction');
      alert('Result has been flagged for review. Thank you for your feedback!');
    } catch (error) {
      console.error('Error flagging result:', error);
      alert('Failed to flag the result. Please try again later.');
    }
  };

  const resetImage = () => {
    setSelectedImage(null);
    setResult(null);
    setIsProcessing(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // New function to handle testing another image
  const testAnotherImage = () => {
    setSelectedImage(null);
    setResult(null);
    setIsProcessing(false);
    setFilters({
      brightness: 100,
      contrast: 100,
      saturation: 100,
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white relative overflow-hidden">
      {/* Animated gradient background */}
      <div className="absolute inset-0 gradient-animate"></div>
      
      {/* Floating animated elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(10)].map((_, i) => (
          <div 
            key={i}
            className="absolute animate-float opacity-20" 
            style={{
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              animation: `float ${8 + Math.random() * 10}s linear infinite`,
              animationDelay: `${Math.random() * 5}s`
            }}
          >
            {i % 2 === 0 ? (
              <Shield size={48 + Math.random() * 24} className="text-orange-500" />
            ) : (
              <AlertTriangle size={48 + Math.random() * 24} className="text-red-500" />
            )}
          </div>
        ))}
      </div>

      {/* Warning Banner */}
      {showWarningBanner && (
        <div className="relative z-20 bg-gradient-to-r from-red-700 via-orange-600 to-red-700 text-white py-2 px-4 animate-pulse">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <AlertTriangle size={24} className="text-yellow-300" />
              <p className="text-sm md:text-base font-medium">
                Warning: Deepfake media can be used for scams and misinformation. Stay vigilant!
              </p>
            </div>
            <button 
              onClick={() => setShowWarningBanner(false)}
              className="text-white hover:text-yellow-300 transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>
      )}

      <div className="relative z-10 p-4 md:p-8">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-8 animate-fadeIn">
            <div className="flex justify-center mb-4">
              <div className="relative">
                <Shield size={64} className="text-orange-500" />
                <div className="absolute -top-1 -right-1 bg-red-500 h-3 w-3 rounded-full animate-ping"></div>
              </div>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 via-yellow-300 to-green-500">
              भारतीय Deepfake Detection
            </h1>
            <p className="text-gray-300 text-lg">Protecting Digital Truth with Advanced AI</p>
          </div>

          {/* Info Tips Carousel */}
          <div className="mb-8 bg-gray-800/30 backdrop-blur-md rounded-xl border border-orange-500/30 overflow-hidden shadow-lg animate-slideIn">
            <div className="relative p-4">
              <div className="flex items-center">
                {infoTips[currentInfoTip].icon}
                <span className="ml-2 font-semibold text-orange-300">Did you know?</span>
              </div>
              <p className="mt-2 text-gray-200 transition-all duration-500 animate-fadeIn">
                {infoTips[currentInfoTip].text}
              </p>
              <div className="mt-3 flex">
                {infoTips.map((_, index) => (
                  <div 
                    key={index} 
                    className={`h-1 rounded-full mr-1 flex-grow transition-all duration-300 ${
                      index === currentInfoTip ? 'bg-orange-500' : 'bg-gray-600'
                    }`}
                  ></div>
                ))}
              </div>
            </div>
          </div>

          {/* Main App Container */}
          <div className="bg-gray-800/80 backdrop-blur-xl rounded-xl p-6 shadow-2xl mb-8 border border-gray-700 transform transition-all duration-500 hover:shadow-orange-500/20 animate-scaleIn">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Left Column - Controls */}
              <div className="space-y-4">
                <div className="flex items-center gap-2 mb-4">
                  <div className="relative">
                    <Eye size={28} className="text-blue-400" />
                    <div className="absolute inset-0 bg-blue-400 rounded-full opacity-30 animate-ping"></div>
                  </div>
                  <h2 className="text-xl font-semibold">Image Analysis</h2>
                </div>

                <div className="flex gap-4">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-orange-600 to-orange-500 hover:from-orange-700 hover:to-orange-600 px-4 py-3 rounded-lg transition-all transform hover:scale-105 shadow-lg hover:shadow-orange-500/50"
                  >
                    <Upload size={20} />
                    Upload Image
                  </button>
                  <button
                    onClick={isCameraActive ? stopCamera : startCamera}
                    className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-green-600 to-green-500 hover:from-green-700 hover:to-green-600 px-4 py-3 rounded-lg transition-all transform hover:scale-105 shadow-lg hover:shadow-green-500/50"
                  >
                    <Camera size={20} />
                    {isCameraActive ? 'Stop Camera' : 'Use Camera'}
                  </button>
                </div>

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleImageUpload}
                  accept="image/*"
                  className="hidden"
                />

                <div className="space-y-4">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value as 'cnn' | 'xception')}
                    className="bg-gray-700 text-white px-4 py-3 rounded-lg w-full border border-gray-600 focus:border-blue-500 transition-colors hover:border-blue-400 cursor-pointer"
                  >
                    <option value="cnn">Custom CNN Model (Faster)</option>
                    <option value="xception">Xception71 Model (More Accurate)</option>
                  </select>

                  <button
                    onClick={() => setShowFilters(!showFilters)}
                    className="flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 px-4 py-3 rounded-lg transition-all w-full transform hover:scale-105 shadow-lg hover:shadow-purple-500/50"
                  >
                    <Sliders size={20} />
                    {showFilters ? 'Hide Image Adjustments' : 'Show Image Adjustments'}
                  </button>
                </div>

                {showFilters && (
                  <div className="space-y-4 bg-gray-700/50 backdrop-blur-sm p-4 rounded-lg border border-gray-600 animate-fadeIn">
                    <div>
                      <label className="block mb-2 flex justify-between">
                        <span>Brightness</span>
                        <span className="text-orange-300">{filters.brightness}%</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="200"
                        value={filters.brightness}
                        onChange={(e) => handleFilterChange('brightness', Number(e.target.value))}
                        className="w-full accent-orange-500"
                      />
                    </div>
                    <div>
                      <label className="block mb-2 flex justify-between">
                        <span>Contrast</span>
                        <span className="text-orange-300">{filters.contrast}%</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="200"
                        value={filters.contrast}
                        onChange={(e) => handleFilterChange('contrast', Number(e.target.value))}
                        className="w-full accent-orange-500"
                      />
                    </div>
                    <div>
                      <label className="block mb-2 flex justify-between">
                        <span>Saturation</span>
                        <span className="text-orange-300">{filters.saturation}%</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="200"
                        value={filters.saturation}
                        onChange={(e) => handleFilterChange('saturation', Number(e.target.value))}
                        className="w-full accent-orange-500"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column - Image Preview - CHANGED TO HANDLE PORTRAIT IMAGES */}
              <div className="relative group">
                <div className="absolute inset-0 bg-gradient-to-b from-transparent to-gray-900/20 pointer-events-none rounded-lg"></div>
                <div className="absolute inset-0 border-2 border-gray-700 group-hover:border-orange-500/50 rounded-lg transition-colors duration-300"></div>
                
                {selectedImage ? (
                  <div className="relative flex justify-center items-center">
                    <div className={`${imageOrientation === 'portrait' ? 'max-h-96' : 'max-h-64'} overflow-hidden rounded-lg w-full flex justify-center`}>
                      <img
                        ref={imageRef}
                        src={selectedImage}
                        alt="Selected"
                        className={`${imageOrientation === 'portrait' ? 'h-auto max-h-96 w-auto' : 'w-full h-64 object-cover'} rounded-lg`}
                        style={{
                          filter: `brightness(${filters.brightness}%) contrast(${filters.contrast}%) saturate(${filters.saturation}%)`,
                        }}
                      />
                    </div>
                    <button
                      onClick={resetImage}
                      className="absolute top-2 right-2 bg-red-600/70 hover:bg-red-700 p-2 rounded-full transition-all transform hover:scale-110"
                    >
                      <X size={16} />
                    </button>
                    
                    {/* Scan overlay animation when processing */}
                    {isProcessing && (
                      <div className="absolute inset-0 rounded-lg overflow-hidden">
                        <div className="h-1 bg-blue-500/50 w-full absolute animate-scanline"></div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="relative">
                    <video
                      ref={videoRef}
                      autoPlay
                      className="w-full h-64 object-cover rounded-lg"
                    />
                    {!isCameraActive && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center text-gray-400">
                          <Scan size={48} className="mx-auto mb-2 opacity-50" />
                          <p>Upload an image or use camera</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
                <canvas ref={canvasRef} className="hidden" />
                {isCameraActive && (
                  <button
                    onClick={capturePhoto}
                    className="absolute bottom-4 right-4 bg-red-600 hover:bg-red-700 p-3 rounded-full transition-all transform hover:scale-110 animate-pulse shadow-lg"
                  >
                    <Camera size={24} />
                  </button>
                )}
              </div>
            </div>

            {selectedImage && !isProcessing && !result && (
              <button
                onClick={processImage}
                className="mt-6 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 px-6 py-3 rounded-lg transition-all w-full flex items-center justify-center gap-2 transform hover:scale-105 shadow-lg hover:shadow-blue-500/50 group"
              >
                <FileWarning size={24} className="group-hover:animate-pulse" />
                <span className="text-lg font-medium">Analyze for Deepfakes</span>
              </button>
            )}

            {isProcessing && (
              <div className="mt-6 text-center">
                <RefreshCw size={36} className="animate-spin mx-auto text-orange-500" />
                <p className="mt-4 text-lg">
                  Analyzing image with advanced AI...
                </p>
                <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
                  <div className="h-2 rounded-full bg-gradient-to-r from-orange-500 to-red-500 animate-loading"></div>
                </div>
              </div>
            )}

            {result && (
              <div
                className={`mt-6 p-6 rounded-lg backdrop-blur-sm border animate-fadeIn ${
                  result.isReal 
                    ? 'bg-green-600/20 border-green-500/30' 
                    : 'bg-red-600/20 border-red-500/30'
                }`}
              >
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                  <div className="flex items-center gap-3">
                    {result.isReal ? (
                      <div className="relative">
                        <div className="absolute inset-0 bg-green-500 rounded-full animate-ping opacity-50"></div>
                        <Check size={36} className="text-green-500 relative z-10" />
                      </div>
                    ) : (
                      <div className="relative">
                        <div className="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-50"></div>
                        <X size={36} className="text-red-500 relative z-10" />
                      </div>
                    )}
                    <div>
                      <h3 className="text-2xl font-bold mb-1">
                        {result.isReal ? 'Real Image Verified' : 'Deepfake Detected'}
                      </h3>
                      <p className="text-sm text-gray-300">
                        {result.isReal 
                          ? 'Our AI confirms this image appears to be authentic.' 
                          : 'Warning: This image shows signs of AI manipulation.'}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={flagResult}
                    className="flex items-center gap-2 bg-yellow-600/20 hover:bg-yellow-600/30 px-4 py-2 rounded-lg transition-all transform hover:scale-105 border border-yellow-600/30"
                  >
                    <AlertTriangle size={20} />
                    Report False Result
                  </button>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                  <div className="bg-gray-800/50 p-4 rounded-lg">
                    <p className="flex items-center gap-2 mb-2">
                      <span className="text-gray-300">AI Confidence:</span>
                      <span className="font-bold text-lg">
                        {result.probability.toFixed(1)}%
                      </span>
                    </p>
                    <div className="w-full bg-gray-700 rounded-full h-3 mt-1">
                      <div
                        className={`h-3 rounded-full ${
                          result.isReal ? 'bg-green-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${result.probability}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 p-4 rounded-lg">
                    <p className="flex items-center gap-2">
                      <span className="text-gray-300">Detection Model:</span>
                      <span className="font-semibold">
                        {result.model === 'CustomCNN' ? 'Custom CNN' : 'Xception71'} 
                        <span className="text-xs ml-2 bg-blue-900/50 px-2 py-1 rounded-full">
                          {result.model === 'CustomCNN' ? 'Optimized' : 'High Precision'}
                        </span>
                      </span>
                    </p>
                    <p className="text-xs text-gray-400 mt-2">
                      {result.model === 'CustomCNN' 
                        ? 'Our custom CNN model is optimized for speed while maintaining high accuracy.' 
                        : 'Xception71 offers the highest precision for detecting sophisticated deepfakes.'}
                    </p>
                  </div>
                </div>
                
                {/* Added "Test Another Image" button here */}
                <button
                  onClick={testAnotherImage}
                  className="mt-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 px-6 py-3 rounded-lg transition-all w-full flex items-center justify-center gap-2 transform hover:scale-105 shadow-lg hover:shadow-purple-500/50"
                >
                  <RefreshCw size={24} />
                  <span className="text-lg font-medium">Test Another Image</span>
                </button>
              </div>
            )}
          </div>

          {/* Info Box */}
          <div className="bg-gray-800/60 backdrop-blur-md rounded-xl p-6 border border-gray-700 shadow-xl animate-slideIn">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Info size={24} className="text-orange-400" />
              About Deepfake Detection
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-4 bg-gray-700/30 rounded-lg border border-gray-600 hover:border-orange-500/30 transition-colors hover:shadow-orange-500/10 hover:shadow-lg">
                <h4 className="font-medium mb-2 text-orange-300">What are Deepfakes?</h4>
                <p className="text-sm text-gray-300">
                  Deepfakes use artificial intelligence to create convincing fake images or videos that 
                  show people saying or doing things they never did in reality.
                </p>
              </div>
              <div className="p-4 bg-gray-700/30 rounded-lg border border-gray-600 hover:border-orange-500/30 transition-colors hover:shadow-orange-500/10 hover:shadow-lg">
                <h4 className="font-medium mb-2 text-orange-300">Our Technology</h4>
                <p className="text-sm text-gray-300">
                  We use advanced neural networks trained on millions of images to detect 
                  subtle inconsistencies and artifacts that are invisible to the human eye.
                </p>
              </div>
              <div className="p-4 bg-gray-700/30 rounded-lg border border-gray-600 hover:border-orange-500/30 transition-colors hover:shadow-orange-500/10 hover:shadow-lg">
                <h4 className="font-medium mb-2 text-orange-300">Stay Safe</h4>
                <p className="text-sm text-gray-300">
                  Verify the source of media before trusting it. If you suspect a deepfake,
                  use our tool to analyze it and report suspicious content.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Bottom Banner */}
      <div className="relative z-10 bg-gradient-to-r from-orange-900/70 via-orange-800/70 to-orange-900/70 backdrop-blur-md py-3 mt-8 border-t border-orange-600/30">
        <div className="max-w-5xl mx-auto px-4 text-center">
          <p className="text-sm text-orange-200">
            भारतीय Deepfake Detection • Fighting Misinformation with AI • © 2025
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;