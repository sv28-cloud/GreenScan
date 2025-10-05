import { useState } from "react";
import "./App.css";
import { GoogleGenerativeAI } from "@google/generative-ai";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [isRecyclable, setIsRecyclable] = useState(null);

  const genAI = new GoogleGenerativeAI(import.meta.env.VITE_GEMINI_API_KEY);

  const handleImageUpload = (e) => {
    setImage(e.target.files[0]);
    setResult("");
    setIsRecyclable(null);
  };

  const analyzeImage = async () => {
    if (!image) {
      console.warn("⚠️ No image selected.");
      return;
    }

    console.log("🟡 Starting image analysis...");
    setLoading(true);
    setResult("");

    const reader = new FileReader();

    reader.onloadend = async () => {
      console.log("📸 Image loaded into base64, sending to Flask backend...");

      try {
        // --- 1️⃣ Send image to Flask backend ---
        const formData = new FormData();
        formData.append("file", image);

        console.log("📤 Sending POST request to Flask backend...");
        const flaskResponse = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          body: formData,
        });

        console.log("🛰️ Flask response status:", flaskResponse.status);

        if (!flaskResponse.ok) {
          const errorText = await flaskResponse.text();
          console.error("❌ Flask error response:", errorText);
          throw new Error("Backend returned an error");
        }

        const flaskData = await flaskResponse.json();
        console.log("✅ Flask returned JSON:", flaskData);

        const predictedClass = flaskData.prediction || flaskData.predicted_class;

        if (!predictedClass) {
          console.error("⚠️ No predicted class found in Flask response");
          setResult("Error: No prediction returned from backend.");
          return;
        }

        console.log("🧠 Backend predicted class:", predictedClass);
        setResult(`Predicted: ${predictedClass}`);

        // --- 2️⃣ Decide recyclable or not ---
        const lower = predictedClass.toLowerCase();
        if (["plastic", "paper", "glass", "metal", "cardboard"].some((m) => lower.includes(m))) {
          console.log("♻️ Marked as recyclable (based on backend prediction)");
          setIsRecyclable(true);
        } else {
          console.log("🚫 Marked as NOT recyclable (based on backend prediction)");
          setIsRecyclable(false);
        }
      } catch (err) {
        console.error("❌ Error during analysis:", err);
        setResult("Sorry, something went wrong while analyzing the image.");
      } finally {
        console.log("✅ Done. Resetting loading state.");
        setLoading(false);
      }
    };

    reader.readAsDataURL(image);
  };

  return (
    <div className="app fade-in">
      <div className="overlay"></div>
      <h1 className="title slide-up"> ♻️ GreenScan – AI Recycling Assistant</h1>

      <div className="card slide-up-delayed">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="upload"
        />

        <button onClick={analyzeImage} disabled={!image || loading}>
          {loading ? "Analyzing..." : "Check Recyclability"}
        </button>

        {image && (
          <div className="preview fade-in-delayed">
            <h3>Uploaded Image:</h3>
            <img src={URL.createObjectURL(image)} alt="preview" />
          </div>
        )}

        {result && (
          <div
            className={`result-box ${
              isRecyclable ? "recycle" : "not-recycle"
            } animate`}
          >
            {isRecyclable ? (
              <>
                <h2>♻️ Recyclable</h2>
                <img
                  src="https://cdn-icons-png.flaticon.com/512/558/558100.png"
                  alt="Recycling Bin"
                  className="icon"
                />
                <p className="explanation">{result}</p>
              </>
            ) : (
              <>
                <h2>🚫 Not Recyclable</h2>
                <div className="icon-row">
                  <img
                    src="/notrecycle.png"
                    alt="Trash Bin"
                    className="icon-large"
                  />
                  <img
                    src="/simple.png"
                    alt="No Recycle Symbol"
                    className="icon-large"
                  />
                </div>
                <p className="explanation">{result}</p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
