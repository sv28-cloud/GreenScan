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
    if (!image) return;
    setLoading(true);
    setResult("");

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Data = reader.result.split(",")[1];
      const prompt =
        "Classify this item as recyclable or not recyclable and explain why in one or two sentences.";

      try {
        const response = await model.generateContent([
          prompt,
          {
            inlineData: { mimeType: image.type, data: base64Data },
          },
        ]);

        const text = response.response.text();
        setResult(text);

        // Detect recyclable vs not recyclable
        if (
          text.toLowerCase().includes("recyclable") &&
          !text.toLowerCase().includes("not recyclable")
        ) {
          setIsRecyclable(true);
        } else {
          setIsRecyclable(false);
        }
      } catch (err) {
        setResult("Sorry, I couldn’t analyze this image.");
      } finally {
        setLoading(false);
      }
    };

    reader.readAsDataURL(image);
  };

  return (
    <div className="app">
      <div className="overlay"></div>
      <h1 className="title">♻️ GreenScan – AI Recycling Assistant</h1>

      <div className="card">
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
          <div className="preview">
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
