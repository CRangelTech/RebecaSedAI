const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const path = require("path");
const app = express();

app.use(cors());
app.use(bodyParser.json());

app.post("/generate", (req, res) => {
  const prompt = req.body.prompt;

  if (!prompt) {
    return res.status(400).json({ error: "Prompt is required" });
  }

  // Ruta relativa al script predict.py
  const scriptPath = path.join(__dirname, "..", "MODEL", "predict.py");

  const python = spawn("python", [scriptPath, prompt]);

  let output = "";
  let errorOutput = "";

  python.stdout.on("data", (data) => {
    output += data.toString();
  });

  python.stderr.on("data", (data) => {
    errorOutput += data.toString();
  });

  python.on("close", (code) => {
    if (errorOutput) {
      console.error("❌ Error de Python:", errorOutput);
      return res.status(500).json({ error: "Python error", details: errorOutput });
    }

    try {
      const parsed = JSON.parse(output.trim());
      return res.json(parsed);
    } catch (err) {
      console.error("❌ Error al parsear JSON:", err.message);
      console.error("Salida recibida:", output);
      return res.status(500).json({ error: "Failed to parse model output" });
    }
  });
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`🚀 Servidor iniciado en http://localhost:${PORT}`);
});
