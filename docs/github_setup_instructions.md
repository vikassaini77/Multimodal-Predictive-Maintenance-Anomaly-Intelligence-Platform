# Final Showcase Checklist

To complete Day 29 and maximize your recruiter impact, you need to perform these final manual steps in your GitHub repository.

## 1. Record & Upload Demo (Video & GIF)

### The 3-Minute Video
1. Start your stack: `docker compose up --build -d`
2. Run the demo data generator in another terminal: `python scripts/demo_generator.py`
3. Use **Loom** or **OBS** to record your screen while the anomaly propagates and the agent diagnoses it.
4. Upload to YouTube (Unlisted is fine).
5. Open `README.md` and `docs/index.html` and replace the video placeholder links with your actual YouTube URL.

### The 10-Second GIF
1. Use **ScreenToGif** or **LICEcap** to capture a highly compressed 10-second loop of the dashboard detecting an anomaly.
2. Save it as `demo.gif`.
3. Upload `demo.gif` to your repository (e.g., inside a `docs/assets/` folder or drag-and-drop into a GitHub issue to get a CDN link).
4. Update the GIF placeholder link at the very top of `README.md`.

## 2. Update GitHub Bio

Navigate to your repository's main page on GitHub and click the **⚙️ (gear icon)** next to the "About" section on the right sidebar. 

Copy and paste this punchy sentence into the **Description**:
> Built IndustrialMind AI — a multimodal predictive maintenance platform with Two-Tower learning, GraphSAGE, ReAct agent, and TensorRT FP16 edge deployment.

## 3. Add GitHub Topics

In the same "About" settings modal, add these exact topics to make your repo highly discoverable in GitHub Search:
- `machine-learning`
- `computer-vision`
- `pytorch`
- `gnn`
- `rag`
- `llm`
- `tensorrt`
- `anomaly-detection`
- `predictive-maintenance`
- `industrial-ai`

## 4. Enable GitHub Pages

Give recruiters a clean URL to click on instead of just raw code.
1. Go to your repository **Settings** tab.
2. Click on **Pages** in the left sidebar.
3. Under **Source**, select `Deploy from a branch`.
4. Under **Branch**, select `main` and then choose the `/docs` folder from the dropdown.
5. Click **Save**.
6. Wait 1-2 minutes for the Action to run. Once complete, grab the public URL (e.g., `https://vikassaini77.github.io/Multimodal-Predictive-Maintenance-Anomaly-Intelligence-Platform/`) and paste it into your repository's "Website" field in the About section.
