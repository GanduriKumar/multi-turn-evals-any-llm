# Frontend (plain explanation)

This is the web app. It lets you click through everything instead of using only the command line.

How to run it
1) Start the backend first (http://localhost:8000)
2) In this folder:
	- npm install (first time)
	- npm run dev
3) Open the URL shown (usually http://localhost:5173)

What you can do here
- Dataset Generator — make new test datasets
- Datasets Viewer — upload or pick a dataset to test
- Runs — start a run and watch progress
- Reports — open or download the results (HTML/CSV/JSON)
- Settings — set API keys, default models, and thresholds
- Metrics — turn metrics on/off and adjust thresholds

Notes
- The dev server proxies calls to http://localhost:8000 so the UI can reach the backend.
- PDF export works when the server has its system tools installed.
