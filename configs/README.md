# Configs (plain explanation)

This folder keeps shared settings and schemas.

What’s inside
- schemas/ — data shapes for datasets, goldens, and run config
- metrics.json — which metrics the UI/backend should use by default
- coverage.json — rules for creating new datasets

Good to know
- The app stores datasets and runs by “vertical” (like commerce or banking):
	- datasets/<vertical>/
	- runs/<vertical>/
- Token and context limits are controlled by settings (see the Settings page).
