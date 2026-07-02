Golf Swing Analyzer: 5-Day Project Roadmap & Engineering Blueprint

https://share.gemini.google/gXF9ycA8BNLD
https://gemini.google.com/app/4af58384991fea17

This document outlines the end-to-end technical roadmap for building a biomechanical golf swing analysis pipeline. The architecture uses MediaPipe Pose to handle spatial coordinate extraction, a Savitzky-Golay filter to remove camera noise, XGBoost with a sliding window to classify timeline milestones, NumPy for 2D biomechanical evaluation, and OpenCV to produce a synchronized side-by-side analysis visualization and coaching report.

Project Architecture Flow

[Raw Input Video] 
       │
       ▼
[MediaPipe Pose] ───► Extract 2D/3D Coordinates (X, Y)
       │
       ▼
[Signal Smoothing] ───► Savitzky-Golay Filter removes micro-jitters
       │
       ▼
[Normalization Engine] ───► Center at mid-hip, scale based on Day 1 Address Torso Length
       │
       ▼
[Feature Engineering] ───► Create +/- 5 frame "Sliding Window" for dynamic joints
       │
       ▼
[XGBoost Event Classifier] ───► Identify exact frame indices for the 8 milestones
       │
       ▼
[Pro Matchmaker & Rules Engine] ──► Match Pro by limb ratio; evaluate 2D angles; generate text feedback
       │
       ▼
[Visual Stitching Engine] ───► Output side-by-side MP4 and Text Report


🗂️ Phase 0: Setup, Data Acquisition & Project Structure

Before writing any code, set up your physical workspace and download the required datasets.

1. Downloading the Data

The Video Files: We will use a pre-packaged Kaggle dataset to avoid broken YouTube links.

Install the Kaggle CLI: pip install kaggle

Run: kaggle datasets download wmcnally/golfdb

Unzip the downloaded file. This contains the raw .mp4 video clips.

The Annotations: Download the official GolfDB.mat file from the original author's GitHub/Google Drive links (search for "wmcnally GolfDB github"). This contains the labeled frame indices.

2. Project Folder Structure
Organize your workspace exactly like this so your modular scripts can easily import each other:

golf_analyzer_project/
│
├── data/
│   ├── raw_videos/              # Unzipped Kaggle .mp4 files go here
│   ├── annotations/             # GolfDB.mat goes here
│   └── processed/               # Day 2's master_training_dataset.csv goes here
│
├── models/
│   └── golf_phase_model.json    # Day 3's trained XGBoost model saves here
│
├── output/                      # User reports and side-by-side MP4s save here
│
├── src/                         # All your Python logic
│   ├── data_processor.py        # (Day 1) MediaPipe & Smoothing
│   ├── dataset_builder.py       # (Day 2) Sliding Window Engineering
│   ├── train_classifier.py      # (Day 3) XGBoost Training
│   ├── coaching_engine.py       # (Day 4) Pro Matcher & Math Rules
│   └── visual_stitcher.py       # (Day 5) OpenCV Video Renderer
│
├── notebooks/                   # Jupyter Notebooks for testing/visual demos
│   └── data_exploration.ipynb   
│
└── analyze_swing.py             # (Day 6) The final End-to-End script


🛠️ Day 1: Frame Processing, Smoothing & Normalization

Objective: Parse incoming video frames, run pose extraction, mathematically smooth the noise, establish the static baseline "anchor" scale, and output normalized joint vectors.

File Workflow: Prototype the code in notebooks/data_exploration.ipynb. Once it works, copy the finalized functions into src/data_processor.py.

Inputs: Raw golf swing video file (.mp4).

Outputs: A Pandas DataFrame where each row is a frame, containing smoothed, normalized $X, Y$ positions for the key joints, plus a video_id column.

Core Tasks:

Extract landmarks using mediapipe.solutions.pose.

Apply a Savitzky-Golay filter (scipy.signal.savgol_filter with window=11, polyorder=3) to the raw coordinate columns to eliminate MediaPipe micro-jitters.

Normalize by centering coordinates relative to the mid-hip.

Calculate static TORSO_SCALE from the Address frames and divide all coordinates by this scalar to make the data resolution-independent.

👀 Visual Demo Step: Draw the raw wrist X/Y as a red dot, and the smoothed wrist X/Y as a green dot on a blank video canvas. Watch the playback: the green dot should travel in a perfectly smooth arc while the red dot jitters.

🤖 Antigravity Prompt: "Act as a computer vision engineer. In our Jupyter notebook, write code to process a golf swing video. Use OpenCV to open the file and MediaPipe Pose to extract landmarks. Apply a Savitzky-Golay filter (window=11, polyorder=3) to smooth the X/Y coordinates. Implement normalization that centers all coordinates relative to the mid-hip point, and scale everything by a static 'torso scale factor' calculated from the first 5 frames. Once we verify this works visually, compile it into a reusable class for src/data_processor.py."

📊 Day 2: GolfDB Parsing & Sliding Window Engineering

Objective: Clean the GolfDB metadata, run Day 1's code on the videos, and engineer the "sliding window" features to give XGBoost temporal context.

File Workflow: Prototype the parsing and pandas logic in notebooks/data_exploration.ipynb. Once the CSV looks right, copy the code into src/dataset_builder.py.

Inputs: GolfDB annotations (GolfDB.mat) and downloaded raw .mp4 videos in data/raw_videos/.

Outputs: A master training_dataset.csv in data/processed/.

Core Tasks:

Load the GolfDB .mat file to get the 8 milestone frame indices for each video.

For each frame $T$, append the X/Y coordinates of high-movement joints (wrists, elbows, shoulders, hips) from frames $T-5$ and $T+5$ as new columns in the same row. Group by video_id so windows don't bleed across different videos.

Labeling: Assign $1-8$ to the exact milestone frames, and $0$ to all other transitional frames.

👀 Visual Demo Step: Open the generated CSV in Excel/Pandas. Look at a "Top of Backswing" row and verify you see the Y-coordinates of the wrist rising in the $T-5$ columns and falling in the $T+5$ columns.

🤖 Antigravity Prompt: "In our notebook, write code to parse the GolfDB mat file and map the 8 key swing events to local videos. For every video, run our Day 1 functions. Grouping by video_id, create 'sliding window' features for the wrists, elbows, shoulders, and hips: for each row T, append the coordinates from T-5 and T+5 as new columns. Append a label column: 1-8 for the milestone frames, and 0 for transition frames. Output a clean master_training_dataset.csv. Once verified, save this logic to src/dataset_builder.py."

🧠 Day 3: XGBoost Classifier Training

Objective: Train an XGBoost multiclass model to act as your sequence locator using the sliding window features.

File Workflow: Train and evaluate the model in notebooks/data_exploration.ipynb. Once accurate, save the training and inference functions to src/train_classifier.py and save the model to models/.

Inputs: Engineered master_training_dataset.csv.

Outputs: Serialized model (golf_phase_model.json) saved to models/ and an inference function returning the 8 frame indices.

Core Tasks:

Group train/test split strictly by video_id to prevent data leakage.

Train an XGBClassifier to predict classes 0-8, using class weights to penalize the model for guessing $0$ (since most frames are $0$).

Inference logic: Run np.argmax() over the probability distributions for classes $1-8$ to find the exact frame where each milestone peaks.

👀 Visual Demo Step: Run the inference script on a test video and use OpenCV to flash giant green text (e.g., "IMPACT!") on the exact frame the model predicts. Watch the video to verify the text flashes at the correct physical moment.

🤖 Antigravity Prompt: "In our notebook, read the Day 2 dataset. Split the dataset ensuring video_ids are kept entirely in either train or test sets. Train an XGBoost multiclass classifier to predict classes 0-8, using sample weights for class imbalance. Save the model as JSON. Write an inference function predict_swing_milestones(video_features, model) that runs predictions and uses argmax on the probabilities to identify the exact frame index where each of the 8 events occur. Finally, format this as src/train_classifier.py."

📐 Day 4: 2D Biomechanical Rules Engine & Matcher

Objective: Match the user to a Pro based on physical build, calculate 2D metrics at the detected milestones, and map failures to pre-written text feedback.

File Workflow: Test the math and rules engine in notebooks/data_exploration.ipynb. Once the angles are accurate, migrate it to src/coaching_engine.py.

Inputs: User's 8 milestone frames, Camera View tag (Face-On vs DTL).

Outputs: Matched Pro ID, and a generated Python dictionary of text feedback (e.g., "Towel Drill").

Core Tasks:

Pro Matcher: At the Address frame, calculate the ratio of the user's arm length to torso length. Find the Pro in the database with the closest ratio.

2D Rules Engine: Using standard Trigonometry (NumPy), calculate specific 2D angles based on the camera view. (e.g., Lead arm angle at Top of Backswing, Lateral hip sway in pixels/normalized units for Face-On). Do not use Z-axis depth.

Feedback Mapping: If a calculated angle fails a hardcoded threshold (e.g., < 160 degrees), append the corresponding COACHING_DB dictionary entry (Issue, Impact, Drill) to the user's report.

👀 Visual Demo Step: Extract the "Top of Backswing" frame. Use OpenCV to draw the measured angle (lines connecting shoulder-elbow-wrist) directly on the image with the calculated degree value. Ensure the math matches visual reality.

🤖 Antigravity Prompt: "In our notebook, implement a 'Pro Matcher' that compares the user's arm-to-torso length ratio at Address to a database of Pros. Second, build a pure 2D Rules Engine using NumPy. Calculate Face-On metrics like Lead Arm Flex (angle between shoulder, elbow, wrist) at the Top of Backswing. Create a COACHING_DB dictionary with text advice. If the user's arm angle is < 160, trigger the 'Towel Drill' text. Once verified, wrap this into src/coaching_engine.py."

📺 Day 5: Side-by-Side Visualization & Reporting

Objective: Stitch the final synchronized .mp4 video and export the plain-text .md coaching report.

File Workflow: Experiment with np.hstack and OpenCV drawing functions in notebooks/data_exploration.ipynb. Once perfect, save to src/visual_stitcher.py.

Inputs: User video, Matched Pro video, both sets of milestone indices (U_idx, P_idx), and the generated coaching text.

Outputs: swing_analysis.mp4 and coaching_report.md exported to output/.

Core Tasks:

Render a side-by-side video. To align the swings, map the timelines relative to the milestone indices rather than raw FPS.

Draw the MediaPipe skeletal wireframe on both videos.

Write the 2D metrics onto the video margins using OpenCV putText.

Compile the feedback dictionary into a cleanly formatted Markdown file.

👀 Visual Demo Step: Play the final exported video. Both players should hit the golf ball at the exact same frame, regardless of how long their pre-shot routines took.

🤖 Antigravity Prompt: "In the notebook, build an integration to take a user video and a matched Pro video. Using the two lists of milestone frame indices, temporally align the videos so the milestones occur simultaneously (using frame interpolation or duplicating frames). Draw the skeletal lines on both. Stitch them side-by-side using np.hstack and export as an MP4. Output the coaching feedback to coaching_report.md. Once the layout looks great, save to src/visual_stitcher.py."

🚀 Day 6: The End-to-End (E2E) Inference Pipeline

Objective: Wire all the individual scripts together into a single, unified pipeline. A user should be able to run one command in the terminal on a brand new video and get their final outputs automatically.

File Workflow: No notebook today! You will create analyze_swing.py at the root of the project to import everything from src/.

Inputs: An unseen video file path (e.g., my_swing.mp4) and camera perspective (--view face_on).

Outputs: Fully generated MP4 and Report in the output/ directory.

Core Tasks: Create an analyze_swing.py file at the root of your project that chains your previous modules together like a factory assembly line.

Call data_processor.py to extract and smooth the new video's coordinates.

Call dataset_builder.py logic to apply the sliding windows to the new data.

Load the XGBoost model and call predict_swing_milestones() to find the 8 key frames.

Pass the coordinate data at those 8 frames to coaching_engine.py to get the Pro Match and the textual feedback.

Pass everything to visual_stitcher.py to compile the final assets.

🤖 Antigravity Prompt: "Write a command-line script named analyze_swing.py using Python's argparse. It should accept a --video_path and --view. The script needs to import and sequentially run our entire pipeline from the src/ folder: 1. data_processor.process_video(), 2. sliding window application, 3. train_classifier.predict_swing_milestones() using our saved JSON model, 4. coaching_engine.generate_feedback(), and finally 5. visual_stitcher.create_outputs(). Ensure it handles paths correctly and logs its progress to the console at each step."