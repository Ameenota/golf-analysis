# Product Features & Requirements: Golf Swing Analyzer

This document describes the user-facing product features, biomechanical rules, and user experience requirements for the Golf Swing Analyzer.

---

## 1. Core Product Features

### A. Smart Video Validation (Gatekeeper)
* **User Experience**: The user uploads a video of their swing. The app instantly verifies if the video contains a golf swing.
* **Behavior**: If the video is invalid (e.g. empty room, bystander walking, different sport), the system gracefully rejects it with an explanatory message rather than failing silently or returning garbled results.

### B. Milestone Detection
* **User Experience**: The system automatically scans the video and identifies the exact frame indices of the **8 Critical Milestones** of the golf swing:
  1. **Address (F1)**: Start of the swing setup.
  2. **Toe-Up (F2)**: Shaft is parallel to the ground during take-away.
  3. **Top of Backswing (F3)**: Club changes direction at the peak.
  4. **Downswing (F4)**: Shaft is parallel to the ground during downswing.
  5. **Impact (F5)**: The exact frame the club strikes the ball.
  6. **Release (F6)**: Shaft is parallel to the ground post-impact.
  7. **Follow-Through (F7)**: Club is vertical post-impact.
  8. **Finish (F8)**: Completion of the swing.

### C. Pro Matchmaker
* **User Experience**: Matches the user's physical build to a professional golfer in the database.
* **Mechanism**: Compares the user's arm-to-torso ratio at the Address frame to the professional database to find the closest skeletal match. This ensures comparison metrics are realistic and biomechanically relevant.

### D. Biomechanical Analysis
* **User Experience**: Calculates key angles and posture metrics.
* **Rules & Metrics**:
  * **Lead Arm Flex**: Measures lead arm straightness at the Top of Backswing. If angle $< 160^\circ$, it flags a bent arm issue.
  * **Spine Tilt**: Measures spine angle relative to the vertical axis at Address and verifies if the angle is maintained at the Top of Backswing.
  * **Lateral Hip Sway**: Tracks the movement of the hip centroid relative to the setup position to flag sliding or swaying.

### E. Interactive Coaching Report
* **User Experience**: Generates a downloadable Markdown report highlighting:
  * Measured biomechanical metrics vs. optimal ranges.
  * Triggered issues (e.g. "Early Release/Casting", "Lead Arm Bend").
  * Actionable golf drills (e.g., "Towel Drill" under the lead armpit to prevent bending).

### F. Synchronized Side-by-Side Visualizer
* **User Experience**: Generates an output MP4 video displaying the user's swing side-by-side with their matched professional golfer.
* **Synchronization**: The two swings are temporally warped/aligned using the 8 milestone frames. Both swings will hit Address, Top of Backswing, and Impact at the *exact same time* in the video playback, regardless of variations in their natural swing tempo.
* **Overlays**: Draws the MediaPipe skeletal wireframe and highlights the angle measurements on-screen at each milestone.

---

## 2. Milestone Definition Reference Table

| Milestone ID | Event Name | Biomechanical Significance | Key Detection Features |
| :---: | :--- | :--- | :--- |
| **1** | Address | Setup alignment and baseline torso scale. | Hands low, body stationary. |
| **2** | Toe-Up | Takeaway path validation. | Club shaft parallel to ground. |
| **3** | Top of Backswing | Determines maximum backswing coil. | Max shoulder rotation, hips loaded. |
| **4** | Downswing | Transition/lag angle assessment. | Club shaft parallel to ground. |
| **5** | Impact | Crucial moment of truth. | Direct contact with ball. |
| **6** | Release | Release of wrist lag. | Club shaft parallel to ground. |
| **7** | Follow-Through | Post-impact extension check. | Arms fully extended towards target. |
| **8** | Finish | Balance and weight transfer check. | Torso facing target, weight on lead foot. |
