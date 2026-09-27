# Competitor solutions to SIH 2026 PS 26174 ("AI Human Activity Recognition for On-board BAS Experiments") published outside GitHub

Research date: 2026-09-27. YouTube upload dates come from each video's page metadata (`uploadDate`, US Pacific time). Transcripts are YouTube auto-captions (ASR) pulled with yt-dlp, so product names are sometimes garbled (for example "Crea" is KRIYA, "Parihar" is PRAHARI, "Astrahar" is Astraea).

## Q1. Non-GitHub public descriptions and demos (YouTube, LinkedIn, blogs, Devfolio/Unstop, Reddit, Instagram/X, college news)

### Takeaway
YouTube is where nearly all non-GitHub competitor material lives: 10 videos from at least 9 teams, all uploaded 4–26 Sep 2026 as SIH idea-submission demo videos. We found no Medium, Dev.to, Hashnode, Devfolio, Unstop or Devpost pages for this PS. LinkedIn has one relevant signal: an unverified "Top 5" claim from a JECRC team. Almost every team converges on the same design: MediaPipe or YOLO perception, a rule-based FSM for step order, offline TTS voice alerts, a web or desktop dashboard, and JSON/CSV logs. Several teams also demo more than one experiment protocol.

### Cited Findings

**YouTube demo videos tied explicitly to PS 26174 (listed newest first)**

1. **KRIYA** ("Kinematic Recognition of In-orbit Yatri Activity"), SIH team ID 134569. Channel "sneha". Uploaded 2026-09-26. 6:03 long, 11 views. — [YouTube](https://www.youtube.com/watch?v=_7xYEvzF8U4)
   - Stack: Python, OpenCV, MediaPipe Pose (33 landmarks), YOLOv8s for objects, and a browser UI.
   - v1 uses rules. They say v2 uses "a pre-trained AI model".
   - It recognises 21 generic crew activities (for example sitting) and keeps an activity timeline.
   - They claim a body-local skeleton reference, so detection is "irrespective of orientation or distance", which they present as their microgravity answer.
   - The 5-step demo uses a bottle, a cup and a phone. "Bottle in wrist" is detected at 88% confidence. A voice warning fires on a skipped step ("Warning, step skipped. Expected: place bottle near cup").
   - Their "later version" runs on real microgravity footage of a scientist doing an 8-step "crew nutrition protocol" (meal kit, condiment pouches, tortilla). It detects a missing "record pre-application mass" step and says "Caution… Return to step four".
   - They pitch it as a general surveillance system, with uses in traffic, manufacturing and cooking.
   - This is the only other team we found that demos on real astronaut or microgravity footage, which is the same move as ACTUS's astronaut clip.
2. **ORION** ("Offline Recognition and Instruction for Operational Navigation"). Team WayFinders, team ID 169987. Uploaded 2026-09-25. 6:43 long. — [YouTube](https://www.youtube.com/watch?v=ih-ViXyOVpQ)
   - The demo turns off the internet first, then runs an offline desktop app.
   - A phone is used as the webcam. Video-file input also works, and one input shown is "an AI-generated video".
   - Three protocols are loaded:
     - Payload transfer: red box to zone A and yellow box to zone B. Each needs a 1 s hold, then "close the rack".
     - Plant monitoring and watering: uses fiducial markers 10, 11 and 12 on the tray, plant and bottle, plus manual confirmations.
     - A third protocol that is not shown.
   - Wrong zone or wrong order gives "Wrong sequence/zone. Remove the misplaced object".
   - Sessions are saved as CSV and JSON with the video.
   - An offline voice assistant answers "what should I do next?".
   - The UI has dark and light themes.
   - Recognition looks zone- and marker-based, not pose-based.
   - Possibly linked GitHub (name match only, not verified): [amitraghvan/Orion](https://github.com/amitraghvan/Orion).
3. **Unnamed prototype**, channel "Deadcoding", titled only "SIH26174". Uploaded 2026-09-25. 3:47 long. — [YouTube](https://www.youtube.com/watch?v=PzQ65QNYl0A)
   - Its "synthetic experiment dataset" has several experiments "taken from IMEX (Indian Microgravity Experiments)".
   - Container-orientation experiment: tilt, shake, return upright, put down. Each step gets a "Correct. Next step." voice prompt.
   - Seed-sorting experiment: red seed into container A, blue seed into container B.
   - Claims: pose estimation, object detection and hand-object interaction; a log space marking steps success or wrong; fully offline including the voice agent; streaming to another IP; local video and text logs.
   - They say orientation-agnostic 3D HMR "can be incorporated". That wording suggests HMR is not yet implemented.
4. **TARA** ("Task-Aware Astronaut Recognition Assistant"). Team Shiriusu, team ID 166918. Members: Harsh Audichya, Drashya Sahu, Krishna Kankane, Giriraj Sharma, Nancy Dangi, Jayant Sharma. Uploaded 2026-09-24. 3:05 long. — [YouTube](https://www.youtube.com/watch?v=3e-GRiH3uxA)
   - Pipeline: "WATCH → UNDERSTAND → VERIFY → ALERT".
   - The description claims object detection, pose estimation, multi-object tracking, HOI, temporal activity recognition, automated dataset generation and edge processing.
   - The demo shows:
     - a "multi-object experiment" module;
     - voice prompts "Step verified" and "Object alert";
     - a recorded video with skeleton and bounding-box overlays;
     - an event log with confidence scores;
     - a timestamped "mission audit report" (target step, detected activity, time) that can be downloaded;
     - JSON metadata export.
5. **Telemetrix**: "Vision-Based State Tracking for On-Board BAS Experiment Sequences". Uploaded 2026-09-22. 1:40 long, 3 views. — [YouTube](https://www.youtube.com/watch?v=z2HCzZ9d1JY)
   - Approach: object detection plus an FSM, with a detector trained on "our own captured dataset".
   - They built ISRO's sample setup as "an outer box with a **pink** and a yellow box inside". The sequence is idle, box opened, pink box removed, yellow box removed.
   - Out-of-order voice alert: "Yellow box was removed before pink box".
   - Pitch claims: pose, object and hand tracking relative to the payload rack; no continuous video stream to Earth; a lightweight timestamped log; video streamed only when required.
   - They present the FSM layer as reusable across experiments.
6. **PRAHARI**. Team TARS-X, team ID 131893. Channel "Nihal Savaliya". Uploaded 2026-09-20. 3:16 long. — [YouTube](https://www.youtube.com/watch?v=jV00Wzg49_k)
   - Six stages: fixed payload camera, then detection of objects, pose, contact and actions, then a state engine (right action, right object, right order), then voice or GUI guidance, then event recording.
   - Stated innovations:
     - microgravity-friendly 3D tracking;
     - unknown-action detection;
     - physical outcome verification;
     - confidence-based alerts;
     - sequence validation.
   - Stack: Python, PyTorch, OpenCV, ONNX Runtime, C++, SQLite, TensorRT with INT8.
   - Protocols are defined in YAML and checked by the FSM.
   - Inputs: webcam, IP camera or phone camera.
   - Planned mitigations: synthetic data, multi-camera fusion, confidence gating and event-based transmission.
   - The demo shows a GitHub clone-and-run step, but the repo URL is not visible in the transcript.
7. **Kartikey Upadhyay**: "AI Human Activity Recognition for On-board BAS Experiments Demo video". Uploaded 2026-09-12. 2:12 long. — [YouTube](https://www.youtube.com/watch?v=aLQtu_kuAVA)
   - The description only says "SIh demo video". There are no captions, so the video is probably silent or music-only.
   - Approach not characterized.
8. **Astraea** (ASR also renders it "Astrahar"). Channel "Subham Dey", titled "SIH26174". Uploaded 2026-09-08. 3:35 long. — [YouTube](https://www.youtube.com/watch?v=skf8u9PODHc)
   - Pitched as a general "security and monitoring system": activity timeline, analytics, and incident records for abnormal behaviour.
   - They admit they have "not yet received the complete real-time data set".
   - Weak fit to the PS: no step sequencing is shown.
9. **EduCare**: "SIH26174 Working prototype video explanation". Uploaded 2026-09-07. 1:12 long. — [YouTube](https://www.youtube.com/watch?v=ISZf7j3_qHM)
   - No descriptive text and no captions. Approach not characterized.
10. **Techtitans**: "AI Human Activity Recognition for On-board BAS Experiments | SIH 2026 | PS 26174". Uploaded 2026-09-04. 2:57 long, 383 views. Members: Shaurya Goel, Ryan Bhatnagar, Vansh, Angello Iwin, Payal. — [YouTube](https://www.youtube.com/watch?v=8F_P0OdRI_w)
    - Proposed stack: pose estimation, 3D/relative skeleton, temporal activity recognition, **wearable IMU fusion**, "spatial action graphs", rule-based protocol compliance and a live dashboard.
    - Tech: Python, OpenCV, MediaPipe, PyTorch, FastAPI, PostgreSQL.
    - Note: the page metadata loaded, but yt-dlp later returned "This video is not available" on 2026-09-27. The video may have been made private or removed.

**Adjacent, not this PS**
- **MAITRI AI**, Team MindLeap. SIH **2025**, ISRO astronaut well-being PS. Uploaded 2025-10-12. — [YouTube](https://www.youtube.com/watch?v=jamkpg3mMcg)
  - Multimodal edge AI (voice, face, biometrics) for BAS crew. Stack: TFLite, PyTorch, React, Jetson Orin.
  - Shows that ISRO ran a BAS-crew AI PS in SIH 2025 too, but it was a different problem.

**LinkedIn**
- A search-engine snippet tied to JECRC University's Arpit Agrawal's LinkedIn says "Team Innovators secured a Top 5 position at Smart India Hackathon 2026 for ISRO's problem statement SIH26174… Their solution, ASTRA…". — [LinkedIn (Arpit Agrawal, JECRC)](https://www.linkedin.com/in/arpitjecrc/)
  - Fetching the profile did not show this text.
  - The same profile does mention "JECRC HackQuest 9.0 × Smart India Hackathon '26, 5–6 September 2026" and "550+ young minds at JECRC went through an important exercise before Smart India Hackathon 2026". — [LinkedIn](https://in.linkedin.com/in/arpitjecrc)
- A GitHub repo named "ASTRA-HAR (Autonomous Space Payload HAR & Sequence Validation System)" claims 16 BAS action classes, rolling frame buffers with "temporal 3D models", ONNX object detectors, an FSM with timeout hazards, local TTS and ≥25 FPS. The search engine linked it to [Monishwarann/SIH](https://github.com/Monishwarann/SIH) (GitHub, for the GitHub researchers). We could not confirm it belongs to Team Innovators at JECRC.

**Other platforms**
- Searches restricted to Medium, Dev.to, Hashnode, Devfolio, Unstop, Devpost, Instagram and X found nothing about PS 26174. — [search via WebSearch; no result URLs to cite]
- The only SIH 2026 Instagram hit was an unrelated consumer-affairs PS post (account karthik_n_iyerr, 2026-08-24). — [Instagram](https://www.instagram.com/p/Dcaqzc1DtwP/)
- Reddit could not be searched: the search tool blocks reddit.com.

**GitHub links surfaced along the way (for the GitHub researchers)**
- hamsadevang849-sys/AI-Human-Activity-Recognition-for-On-board-BAS-Experiments
- Soubhagya-Kabiraj/HOMOSENSYS-AI-Assistant and rudra00434/HOMOSENSYS-AI-Assistant
- haasitha1603/NAVIKSHA
- atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174
- Dynamo-SJoshi/VisionX_SIHproject
- yash2744/AI---Human---Activity---Recognition-for-Bas-Experiment-SIH-2026-
- Vynex-Labs/AI-Human-Activity-Recognition-for-On-Board-BAS-Experiments
- Pranvkumar/AI-HAR-For-BAS
- Thiru-Lokesh-24/cv_module
- Omkar96-18/SIH26174_BAS_AgentByXathonerTeam
- Monishwarann/SIH
- amitraghvan/Orion
- ishwaryajuttu-gif/SIH_2026
- yashkhochare-79/HAR_satellite
- arnab9957/realtime-work-detection and Ocean-101/realtime-work-detection ("BAS-03/BAS-04 science modules")
- h1e2l3l4o5/ORBIT-HAR and Ozair-aly/orbit-har

— [WebSearch results listing these repos](https://github.com/topics/sih?o=desc&s=updated)

### Inferences
- Teams have converged on the same design: detector and pose model, then a hand-written FSM, then offline TTS, a dashboard and JSON/CSV logs. Tables stakes now include the FSM, voice alerts, local logs and an offline demo, so none of these set a team apart.
- Few competitors actually use a learned temporal action model. Most visible demos run on object or zone heuristics. TARA and Techtitans claim temporal recognition, and ASTRA-HAR claims 3D temporal models, but none of the videos show evidence of one.
- On orientation-agnostic HMR, what we saw was stated intent, not demonstrated work:
  - Deadcoding says it "can incorporate" HMR.
  - KRIYA uses a body-local skeleton reference.
  - PRAHARI and Telemetrix list HMR-like claims in their pitch.
  - No video shows real 3D HMR (SMPL or HMR2.0) working.
- KRIYA's second version runs on real microgravity footage of a food-prep protocol. That is the closest competitor to ACTUS's "real astronaut clip end-to-end" demo half, so ACTUS's framing of the skeleton as the bridge from Earth to orbit needs to go beyond simply showing a microgravity clip.
- Several teams show multiple protocols (ORION 3, Deadcoding 2, KRIYA 2). Judges may therefore expect a generality or protocol-authoring story.

### Gaps
- Only titles and metadata were available for the EduCare and Kartikey Upadhyay videos (no captions), and only the description for Techtitans (unavailable by 27 Sep).
- LinkedIn posts cannot be fetched directly, and the "Top 5" text was seen only in a search snippet.
- Instagram, X and Reddit are poorly indexed or blocked. There may be team reels or posts we could not find.
- We did not look at video frames or pixels, so UI details come only from narration.

## Q2. Hugging Face, Kaggle, Roboflow Universe, GitLab/Bitbucket artifacts

### Takeaway
We found no public Hugging Face models, datasets or Spaces, Kaggle datasets or notebooks, or Roboflow Universe projects tied to PS 26174. Teams appear to keep their custom red/yellow-box datasets inside their GitHub repos or unpublished.

### Cited Findings
- **Hugging Face:** the Hub API was searched for models, datasets and Spaces with "SIH26174", "sih-26174", "sih_26174", "26174", "astronaut-activity", "bas-experiment", "red-yellow-box" and "antariksh". There were zero relevant hits.
  - Nearest hit: `jialeh23/red-cube-yellow-box-v2`, a small video dataset created 2026-03-29, with no card. It predates the SIH 2026 PS launch and is probably a robotics dataset, not related. — [HF API](https://huggingface.co/api/datasets/jialeh23/red-cube-yellow-box-v2)
  - SIH 2026 Spaces do exist for other PSs, so teams do use HF for SIH work, just not for this PS. Examples: "SIF Precursor Triage — SIH 2026 PS 26165" ([HF Space](https://huggingface.co/spaces/beakthoven/echarikai)) and "PS-07-Bharatiya-Antariksh-Hackathon-2026" (Lalitjindal-code). — [HF Space](https://huggingface.co/spaces/beakthoven/echarikai)
- **Kaggle:** the dataset API search for "SIH26174" returned an empty list `[]`. Searching "astronaut activity" returned only unrelated items (for example the TidyTuesday astronaut database). — [Kaggle API](https://www.kaggle.com/api/v1/datasets/list?search=SIH26174)
  - Kaggle does host SIH meta-datasets: an SIH 2025+ PS archive and SIH 2024 PS with winners. — [Kaggle](https://www.kaggle.com/datasets/theprtsh/sih-2025-problem-dataset); [Kaggle](https://www.kaggle.com/datasets/adharshinikumar/sih-2024-ps-with-winning-teams-and-solutions)
- **Roboflow Universe:** web search found no red/yellow-box experiment or astronaut-experiment project. The Universe search page is rendered in the browser, so curl returned no project links. — [Roboflow Universe search](https://universe.roboflow.com/search)
- **GitLab and Bitbucket:** no hits in search.

### Inferences
- The datasets are proprietary per team, as the PS requires ("build a custom, highly focused local dataset (even just using a webcam)"). ACTUS could stand out by publishing a small labelled clip set or an HF Space demo, since nobody else has on this PS.

### Gaps
- We did not browse Roboflow Universe's in-browser search, so private or unlisted Roboflow workspaces cannot be ruled out.
- Kaggle notebooks (as opposed to datasets) were not queried through the API.

## Q3. What teams claimed in idea submissions and PPTs; shortlisted or winning teams

### Takeaway
No idea-submission PPTs or PDFs for this PS were found outside GitHub. What teams claim comes from their YouTube pitch videos (the SIH idea template includes a demo-video link). No national shortlist or winners exist yet: idea submission closes 30 Sep 2026. The one "Top 5" claim (Team Innovators, ASTRA, JECRC) is almost certainly an internal college-hackathon result.

### Cited Findings
- PRAHARI (TARS-X) follows the SIH idea-template structure (problem, approach, feasibility and viability, technical approach, impact, research papers). Its claims:
  - Feasibility: commodity camera, consumer GPUs, open source.
  - Five-stage plan: data collection, model training, state engineering, guidance integration, edge optimization (ONNX, TensorRT, INT8).
  - Impact: fewer experiment errors, lower bandwidth, less crew workload, and reuse across step-based experiments.
  - Source: [YouTube, 2026-09-20](https://www.youtube.com/watch?v=jV00Wzg49_k)
- Techtitans' stated novelty is multimodal fusion of video pose with wearable IMU, plus "spatial action graphs". Their backend is FastAPI with PostgreSQL. — [YouTube, 2026-09-04](https://www.youtube.com/watch?v=8F_P0OdRI_w)
- TARA claims "Automated Dataset Generation" and "Experiment Metadata Generation" as features. — [YouTube, 2026-09-24](https://www.youtube.com/watch?v=3e-GRiH3uxA)
- KRIYA claims market reach well beyond space: traffic, manufacturing, robot and worker training, and cooking. — [YouTube, 2026-09-26](https://www.youtube.com/watch?v=_7xYEvzF8U4)
- Telemetrix claims its FSM layer is "a general sequencing framework rather than a single-purpose detector". — [YouTube, 2026-09-22](https://www.youtube.com/watch?v=z2HCzZ9d1JY)
- The "Top 5 at SIH 2026 for SIH26174" claim by Team Innovators (solution ASTRA) comes from a search snippet attributed to a JECRC University LinkedIn. The same source references JECRC's internal "HackQuest 9.0 × SIH '26" on 5–6 Sep 2026. — [LinkedIn](https://www.linkedin.com/in/arpitjecrc/)
- Team IDs visible in the videos are 131893 (TARS-X), 134569 (KRIYA), 166918 (Shiriusu) and 169987 (WayFinders). — [YouTube](https://www.youtube.com/watch?v=jV00Wzg49_k); [YouTube](https://www.youtube.com/watch?v=_7xYEvzF8U4); [YouTube](https://www.youtube.com/watch?v=3e-GRiH3uxA); [YouTube](https://www.youtube.com/watch?v=ih-ViXyOVpQ)

### Inferences
- National screening results come out in October 2026 and finalists are announced in November. A "Top 5" dated September can therefore only be an internal-hackathon ranking, probably at JECRC's HackQuest 9.0.
- Institutions are mostly not named in the videos. Apart from JECRC (Jaipur), the teams' institutions are unknown.

### Gaps
- We found no idea PPTs on Scribd, SlideShare or Google Drive for this PS.
- The institutions behind TARA, ORION, PRAHARI, KRIYA, Telemetrix, Deadcoding and Techtitans are not stated.
- No official shortlist or winner list exists yet.

## Q4. Official SIH context: edition, IDs, submissions, timeline

### Takeaway
PS 26174 belongs to **SIH 2026** only: ISRO, Software category, Space Technology theme, one of 11 ISRO PSs out of 233 in total. On 2026-09-27 the official portal showed **52 of 500** idea slots used, with submissions closing **30 Sep 2026**. We found no evidence of a 16174 or 25174 version of this PS in earlier SIH editions. The Grand Finale is a 36-hour hackathon in December 2026.

### Cited Findings
- Official portal row: "Software SIH26174 52/500 Space Technology 30 September 2026", retrieved 2026-09-27. — [sih.gov.in PS list](https://sih.gov.in/sih2026PS)
- The official dataset text is cut off at the source ("…two smaller boxes of color red and yello"). Teams therefore saw an incomplete sample-experiment spec.
  - Telemetrix built "pink and yellow" boxes, which may be a side effect of that ambiguity. — [sih.gov.in](https://sih.gov.in/sih2026PS); [YouTube](https://www.youtube.com/watch?v=z2HCzZ9d1JY)
- A third-party mirror showed 46 submissions at an earlier snapshot. — [zaidsayyed.in](https://zaidsayyed.in/tools/sih-problem-statements/sih26174)
- The SIH 2026 totals of 233 PSs (176 software, 57 hardware) from 33 organisations, including 11 from ISRO, come from a third-party browser with snapshot date 2026-09-03. — [sih2026.vuce.in](https://sih2026.vuce.in/)
- SIH 2026 was launched on 21 Aug 2026 by AICTE Chairman Prof. Yogesh Singh. The idea and nomination deadline is 30 Sep 2026, and the Grand Finale is 36 hours at nodal centres in December 2026. — [Reskilll](https://reskilll.com/blogs/smart-india-hackathon-2026-launched-timeline-registration-how-to-participate/); [FirstVidya](https://firstvidya.com/sih-2026-guide/)
- The timeline by month (article dated 2026-09-23):

  | Phase | Month(s) |
  |---|---|
  | Team nomination | Aug–Sep |
  | Idea screening | Sep–Oct |
  | Screening results | Oct |
  | Finalist communication, mentoring, final shortlist | Nov |
  | Grand Finale | Dec 2026 |

  — [TechPathDaily](https://techpathdaily.com/sih-2026-official-timeline-is-here/)
- An SIH 2025 PS browser shows no ISRO human-activity-recognition PS. — [sih-nav.vercel.app](https://sih-nav.vercel.app/)
- In SIH 2025, ISRO's BAS-related PS was astronaut well-being (MAITRI). — [YouTube](https://www.youtube.com/watch?v=jamkpg3mMcg)
- Separate from SIH, ISRO's Bharatiya Antariksh Hackathon 2026 (Hack2skill) launched 10 Jun 2026 with 15 PSs and held its finale on 6–7 Aug 2026. It is a different competition. — [ISRO](https://www.isro.gov.in/Bharatiya_Antariksh_Hackathon_2026_Grand_Finale.html)

### Inferences
- 52 submissions against 500 slots means low direct competition at the national screening stage. Several teams per college go through internal hackathons first, as JECRC's did.
- The IDs "16174" and "25174" appear to be mis-remembered IDs, not earlier editions of this PS.

### Gaps
- The Bharatiya Antariksh Hackathon 2026 PS list was not checked for a HAR-like PS.
- Exact screening and finale dates for SIH 2026 have not been published.
- Final idea-submission counts will only be known after 30 Sep.
