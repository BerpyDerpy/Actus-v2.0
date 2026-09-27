# GitHub competitor repos for SIH 26174 found by descriptive/topical keyword search

Method note: `gh` CLI is not installed on this machine, so all searches used the unauthenticated GitHub REST search API (`/search/repositories`, name/description by default plus an `in:name,description,readme,topics` variant). Candidates were shallow-cloned and their source was grepped and read to check what the READMEs claim. The search date is 2026-09-27. Star, fork, created and pushed figures come from the search API on that date. Every repo URL below is a primary source, the repo itself. File-level links use `blob/HEAD/<path>`. A shared scratchpad with the ID-based researcher caused one query batch (batch b) to run through their script, and those results were merged in. Overlap with the ID-based researcher's list is expected: most competitors name "SIH26174" in their README even when their description doesn't.

## Q1: Which repos match the descriptive/topical searches (astronaut HAR, experiment step recognition, BAS experiment, Bharatiya Antariksh Station, microgravity, payload rack, topic:sih2026, etc.)?

### Takeaway
Descriptive searches turned up about 45 repos created from 2026-08-23 to 2026-09-27 that target PS 26174. The productive terms were "astronaut activity recognition", "astronaut HAR", "BAS experiment(s)", "Bharatiya Antariksh Station", "experiment copilot/assistant", "on-board experiments", "ISRO HAR", "human mesh recovery astronaut" and "microgravity HMR". These terms returned nothing relevant: "red box yellow box", "SOP compliance computer vision", "lab protocol monitoring", "voice alert next step", "action sequence FSM", "payload rack" and "hand object interaction experiment". GitHub topics (`topic:sih2026`, `topic:sih-2026`, `topic:smart-india-hackathon`, `topic:isro`) surfaced other SIH 2026 problem statements (26028, 26089, 26145, 26153, 26166, 26167, 26170, 26171) but no 26174 repo, so competitors on this problem almost never set topics. Every relevant repo is small: the maximum is 5 stars and 6 forks.

### Cited Findings
**Search-term yield (API `total_count`, name+description scope)**
- "astronaut activity recognition" returned 14 repos, 12 of them SIH 26174 entries created 2026-08-24 to 2026-09-25 (e.g. [Krishna105-ui/Astronaut-Activity-Recognition-System](https://github.com/Krishna105-ui/Astronaut-Activity-Recognition-System), [StarCodes03/TARA-Astronaut-Assistant](https://github.com/StarCodes03/TARA-Astronaut-Assistant), [amitraghvan/Orion](https://github.com/amitraghvan/Orion)). The two outliers are [Ariaeimehr/Zero-G-Space-HAR](https://github.com/Ariaeimehr/Zero-G-Space-HAR) (a generic federated IMU project about exercise countermeasures, not SIH) and [deepakbareth01/astra](https://github.com/deepakbareth01/astra), which is empty (size 0).
- "BAS experiments activity" and "BAS experiment" surfaced [atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174), [MakInShort/Hangla-Coders-SIH2026174](https://github.com/MakInShort/Hangla-Coders-SIH2026174), [rudra00434/HOMOSENSYS-AI-Assistant](https://github.com/rudra00434/HOMOSENSYS-AI-Assistant), [haasitha1603/NAVIKSHA](https://github.com/haasitha1603/NAVIKSHA), [ArrushiTripathi2429/Space-Sight2](https://github.com/ArrushiTripathi2429/Space-Sight2), [Nimbus3001/KRAMA](https://github.com/Nimbus3001/KRAMA), [PriyankShingala/Stellar](https://github.com/PriyankShingala/Stellar), [Vynex-Labs/…BAS-Experiments](https://github.com/Vynex-Labs/AI-Human-Activity-Recognition-for-On-Board-BAS-Experiments), [xarjunpatil/SIH26174-…](https://github.com/xarjunpatil/SIH26174-AI-Human-Activity-Recognition-for-On-board-BAS-Experiments) and [SAMEERSACHDEVA2817/…](https://github.com/SAMEERSACHDEVA2817/AI-Human-Activity-Recognition-for-On-board-BAS-Experiments).
- "Bharatiya Antariksh Station" and "astronaut assistant" surfaced [arnab9957/realtime-work-detection](https://github.com/arnab9957/realtime-work-detection). Its name gives no hint of SIH; only the description ("deterministically validate procedural experiments") matched. The same searches also found [Rahulrwt99/bas-experiment-assistant](https://github.com/Rahulrwt99/bas-experiment-assistant), [Kshitij0275/BAS-Mission-Control](https://github.com/Kshitij0275/BAS-Mission-Control) (0 files), [Pranvkumar/AI-HAR-For-BAS](https://github.com/Pranvkumar/AI-HAR-For-BAS), [sohamjachak003/AI-BAS-Assistant](https://github.com/sohamjachak003/AI-BAS-Assistant) (plus its `_2` copy and a copy at [baradaditya17-creator/AI-BAS-Assistant](https://github.com/baradaditya17-creator/AI-BAS-Assistant)) and [Aryasurya12/AstroFlow-AI](https://github.com/Aryasurya12/AstroFlow-AI).
- "experiment sequence validation", "orientation agnostic" and "human mesh recovery" surfaced [Shre-Sat/BASGUARD](https://github.com/Shre-Sat/BASGUARD). "microgravity HMR" surfaced arnab9957. "ISRO HAR" surfaced [notaditya28/isro_har_sih_project](https://github.com/notaditya28/isro_har_sih_project) and [krishnakankane/isro-bas-har](https://github.com/krishnakankane/isro-bas-har). "ISRO activity recognition" surfaced [gsaivenkateshkumar/ASTRA-HAR-EdgeAI](https://github.com/gsaivenkateshkumar/ASTRA-HAR-EdgeAI).
- "astronaut HAR" and "astronaut human activity recognition" also surfaced [JaiPrakashVI/AI-Human-Activity-Recognition-Space](https://github.com/JaiPrakashVI/AI-Human-Activity-Recognition-Space), [Sheen11-ing/AI-based-Human-Activity-Recognition-BAS-](https://github.com/Sheen11-ing/AI-based-Human-Activity-Recognition-BAS-), [Himashwetha777/SIH26174-Astronaut-HAR-System](https://github.com/Himashwetha777/SIH26174-Astronaut-HAR-System) (with byte-identical siblings `ASTRA-HAR-System` and an empty `ASTRA-HAR-Deployment`), [ThouheedAhmed/DRISHTI-Edge-AI](https://github.com/ThouheedAhmed/DRISHTI-Edge-AI), [dnvrmhra/ASTRA-HAR](https://github.com/dnvrmhra/ASTRA-HAR), [Kartikey2006/Astra](https://github.com/Kartikey2006/Astra) and [hamsadevang849-sys/…BAS-Experiments](https://github.com/hamsadevang849-sys/AI-Human-Activity-Recognition-for-On-board-BAS-Experiments).
- README-scope and date-filtered variants ("astronaut experiment created:>2026-07-01", "experiment copilot", "on-board experiments", "activity recognition created:>2026-08-01 …", "procedure validation", "experiment assistant", "astronaut pose") added these: [SyroSyrup98/ORBIT-HAR](https://github.com/SyroSyrup98/ORBIT-HAR), [kavyareddypallala-cell/BAS-Vision](https://github.com/kavyareddypallala-cell/BAS-Vision), [MrRouhin/Astronaut-AI-Assistance](https://github.com/MrRouhin/Astronaut-AI-Assistance), [antas06/exp-ass-bas](https://github.com/antas06/exp-ass-bas), [aayushjadhav05128-sudo/bas-experiment-copilot](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot), [samaksh-divs/bas-experiment-copilot-frontend](https://github.com/samaksh-divs/bas-experiment-copilot-frontend), [yamantaka-singh/BAS](https://github.com/yamantaka-singh/BAS), [LakshanyaaJ/ORBITA](https://github.com/LakshanyaaJ/ORBITA), [Ayush-sharma-mx/bas-ai-monitor](https://github.com/Ayush-sharma-mx/bas-ai-monitor), [mihikahetamsaria/SIH-Hackathon](https://github.com/mihikahetamsaria/SIH-Hackathon), [renukathombre1111/BAS-AI-Experiment-Assistant](https://github.com/renukathombre1111/BAS-AI-Experiment-Assistant), [saurav158204/vistaspace-site](https://github.com/saurav158204/vistaspace-site), [AjitKumar166/sih26-project](https://github.com/AjitKumar166/sih26-project) and [Dushy2006/AGASTYA-AI-BAS-Experiment-Assistant](https://github.com/Dushy2006/AGASTYA-AI-BAS-Experiment-Assistant) (created 2026-09-27, empty).
- The ID-based researcher's clone set also contains repos found by our topical queries that we did not re-profile: Omkar96-18/SIH26174_BAS_AgentByXathonerTeam, Shreyansh-DevHub/AI-HAR-_On-board_BAS_Experiment, dhanya-s-coder/sih26174, h1e2l3l4o5/ORBIT-HAR, swatikrishnamurthy24-dotcom/AI-ASSISTANCE-ISRO-, madhurgitbub/ISRO, nonugit/An-AI-based-HAR-system, chaitanyareddykarri/…BAS-, moiz2405/ASTRA-SEQ, sangsaptak0704/KriyaSense, k-a-v-i-n-0-0-2/hawk-bas, Dynamo-SJoshi/VisionX_SIHproject, Shrishti08raghav/Viksit-Coders-SIH2026 and yash2744/…SIH-2026- (surfaced by "on-board experiments", "HAR BAS"/"BAS HAR" in README scope, "experiment monitoring AI" and "astronaut experiment"). See that researcher's notes for their profiles.
- Terms that returned no SIH 26174 repo: "red box yellow box" (86 name/description hits, all unrelated), "SOP compliance computer vision" (only a 2025 COVID-SOP repo, [Maaheen11/…COVID-19-SOPs-Monitoring](https://github.com/Maaheen11/An-Automated-COVID-19-SOPs-Monitoring-and-Management-System)), "lab protocol monitoring", "procedure step verification", "step skipped alert", "out of sequence step", "voice alert next step", "action sequence FSM", "Gaganyaan activity" and "payload rack" (VCV Rack and drone payload racks only).
- "astronaut assistant AI" is dominated by about 25 repos for a different ISRO problem, MAITRI (astronaut psychological/physical well-being, SIH 2025). Examples: [GauravJha-Git/MAITRI-SIH-](https://github.com/GauravJha-Git/MAITRI-SIH-) and [RahulGodara14/MAITRI-…](https://github.com/RahulGodara14/MAITRI-Mission-Aware-Intelligent-Therapeutic-Response-Interface-). These were discarded as out of scope.
- The topic searches `topic:sih2026` (100+ results), `topic:sih-2026` (32) and `topic:isro created:>2026-08-01` (26) returned other problem statements, e.g. [thesyedammar/RailETA](https://github.com/thesyedammar/RailETA) (SIH26028), [darshiljaiswal2-afk/astrix-sih26](https://github.com/darshiljaiswal2-afk/astrix-sih26) (ISRO PS 26170) and [AtharvaSamant4/DrishtiGuard](https://github.com/AtharvaSamant4/DrishtiGuard) (SIH26171). None of these was a 26174 repo.
- [NoBugNinja/Smart-India-Hackathon-SIH-2026-Problem-Statements](https://github.com/NoBugNinja/Smart-India-Hackathon-SIH-2026-Problem-Statements) (30 stars, created 2026-08-22) publishes all SIH 2026 problem statements as CSV/JSON/MD. It could be a source for the full sample-experiment text, which our local copy truncates at "red and yello".
- None of the relevant repos is a GitHub fork. The most-forked are [rudra00434/HOMOSENSYS-AI-Assistant](https://github.com/rudra00434/HOMOSENSYS-AI-Assistant) (5★/6 forks, MIT) and [MakInShort/Hangla-Coders-SIH2026174](https://github.com/MakInShort/Hangla-Coders-SIH2026174) (4★/5 forks, MIT; its README clone URL points to an org repo, `Hangla-Coders/SIH2026174`) ([README](https://github.com/MakInShort/Hangla-Coders-SIH2026174/blob/HEAD/README.md)).
- Instead of forking generic HAR repos, some competitors copy research code into their repo: arnab9957 vendors NAVER's Multi-HMR 2 ([src/multihmr2/api.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/multihmr2/api.py), header "Copyright (c) 2026-present NAVER Corp.") and a "RelateAnything" open-vocabulary relation/scene-graph package ([relsgg/__init__.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/relsgg/__init__.py), loading `maelic/relsgg-vits16plus`).

**Near-duplicates and team clusters**
- [Pranvkumar/AI-HAR-For-BAS](https://github.com/Pranvkumar/AI-HAR-For-BAS) (created 2026-08-23) and [Krishna105-ui/Astronaut-Activity-Recognition-System](https://github.com/Krishna105-ui/Astronaut-Activity-Recognition-System) (created 2026-08-24; README plus `AEGIS-AI-HAR-v1.0.0.zip` with 89 files) both describe "AEGIS · AI-HAR": ArUco rack homography, Tier 0 zones, and a Tier 1 GRU over 32-frame windows exported to ONNX ([Krishna105 README](https://github.com/Krishna105-ui/Astronaut-Activity-Recognition-System/blob/HEAD/README.md); [Pranvkumar src/aegis](https://github.com/Pranvkumar/AI-HAR-For-BAS/tree/HEAD/src/aegis)).
- Two unrelated teams use the name ORBITA: [LakshanyaaJ/ORBITA](https://github.com/LakshanyaaJ/ORBITA) and [kavyareddypallala-cell/BAS-Vision](https://github.com/kavyareddypallala-cell/BAS-Vision) ("ORBITA — On-board Recognition & Behavioral Intelligence…"). The name ASTRA is also reused: dnvrmhra/ASTRA-HAR, gsaivenkateshkumar/ASTRA-HAR-EdgeAI, Kartikey2006/Astra and Himashwetha777/ASTRA-HAR-System.

### Inferences
- Most competitors put "SIH26174" only in the README. Descriptive-keyword search in name/description therefore finds them only when the description says "astronaut", "BAS" or "Bharatiya Antariksh". README-scope search (`in:readme`) is noisier but catches the rest.
- The low star counts (≤5) and the burst of creation dates (late August to late September 2026) fit internal-hackathon/college-round submissions. None of these repos has an audience yet, so stars say nothing about quality.

### Gaps
- GitHub code search (`/search/code`) requires authentication, so we could not search source for rare strings like "yellow_box", "payload_rack" or "HMR2" across all of GitHub. Only the cloned repos were grepped.
- The unauthenticated search API caps results at 100 per query and rate-limits at 10 requests/min. Some broad README-scope queries ("skipped step", "sequence validation") returned 100k+ hits and could not be exhaustively filtered.
- Private repos and repos created after 2026-09-27 are not covered.

## Q2: Technical profile of each relevant repo (models, HOI, step logic, voice, logs, streaming, GUI, hardware, dataset, HMR/microgravity, LLM, accuracy)

### Takeaway
Most entries follow the same template: YOLOv8/11n objects, MediaPipe Hands/Pose (or YOLO-pose) wrists, distance-threshold hand-object interaction, a JSON/YAML-driven deterministic FSM, pyttsx3/Piper/Web Speech voice, JSONL/SQLite logs, an MJPEG-over-HTTP stream, and a PySide6 or React dashboard. They differ in maturity. Only a handful have a trained, experiment-specific model with reported numbers:
- **Orion:** ST-GCN, 24.78% val accuracy.
- **gsaivenkateshkumar:** YOLO11n, mAP50 0.928.
- **notaditya28:** YOLO11n cube detector on a 362-clip, 4-view corpus.
- **JaiPrakashVI:** 1D-CNN on PAMAP2, not video.

Only **arnab9957** and **mihikahetamsaria** actually run a real 3D HMR network (Multi-HMR 2 and HMR2/4D-Humans respectively), and both run it offline/batch, not in the live loop. Several repos are README-only, UI-only or run a "demo mode" simulator.

### Cited Findings

**Tier A: substantial, working code, distinctive**

- **amitraghvan/Orion** (0★, created 2026-09-06, pushed 2026-09-20; about 34k code lines, including a C++20/pybind11 engine) ([repo](https://github.com/amitraghvan/Orion))
  - **Perception:** YOLO11n detection, YOLO11n-pose (17 COCO joints) and ByteTrack with Kalman. Hands are cropped geometrically from the wrist/elbow. HOI is a proximity state machine: APPROACH/TOUCH/MANIPULATE/RELEASE ([README](https://github.com/amitraghvan/Orion/blob/HEAD/README.md)).
  - **Step recognition:** an ST-GCN over 32-frame skeletal windows (stride 8). Shannon-entropy gating (H ≤ 1.40) plus debouncing stops state advances on single-frame spikes. An 11-state FSM detects SKIPPED, OUT_OF_SEQUENCE, WRONG_OBJECT, STEP_UNCERTAIN and TIMEOUT, and records `retroactive_skip_step_ids` ([decision_engine.py](https://github.com/amitraghvan/Orion/blob/HEAD/app/intelligence/decision_engine.py)).
  - **Outputs:** OS-speech TTS with priority preemption and cooldown; SQLite plus `events.json` plus `timeline.log`; async MP4 recording; an HTTP MJPEG stream on port 8080; a PySide6 "cockpit" GUI; backends for MPS, CUDA, ONNX and TensorRT. The README claims 205 tests passing.
  - **Microgravity handling:** only torso-centred, scale-normalised 2D skeletons. The README says 3D SMPL HMR is "PLANNED", with "FastHMR" distillation as long-term work.
  - **Accuracy (the team's own honest numbers):** 95.56% train accuracy but only 24.78% validation accuracy on held-out subjects, trained on 20 videos (`stgcn_har_v1.pt`, 455k params). Its action vocabulary matches the red/yellow box sample (pick_yellow, place_red, move_box, inspect_box).

- **arnab9957/realtime-work-detection** (2★/2 forks, created 2026-09-09, pushed 2026-09-27; about 31.7k code lines) ([repo](https://github.com/arnab9957/realtime-work-detection))
  - **Architecture:** an "8-Agent Blackboard / Digital Twin" design with agents for Perception (YOLOv8n plus YOLOv8n-pose), IMU, Fusion (constrained UKF), HAR ("AdaSpot RoI" plus a HOI engine), Digital Twin, Validation (deterministic FSM, 15-frame debounce), Reasoning (next step) and Monitoring (GUI, offline TTS, RTSP/HTTP, JSONL) ([README](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/README.md)).
  - **The IMU is virtual:** `IMUAgent.update_from_vision(pose)` derives 128 Hz IMU readings from vision pose, with ZUPT ([imu_agent.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/agents/imu_agent.py)).
  - **3D HMR:** a vendored NAVER Multi-HMR 2 (with the `anny` body model) runs asynchronously on the recorded video after the session ([mesh_recovery.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/core/mesh_recovery.py)). It is not in the live loop.
  - **VLM verifier:** a real-time step verifier sends Qwen3-VL-2B-Instruct through local Ollama, with 384² frames plus telemetry windows once per second, "to adjudicate procedural transitions and diagnose 'what wrong is going on'" ([realtime_llm_verifier.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/llm/realtime_llm_verifier.py)). An offline Qwen session analyser also exists.
  - **Other components:** a relational scene-graph model (RelSGG / RelateAnything) and ONNX export ([relsgg](https://github.com/arnab9957/realtime-work-detection/tree/HEAD/relsgg)); a metric-depth module (`src/ml_distance/depth/metric_depth.py`); Windows SAPI TTS with a pyttsx3 fallback ([offline_tts.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/audio/offline_tts.py)); threshold calibration scripts (`deploy/fit_calibration.py`, `deploy/calibrate_thresholds.py`).
  - **Synthetic data:** generated with OpenCV 2D drawing (cv2/numpy only), described as "domain-randomized", covering container_box, container_lid, red_box, yellow_box and astronaut_hand ([generate_synthetic_data.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/tools/generate_synthetic_data.py)). The README calls it a "3D generator" with inverted 0G angles.
  - **Claims:** JSONL telemetry at a "3,000,000:1 compression ratio (<15 KB per 30-min run)". The FSM covers lid-open angle ≥ 40°, red extraction, then yellow, with ERROR_SEQ (yellow first) and ERROR_SKIP (premature close) ([README](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/README.md)). Models are hosted on Google Drive.

- **Pranvkumar/AI-HAR-For-BAS**, "AEGIS" (0★, created 2026-08-23, pushed 2026-09-07; about 16.5k code lines; ships `models/exported/best.pt`, `objects.onnx` and `gesture_recognizer.task`) ([repo](https://github.com/Pranvkumar/AI-HAR-For-BAS))
  - **Orientation handling:** an explicitly "orientation-agnostic rack-relative coordinate frame". Four ArUco markers give a planar homography to normalised rack coordinates in [0,1]², plus the rack roll angle, with a static clicked-corner fallback ([rack_frame.py](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/src/aegis/perception/rack_frame.py)).
  - **Step model:** Tier 0 uses zones, grip aperture and dwell with no training. Tier 1 is a 2-layer GRU with attention pooling. It trains in PyTorch or in pure NumPy ("removes a 2.5 GB dependency") and emits ONNX directly. Validation holds out a whole operator by default, because "a random split reports 98% and then collapses on a new person" ([train_action_model.py](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/src/aegis/tools/train_action_model.py)).
  - **Fault-injection mode:** keys 1–5 simulate lens occlusion, dim lights, hidden markers, an unplugged camera or a disabled action model, to show graceful degradation ([failure_injection.py](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/src/aegis/safety/failure_injection.py)).
  - **Voice and I/O:** faster-whisper voice input in the GUI ([gui/app.py](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/src/aegis/gui/app.py)); pyttsx3/Piper TTS; NVENC recording; MJPEG at `:8000/stream`; JSONL; a Streamlit dashboard; MediaPipe Gesture Recognizer; a TensorRT export script. The target is a Legion laptop with an RTX 4060 ([README](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/README.md)).
  - **Dataset:** classes are outer_container, red_box, second_colored_box and astronaut_hand. The team notes the official text they saw "does not show the second colour", so they didn't guess it. A `contribute_images.py` script crowdsources photos from friends.
  - **Accuracy:** docs show "Held-out accuracy: 84.7% (operator: op3)", with retrieve_red_box F1 0.69 ([docs/TRAINING.md](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/docs/TRAINING.md)). It sits in a code block, so it may be example output.

- **notaditya28/isro_har_sih_project** (0★, created 2026-09-11; about 5.1k lines; ships `best.pt`) ([repo](https://github.com/notaditya28/isro_har_sih_project))
  - **Experiment:** a custom four-cube stacking protocol (green, yellow, red, blue, bottom-up). Re-touching a placed cube is recoverable.
  - **Architecture:** two independent readings cross-check each other. `stack_verifier.py` reads cube geometry from YOLO11n; `activity.py` reads reach/grasp/transport/place/retract from MediaPipe hand pinch points and cube motion. Disagreement between them is treated as an occlusion/collapse signal ([README](https://github.com/notaditya28/isro_har_sih_project/blob/HEAD/README.md)).
  - **Rejected alternative, with a measured reason:** a clip-length-only baseline scores 80.1% vs a 53.9% majority baseline, because Correct trials run 9.3 s median, Incorrect 6.2 s and Partial 4.3 s. So an end-to-end clip classifier would "learn duration instead of behaviour".
  - **Dataset:** 362 clips, 44 minutes, 4 simultaneous views. Clips are grouped into 102 trials by capture time, with iPhone UTC vs laptop IST corrected, and split at trial level to avoid leakage. Thresholds are expressed in seconds and cube-widths, never frames or pixels. `autolabel.py` uses a strong teacher and retrains at 1280 px.
  - **Tooling:** a PowerPoint deck built by script (`build_deck.py`).

- **atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174** (4★, created 2026-09-16, pushed 2026-09-26; about 12k lines) ([repo](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174))
  - **Perception:** YOLOv8 plus YOLOv8-Pose wrists, **ORB feature matching** (`cv2.ORB_create(2000)`) for open/closed container state, HSV colour, and a multi-object Kalman tracker with "coasting" ([engine.py](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174/blob/HEAD/bas-apg-backend/app/core/engine.py)).
  - **Pre-flight gate:** face-recognition biometric authorisation of the operator (`face_recognition`, `biometrics/encodings_cache.pkl`), then PPE checks: bare hands, gloves, eye protection.
  - **Guidance and telemetry:** deterministic template recovery guidance ("never generated by AI/LLM at runtime"); macOS `say` with a pyttsx3 fallback; SQLite in WAL mode; Protocol-Buffers telemetry schema; 30 Hz WebSocket; MJPEG; React 19 HUD. The WebSocket carries a "HOI Pearson's r" field ([README](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174/blob/HEAD/README.md)). The real computation of that field wasn't located; `scratch_engine.py` hard-codes values of 0.85/0.15/0.99.

- **gsaivenkateshkumar/ASTRA-HAR-EdgeAI** (0★, created 2026-08-29, pushed 2026-09-06; about 40k lines, about 10.7k JPEG frames) ([repo](https://github.com/gsaivenkateshkumar/ASTRA-HAR-EdgeAI))
  - **Data pipeline:** raw webcam recordings become per-video frame directories with a JSONL manifest. A deterministic, temporally uniform ~243-frame annotation subset is labelled in CVAT, then split at source-video level (171/37/35) ([astra-har/README.md](https://github.com/gsaivenkateshkumar/ASTRA-HAR-EdgeAI/blob/HEAD/astra-har/README.md)).
  - **Detector:** YOLO11n fine-tuned on CPU (v8 run). Best epoch: P 0.930, R 0.830, **mAP50 0.928**, mAP50-95 0.582 ([metrics.json](https://github.com/gsaivenkateshkumar/ASTRA-HAR-EdgeAI/blob/HEAD/astra-har/ai/training/detection/astra_har_detection_v8/metrics.json)).
  - **App:** an Electron plus React desktop app with a training/model-comparison UI. The README states that the WebSocket, streaming and dashboard were "not implemented yet" at the time of writing. A `claude.md` file suggests AI-assisted development.

- **MakInShort/Hangla-Coders-SIH2026174** (4★/5 forks, created 2026-08-27, pushed 2026-09-03; about 12.6k lines; 514 MB including `yolo26n/s.pt` weights and training runs) ([repo](https://github.com/MakInShort/Hangla-Coders-SIH2026174))
  - **Pipeline:** MediaPipe Hands and YOLO features feed an **XGBoost step classifier**, followed by a deterministic FSM (skip, repeat, out-of-sequence).
  - **Experiment:** the team chose its own **wireless-microphone protocol (M0–M6) with G1/G2 LED verification gates** instead of the box sample ([README](https://github.com/MakInShort/Hangla-Coders-SIH2026174/blob/HEAD/README.md); `runs/detect/runs/microphone_yolo`).
  - **Stack:** PySide6 and Streamlit GUIs, JSONL logs, 114 tests, GitHub Actions CI.

- **krishnakankane/isro-bas-har** (0★, created 2026-09-02, pushed 2026-09-11; about 11k lines; ships `conv1d_bigru_best.pt`) ([repo](https://github.com/krishnakankane/isro-bas-har))
  - **Pipeline:** YOLOv8n, MediaPipe Pose (33-point 3D) and Hands (21-point); a 60-frame Conv1D + BiGRU sequence classifier; a JSON protocol FSM with dwell-time violations; SAPI5/Piper voice; JSONL, SQLite and CSV logs; MJPEG; H.264 ring-buffer recording; PyQt6.
  - **Hardware and speed:** targets a GTX 1650 4 GB, measured at 16.8 FPS mean ([README](https://github.com/krishnakankane/isro-bas-har/blob/HEAD/README.md); [PERFORMANCE_BASELINE.md](https://github.com/krishnakankane/isro-bas-har/blob/HEAD/docs/PERFORMANCE_BASELINE.md)).
  - **Microgravity handling:** the README mentions "optional async CLIFF/SMPL HMR" and a "zero-g frame orientation normaliser", but source only has a note on BlazePose orientation limits (`src/pipeline_runner.py`). The 10-step protocol is openly labelled synthetic.

- **LakshanyaaJ/ORBITA** (0★/1 fork, created 2026-08-29, pushed 2026-09-20; 3,307 files) ([repo](https://github.com/LakshanyaaJ/ORBITA))
  - **Pipeline:** YOLO11n, then "HMR (24 SMPL joints)", then a dual-head GRU over 30 frames, then a 13-step FSM plus a JSON rule engine (R01–R07). Results are canonical JSON with SHA-256 checksums in SQLite, with an offline **sync queue to a "Ground Station Receiver"** that uses HTTP ACK and idempotency ([README](https://github.com/LakshanyaaJ/ORBITA/blob/HEAD/README.md)).
  - **The HMR is a fallback:** it looks for `4dhumans.pth` or `smpl_neutral.pkl`. Otherwise it runs a "Deterministic 3D Kinematic Lifter" and flags `is_fallback=True` ([hmr_pipeline.py](https://github.com/LakshanyaaJ/ORBITA/blob/HEAD/core_ai/perception/hmr_pipeline.py)).
  - **Experiment:** its own objects (pen, watch, blue box, yellow box, location A/B).
  - **Other:** phone-camera and IP-camera receivers; Jetson Orin Nano target; React 19 plus FastAPI.

- **mihikahetamsaria/SIH-Hackathon** (0★, created 2026-09-06, pushed 2026-09-19; about 8k lines) ([repo](https://github.com/mihikahetamsaria/SIH-Hackathon))
  - **Real HMR on real astronaut footage:** a genuine HMR2 (4D-Humans) SMPL pipeline runs on ESA astronaut videos (README cites ESA "Gripping experiment in space" and "Grasping for space"; usage example `1707_020_AR_EN.mp4`). It decodes named SMPL joints and computes a tilt angle from `global_orient`. It then **feeds frames plus 3D pose telemetry to Gemini** (cloud) for grounded step/error judgement, and plays back in a Tk UI ([astronaut_pipeline.py](https://github.com/mihikahetamsaria/SIH-Hackathon/blob/HEAD/Semantics/astronaut_pipeline.py); [extract_pose_features.py](https://github.com/mihikahetamsaria/SIH-Hackathon/blob/HEAD/Semantics/extract_pose_features.py)). The latter notes that a rack-relative "up" is still needed.
  - **Live path:** a `QwenRecognizer` sends frames to Ollama; pyttsx3 voice alerts; a ground-station receiver ([qwen_recognizer.py](https://github.com/mihikahetamsaria/SIH-Hackathon/blob/HEAD/live_camera/event_sequencing/qwen_recognizer.py)).

- **yamantaka-singh/BAS** (0★, created 2026-09-04) ([repo](https://github.com/yamantaka-singh/BAS))
  - **File pipeline:** the four stages are decoupled through files: landmarks.jsonl, features.jsonl, predictions.jsonl, events.jsonl, then payload.json. Changing a threshold means re-running the pipeline, not re-filming.
  - **Classifier and FSM:** MediaPipe Pose/Hands lite; torso-normalised kinematics; a **nearest-centroid classifier**; a deterministic FSM.
  - **Downlink and audit:** a downlink payload under 3 KB with "zero video", and a **tamper-evident Ed25519-signed hash chain** audit log.
  - **Honest stubs:** a "null provider" analysis module that renders "analysis unavailable" rather than faking output ([README](https://github.com/yamantaka-singh/BAS/blob/HEAD/README.md); [analysis/null.py](https://github.com/yamantaka-singh/BAS/blob/HEAD/analysis/null.py)). 105 tests.

- **aayushjadhav05128-sudo/bas-experiment-copilot** (0★, created 2026-09-05) ([repo](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot))
  - **Rust core:** the protocol FSM is written in **Rust and exposed through PyO3** (`bas_core`), with a Python fallback. It covers "confidence safety gating, cognitive hesitation tracking, and deviation classification" ([bas_core/src/lib.rs](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot/blob/HEAD/bas_core/src/lib.rs)).
  - **Adaptive voice:** the engine goes quiet after N nominal steps and prompts on hesitation ([voice.py](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot/blob/HEAD/backend/voice.py)).
  - **Explainer:** local Ollama (llama3.2:3b or phi3) with a template fallback ([explain.py](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot/blob/HEAD/backend/explain.py)).
  - **Other:** a 2-layer GRU action model; `events.jsonl` plus analytics.

**Tier B: working but conventional pipelines**
- **Shre-Sat/BASGUARD** (created 2026-09-10, pushed 2026-09-26) ([repo](https://github.com/Shre-Sat/BASGUARD))
  - **Perception:** YOLOv8x plus HSV for boxes; MediaPipe Hands grasp/pinch.
  - **Sequence logic:** a "Petri-net" token FSM with duration anomalies (too fast or too slow) ([fsm.py](https://github.com/Shre-Sat/BASGUARD/blob/HEAD/src/sequence/fsm.py)).
  - **Outputs:** **Piper neural TTS** with a priority queue; a **GStreamer RTSP stream plus a splitmuxsink ring-buffer MP4**; per-frame JSONL; PyQt6 with PyQtGraph.
  - **Protocol:** open outer box, touch/grasp/place red, touch/grasp/place yellow, close ([README](https://github.com/Shre-Sat/BASGUARD/blob/HEAD/README.md)).
  - **Claim vs code:** the description claims "orientation-agnostic 3D human mesh", but no HMR code was found.
- **StarCodes03/TARA-Astronaut-Assistant** (created 2026-09-06; about 14k lines) ([repo](https://github.com/StarCodes03/TARA-Astronaut-Assistant))
  - **Pipeline:** YOLOv8n plus YOLOv8n-pose; a "hand-coupled object tracker" that extrapolates held objects between inference cycles; an FSM-style trajectory classifier (PICK_UP, MOVE_TO_INSPECTION, INSPECT, PLACE, VALVE, PANEL).
  - **Platform:** Windows MSMF camera with AGC calibration; CPU-only; a web HUD that also shows ECLSS telemetry ([README](https://github.com/StarCodes03/TARA-Astronaut-Assistant/blob/HEAD/README.md)).
  - **Objects:** a custom model for bottle, with cup from COCO.
- **Sheen11-ing/AI-based-Human-Activity-Recognition-BAS-** (created 2026-09-03) ([repo](https://github.com/Sheen11-ing/AI-based-Human-Activity-Recognition-BAS-))
  - **Pipeline:** YOLO person and objects, YOLO-pose, an LSTM/GRU temporal model, SQLite, PySide6, MJPEG.
  - **HMR:** `hmr_interface.py` is a pluggable `HMRProvider` (HMR2/SMPL) interface ([hmr_interface.py](https://github.com/Sheen11-ing/AI-based-Human-Activity-Recognition-BAS-/blob/HEAD/BAS/src/hmr_interface.py)).
- **Rahulrwt99/bas-experiment-assistant** (created 2026-09-02): an **AprilTag** payload-rack frame via `pupil_apriltags` ([rack_calibration.py](https://github.com/Rahulrwt99/bas-experiment-assistant/blob/HEAD/src/perception/rack_calibration.py)). A step advances only after repeated above-threshold observations within a window, and each decision logs expected vs observed action, confidence and evidence. It has demo and deviation modes without camera or models ([README](https://github.com/Rahulrwt99/bas-experiment-assistant/blob/HEAD/README.md)).
- **Ayush-sharma-mx/bas-ai-monitor** (created 2026-08-26): YOLOv8, MediaPipe Holistic, HOI distances, and SKIPPED/OUT_OF_ORDER/REPEATED detection. pyttsx3, a Flask MJPEG stream on :5000, 5-minute MP4 chunks, JSONL and PyQt5. Includes `step_graph.json`, ONNX export and real recorded sessions (`data/raw/session_20260825_*.mp4`) ([README](https://github.com/Ayush-sharma-mx/bas-ai-monitor/blob/HEAD/README.md)).
- **PriyankShingala/Stellar** (created 2026-09-10): PySide6, YOLOv8n, MediaPipe, an interaction reasoner, a sequence validator and pyttsx3. The README is only 2 lines ([repo](https://github.com/PriyankShingala/Stellar)).
- **kavyareddypallala-cell/BAS-Vision** ("ORBITA", created 2026-09-09): YOLO for person, MediaPipe, OpenCV HSV for objects, heuristic activity inference, FSM, a vanilla-JS UI with Web Speech `SpeechSynthesis`, and MJPEG via FastAPI `StreamingResponse`. Openly an MVP "before custom … datasets are trained" ([README](https://github.com/kavyareddypallala-cell/BAS-Vision/blob/HEAD/README.md)).
- **Kartikey2006/Astra** (created 2026-08-28):
  - **Zero-shot HAR:** HuggingFace X-CLIP with text prompts per action for a notebook/pen experiment ([xclip_har.py](https://github.com/Kartikey2006/Astra/blob/HEAD/backend/app/har/xclip_har.py)).
  - **Cloud detector:** a `GeminiDetector` calling `gemini-3.6-flash` with an API key ([gemini_detector.py](https://github.com/Kartikey2006/Astra/blob/HEAD/backend/app/vision/gemini_detector.py)). This contradicts the offline claim.
  - **Other:** a synthetic feature generator; React plus FastAPI.
- **antas06/exp-ass-bas** (created 2026-09-06):
  - **LLM protocol parsing:** an LLM turns a free-text protocol into JSON (target classes, steps, subject/container, inside/outside). The target classes then configure **YOLO-World** open-vocabulary detection (`set_classes`).
  - **LLM judge:** each step's completion is decided by an LLM reading summaries of recent detections.
  - **Not offline:** both LLM calls go to `openai/gpt-oss-20b` through **Groq's cloud API** ([parser.py](https://github.com/antas06/exp-ass-bas/blob/HEAD/src/parser.py); [state_machine.py](https://github.com/antas06/exp-ass-bas/blob/HEAD/src/state_machine.py); [detector.py](https://github.com/antas06/exp-ass-bas/blob/HEAD/src/detector.py)).
  - **Demo protocol:** "Mouse and Water Bottle Placement".
- **ThouheedAhmed/DRISHTI-Edge-AI** (created 2026-09-12): MediaPipe Pose plus a scikit-learn RandomForest posture classifier (`model/posture_classifier.pkl`), pyttsx3, Flask and VideoWriter ([repo](https://github.com/ThouheedAhmed/DRISHTI-Edge-AI)).
- **Himashwetha777/SIH26174-Astronaut-HAR-System** (0★/1 fork, created 2026-09-13): MediaPipe and pyttsx3, plus **astronaut face identification, Fernet encryption, auth/session and audit managers**, and zone management; Streamlit/Flask ([repo](https://github.com/Himashwetha777/SIH26174-Astronaut-HAR-System)).
- **SAMEERSACHDEVA2817/…** (created 2026-09-09): MediaPipe features, a RandomForest activity classifier with a training script, an anomaly detector, FastAPI and React ([repo](https://github.com/SAMEERSACHDEVA2817/AI-Human-Activity-Recognition-for-On-board-BAS-Experiments)).
- **ArrushiTripathi2429/Space-Sight2** (created 2026-09-11): a single 743-line `har_prototype.py` with MediaPipe Tasks pose and hand landmarkers ([repo](https://github.com/ArrushiTripathi2429/Space-Sight2)).
- **MrRouhin/Astronaut-AI-Assistance** (created 2026-09-06): YOLOv8 trained on a Roboflow-labelled "hazard-triage" webcam dataset (hazardous/rare classes, captured 2026-09-07), MediaPipe hands and PyQt5 ([repo](https://github.com/MrRouhin/Astronaut-AI-Assistance)).

**Tier C: README, design-doc, UI or simulator only**
- **JaiPrakashVI/AI-Human-Activity-Recognition-Space** (pushed 2026-09-24): **wearable-IMU HAR, not video**. Wrist, chest and ankle IMUs from **PAMAP2**, with simulated SO(3) rotations, gravity-cue scaling, noise, drift and dropout; a 1D-CNN under 100k params (161 KB INT8); **HMM-Viterbi smoothing** that raised macro-F1 from 0.3560 to 0.4362 and cut false transitions by 88.35%; a procedural state engine; and a 98.86% telemetry reduction. Best configuration: chest IMU, 74.27% accuracy, 0.714 macro-F1 ([README](https://github.com/JaiPrakashVI/AI-Human-Activity-Recognition-Space/blob/HEAD/README.md)).
- **haasitha1603/NAVIKSHA** (pushed 2026-09-17): React plus FastAPI; a "ColorSort-Rack" protocol; an animated synthetic payload-rack feed for demos without a camera; a visual **Protocol Designer**; Web Speech TTS; SQLite; JSON/JSONL/CSV/TXT export ([README](https://github.com/haasitha1603/NAVIKSHA/blob/HEAD/README.md)).
- **rudra00434/HOMOSENSYS-AI-Assistant**: a "simulated pipeline" with DEMO_MODE on by default; React plus FastAPI WebSocket; "grounded guidance assistant" ([README](https://github.com/rudra00434/HOMOSENSYS-AI-Assistant/blob/HEAD/README.md)).
- **dnvrmhra/ASTRA-HAR**: a TypeScript web app with simulation and a local IndexedDB audit ledger. The README says the astronaut gets visual cues "without audio", which conflicts with the voice-alert requirement, and claims 18 MB/hr of telemetry vs 2.4 GB/hr of video ([README](https://github.com/dnvrmhra/ASTRA-HAR/blob/HEAD/README.md)).
- **Nimbus3001/KRAMA** (created 2026-09-24): Phase 2 of a planned build. It has video ingest, sessions, MJPEG and a deterministic sequence engine. "No AI model is implemented yet"; RTMPose appears only in config ([KRAMA/README.md](https://github.com/Nimbus3001/KRAMA/blob/HEAD/KRAMA/README.md)).
- **Aryasurya12/AstroFlow-AI** (2★): README and docs only. Planned YOLOv8s/11s, MediaPipe/YOLO-Pose, anchor-relative normalisation, Piper, NVENC, MJPEG/RTSP on an RTX 3050 edge server. "Dataset collection … not yet started" ([README](https://github.com/Aryasurya12/AstroFlow-AI/blob/HEAD/README.md)).
- **Vynex-Labs/…**: a 1,584-line `MASTER_PROMPT.md` (an LLM build prompt) planning RTMPose, TCN/ST-GCN, TensorRT and synthetic data ([MASTER_PROMPT.md](https://github.com/Vynex-Labs/AI-Human-Activity-Recognition-for-On-Board-BAS-Experiments/blob/HEAD/MASTER_PROMPT.md)).
- **UI-only or mock:** [hamsadevang849-sys/…](https://github.com/hamsadevang849-sys/AI-Human-Activity-Recognition-for-On-board-BAS-Experiments) (Three.js station vista, browser TTS, "BAS-EXP-04: Biological Specimen Extraction" simulator), [renukathombre1111/BAS-AI-Experiment-Assistant](https://github.com/renukathombre1111/BAS-AI-Experiment-Assistant) (`mock_engine.py` adapter), [sohamjachak003/AI-BAS-Assistant_2](https://github.com/sohamjachak003/AI-BAS-Assistant_2) (`simulation/demo_action_engine.py`), [samaksh-divs/bas-experiment-copilot-frontend](https://github.com/samaksh-divs/bas-experiment-copilot-frontend) (mock adapter plus a FastAPI contract doc), [SyroSyrup98/ORBIT-HAR](https://github.com/SyroSyrup98/ORBIT-HAR) (FastAPI camera stub) and [xarjunpatil/SIH26174-…](https://github.com/xarjunpatil/SIH26174-AI-Human-Activity-Recognition-for-On-board-BAS-Experiments).
- **[saurav158204/vistaspace-site](https://github.com/saurav158204/vistaspace-site)**: a marketing and demo site for "VISTASpace — Visual Intelligent Sequence Tracking & Alert Assistant" on NVIDIA Jetson. Its reference list cites **MicroG-4M**, PhysAstro-Pose, a NASA ISS command-error study and a NASA 2025 communication-delay assessment ([README](https://github.com/saurav158204/vistaspace-site/blob/HEAD/README.md)).
- **[AjitKumar166/sih26-project](https://github.com/AjitKumar166/sih26-project)**: README only. An offline RAG/LLM "astronaut assistant" (Ollama or llama.cpp) whose protocol database is 7 real Indian microgravity experiments (tardigrade, myogenesis, methi/moong sprouting, cyanobacteria, microalgae, crop-seed exposure…).

**Discarded (surfaced by topical terms, not PS 26174):** [Ariaeimehr/Zero-G-Space-HAR](https://github.com/Ariaeimehr/Zero-G-Space-HAR) (federated IMU transformer for ARED/T2/CEVIS exercise); [nareshkumarchellani/AI-Astronaut](https://github.com/nareshkumarchellani/AI-Astronaut) (Mars mission simulator); [sakshicharlewar/Antariksh-X](https://github.com/sakshicharlewar/Antariksh-X) (satellite-image search app); the MAITRI well-being repos; generic HAR repos such as [samhithaharish/edge-har](https://github.com/samhithaharish/edge-har) and [HanieKhoshkenar/Deep-HMM-HAR](https://github.com/HanieKhoshkenar/Deep-HMM-HAR) (no evidence competitors forked them).

### Inferences
- The full sample experiment seems to be "open outer box, take out red, take out yellow (then close/return)". Competitors encode it as: lid ≥ 40°, red extracted, yellow extracted, with a premature-close skip (arnab9957); touch/grasp/place red, then yellow, then close (BASGUARD); pick/place red/yellow, move_box, inspect_box (Orion). At least one team (Pranvkumar) saw the same truncated text we have. The full statement should be confirmed from the NoBugNinja dump or the SIH portal.
- **Trained models:** very few repos ship a trained, experiment-specific model with honest numbers. Orion's 24.78% held-out validation shows how badly a 20-video skeleton classifier generalises. That supports ACTUS's choice of rules over pretrained evidence rather than a thinly trained classifier, provided it is described honestly.
- **Real 3D HMR:** only two teams run a real HMR network (arnab9957 with Multi-HMR 2; mihikahetamsaria with HMR2/4D-Humans), and both run it offline/batch. Most others either fake it (ORBITA's deterministic lifter), stub it (Sheen11's interface, krishnakankane's README) or explicitly defer it (Orion). ACTUS's cut of HMR in favour of rotation-TTA plus canonicalised skeletons matches the field. The only real-footage peers are mihikahetamsaria (ESA videos with HMR2 and Gemini) and VISTASpace (cites MicroG-4M, no code seen).
- **Offline claims:** several repos that claim "100% offline" actually call cloud LLMs: Kartikey2006 (Gemini), antas06 (Groq) and mihikahetamsaria's batch path (Gemini). Judges who check could mark this down.

### Gaps
- Repos in the ID-based researcher's clone set (e.g. KriyaSense, ASTRA-SEQ, VisionX, Viksit-Coders, hawk-bas, ORBIT-HAR by h1e2l3l4o5, Omkar96-18) were not re-profiled here.
- We did not run any repo, so all runtime behaviour is inferred from source and README. The accuracy numbers are the teams' own and were not reproduced.
- Pranvkumar's "84.7% held-out" may be illustrative output in the docs, not a real run. We couldn't confirm it.
- ATLAS's "HOI Pearson's r" computation wasn't located in the live engine, so it may be partly mocked.
- Google Drive assets (arnab9957 models and datasets) were not inspected.

## Q3: Which ideas are unusual or novel compared with the typical YOLO + MediaPipe + rules pipeline, and how do they compare with ACTUS?

### Takeaway
The notable departures from the standard pipeline fall into seven groups:
1. Real 3D HMR, run offline (Multi-HMR 2, HMR2).
2. Local VLM/LLM step adjudicators (Qwen3-VL-2B or Qwen via Ollama).
3. Fiducial rack frames (ArUco homography, AprilTag) as the orientation-agnostic answer.
4. Evaluation rigour: operator- or trial-held-out splits, and a clip-duration leakage analysis.
5. Safety and assurance features: fault injection, Ed25519-signed audit chains, SHA-256 ground-sync queues, biometric and PPE gates.
6. Alternative sequence formalisms: Petri-net tokens, HMM-Viterbi smoothing, entropy gating, a Rust PyO3 FSM.
7. Open-vocabulary pipelines: LLM protocol parsing into YOLO-World classes, X-CLIP zero-shot actions, relational scene graphs.

No competitor seen uses ACTUS's 4-rotation pose test-time augmentation with confidence selection, and none uses a pretrained microgravity action model (MicroG-4M) as an evidence stream. One site only cites MicroG-4M.

### Cited Findings
**3D HMR, run offline**
- **arnab9957:** runs NAVER Multi-HMR 2 on the recorded session after the run ([mesh_recovery.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/core/mesh_recovery.py)).
- **mihikahetamsaria:** runs HMR2/4D-Humans on ESA astronaut footage and passes SMPL-joint telemetry to a VLM as grounding ([astronaut_pipeline.py](https://github.com/mihikahetamsaria/SIH-Hackathon/blob/HEAD/Semantics/astronaut_pipeline.py)).

**Local VLM/LLM step adjudication**
- **arnab9957:** Qwen3-VL-2B via Ollama at 1 Hz, fused with YOLO boxes ([realtime_llm_verifier.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/llm/realtime_llm_verifier.py)).
- **mihikahetamsaria:** a Qwen recognizer through Ollama ([qwen_recognizer.py](https://github.com/mihikahetamsaria/SIH-Hackathon/blob/HEAD/live_camera/event_sequencing/qwen_recognizer.py)).
- **aayushjadhav:** a llama3.2:3b explainer with template fallback ([explain.py](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot/blob/HEAD/backend/explain.py)).
- **ATLAS takes the opposite stance:** recovery text is deterministic and "never generated by AI/LLM at runtime" ([README](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174/blob/HEAD/README.md)).

**Fiducial rack frames for the orientation-agnostic requirement**
- **Pranvkumar/AEGIS:** a 4-ArUco homography to rack coordinates, with a test that renders the scene rotated and checks invariance ([rack_frame.py](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/src/aegis/perception/rack_frame.py); [Krishna105 README](https://github.com/Krishna105-ui/Astronaut-Activity-Recognition-System/blob/HEAD/README.md)).
- **Rahulrwt99:** an AprilTag rack frame ([rack_calibration.py](https://github.com/Rahulrwt99/bas-experiment-assistant/blob/HEAD/src/perception/rack_calibration.py)).
- **Torso normalisation without markers:** Orion ([README](https://github.com/amitraghvan/Orion/blob/HEAD/README.md)) and yamantaka-singh ([README](https://github.com/yamantaka-singh/BAS/blob/HEAD/README.md)).

**Evaluation rigour**
- **Pranvkumar:** operator-held-out validation, with the explicit warning that random splits give a falsely high 98% ([train_action_model.py](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/src/aegis/tools/train_action_model.py)).
- **notaditya28:** trial-level splits across 4 simultaneous views, UTC/IST timestamp repair, and a clip-duration leakage baseline of 80.1% vs 53.9% majority. Also dual-channel cross-checking (geometry vs activity), where disagreement is treated as a signal ([README](https://github.com/notaditya28/isro_har_sih_project/blob/HEAD/README.md)).
- **gsaivenkateshkumar:** source-video-level splits for the detector ([astra-har/README.md](https://github.com/gsaivenkateshkumar/ASTRA-HAR-EdgeAI/blob/HEAD/astra-har/README.md)).

**Safety and assurance extras**
- **Fault injection with live degradation modes:** Pranvkumar ([failure_injection.py](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/src/aegis/safety/failure_injection.py)).
- **Ed25519-signed audit chain plus a <3 KB downlink payload:** yamantaka-singh ([README](https://github.com/yamantaka-singh/BAS/blob/HEAD/README.md)).
- **SHA-256 canonical JSON plus an offline sync queue to a ground receiver:** ORBITA ([README](https://github.com/LakshanyaaJ/ORBITA/blob/HEAD/README.md)).
- **Face-biometric operator auth plus glove and eye-protection PPE gates:** ATLAS ([README](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174/blob/HEAD/README.md)).
- **Face identification plus Fernet encryption:** Himashwetha777 ([repo](https://github.com/Himashwetha777/SIH26174-Astronaut-HAR-System)).

**Alternative sequence and uncertainty formalisms**
- **Petri-net token FSM with too-fast/too-slow duration anomalies:** BASGUARD ([fsm.py](https://github.com/Shre-Sat/BASGUARD/blob/HEAD/src/sequence/fsm.py)).
- **Shannon-entropy gating with retroactive skip IDs:** Orion ([decision_engine.py](https://github.com/amitraghvan/Orion/blob/HEAD/app/intelligence/decision_engine.py)).
- **HMM-Viterbi smoothing (+22.5% macro-F1):** JaiPrakashVI ([README](https://github.com/JaiPrakashVI/AI-Human-Activity-Recognition-Space/blob/HEAD/README.md)).
- **Rust/PyO3 FSM with "cognitive hesitation tracking" and adaptive voice silence:** aayushjadhav ([lib.rs](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot/blob/HEAD/bas_core/src/lib.rs); [voice.py](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot/blob/HEAD/backend/voice.py)).
- **Nearest-centroid step classifier with a re-runnable file pipeline:** yamantaka-singh ([README](https://github.com/yamantaka-singh/BAS/blob/HEAD/README.md)).

**Open-vocabulary and zero-shot**
- **LLM parses free-text protocol into JSON, then YOLO-World `set_classes`, then an LLM judges step completion:** antas06, cloud Groq ([parser.py](https://github.com/antas06/exp-ass-bas/blob/HEAD/src/parser.py); [detector.py](https://github.com/antas06/exp-ass-bas/blob/HEAD/src/detector.py)).
- **X-CLIP text-prompt zero-shot actions:** Kartikey2006 ([xclip_har.py](https://github.com/Kartikey2006/Astra/blob/HEAD/backend/app/har/xclip_har.py)).
- **RelateAnything relation triplets / scene graph:** arnab9957 ([relsgg](https://github.com/arnab9957/realtime-work-detection/tree/HEAD/relsgg)).
- **Visual protocol designer:** NAVIKSHA ([README](https://github.com/haasitha1603/NAVIKSHA/blob/HEAD/README.md)).

**Other object-state and streaming tricks**
- **ORB feature matching for container open/closed:** ATLAS ([engine.py](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174/blob/HEAD/bas-apg-backend/app/core/engine.py)).
- **Virtual IMU derived from vision pose, fused by UKF:** arnab9957 ([imu_agent.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/agents/imu_agent.py)).
- **GStreamer RTSP plus splitmuxsink ring buffer:** BASGUARD ([README](https://github.com/Shre-Sat/BASGUARD/blob/HEAD/README.md)).
- **faster-whisper voice commands:** Pranvkumar ([gui/app.py](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/src/aegis/gui/app.py)).

**Choosing your own experiment instead of the box sample**
- Wireless microphone with LED gates: Hangla-Coders ([README](https://github.com/MakInShort/Hangla-Coders-SIH2026174/blob/HEAD/README.md)).
- Four-cube tower: notaditya28.
- Notebook/pen: Kartikey2006.
- Mouse on bottle: antas06.
- Pen/watch/boxes: ORBITA.

**Real astronaut footage or microgravity datasets**
- Only mihikahetamsaria uses ESA astronaut videos.
- VISTASpace cites MicroG-4M and PhysAstro-Pose as references ([README](https://github.com/saurav158204/vistaspace-site/blob/HEAD/README.md)).
- We found no competitor that loads MicroG-4M weights.

### Inferences
- **Where ACTUS looks differentiated:**
  - It is the only approach seen that (a) runs a pretrained microgravity action-recognition model (MicroG-4M) as an evidence stream on real astronaut footage, and (b) does marker-free orientation handling by 4-rotation pose TTA with confidence selection, plus skeleton canonicalisation that keeps a separate raw torso-angle channel.
  - Its two-channel belief split (protocol prior vs observation), and the abstain/ask path, have no direct analogue. The closest are Orion's STEP_UNCERTAIN/entropy gate and Rahulrwt99's "prediction as uncertain evidence".
- **Where competitors look stronger or could pressure ACTUS in judging:**
  - Fiducial rack frames (AEGIS, Rahulrwt99) answer "relative to the payload rack" more literally than torso canonicalisation. ACTUS's retained ArUco levelling could be pitched as an optional rack-frame channel.
  - Real HMR demos (arnab9957, mihikahetamsaria) produce visually impressive SMPL overlays.
  - Local VLM verifiers are a trendy "AI" story.
  - Rigorous held-out evaluation (notaditya28, Pranvkumar) gives credible numbers.
  - Assurance features (signed audit chains, fault injection) are cheap to add and play well with ISRO judges.
- **Openings for ACTUS:**
  - Several competitors' "offline" claims are contradicted by cloud LLM calls (Gemini, Groq), and several "HMR" claims are stubs or fakes. ACTUS can win on honest, verifiable claims: state clearly what is trained, what is rule-based and what runs offline.
  - Borrowing ideas worth low effort: notaditya28's duration-leakage check (clip length alone predicting the verdict), Pranvkumar's operator-held-out split, and Orion's retroactive-skip record.

### Gaps
- We cannot tell whether any team integrated MicroG-4M or other real microgravity datasets beyond the reference citations. Their private Drive assets and demo videos were not examined.
- There is no information on how SIH judges will weigh the optional HMR requirement against the core requirements, so we can't rank how much the HMR demos threaten ACTUS.
- We did not benchmark whether the local VLM verifiers (Qwen3-VL-2B) run at useful latency on edge hardware. arnab9957 polls at 1 Hz but reports no measured latency.
