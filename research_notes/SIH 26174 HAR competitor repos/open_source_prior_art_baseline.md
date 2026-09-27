# Open-source prior art baseline for SIH 26174 (AI HAR for on-board BAS experiments): commodity vs novel

Scope note: star counts, last-push dates and licences were pulled live from the GitHub REST API (`api.github.com/repos/...`) on 2026-09-27 unless another source is given. `gh` CLI was not installed, and unauthenticated GitHub search hit its rate limit partway through, so the repo searches were not exhaustive.

## Q1. Open-source repos for procedural/assembly step recognition and mistake detection, and what they ship

### Takeaway
The procedural-mistake-detection field is mostly datasets plus research code built on pre-extracted features: I3D/VideoMAE/Omnivore features, then ActionFormer-style segmentation, task graphs or LLM reasoning. The repos are low-star (roughly 10 to 80), trained on egocentric kitchen and assembly footage, and ship no plug-and-play live-video pipeline. A hackathon team can borrow the ideas (error taxonomy, task graphs, "skip/out-of-order" framing) but can't realistically fork a working live system from them. Competitors who say they "use EgoPER/Assembly101" are almost certainly borrowing concepts, not running the code.

### Cited Findings
**Datasets and benchmarks (what exists)**
- **EgoPER** (CVPR 2024), official repo `robert80203/EgoPER_official`: 41 stars, last push 2025-09-20, no licence file. It ships an ActionFormer-based segmentation backbone with "2 prototypes per step", pre-extracted I3D (2048-d) and VideoCLIP features, and train/eval shell scripts. The dataset needs an email request. It covers 5 egocentric procedural (cooking) tasks with RGB, depth, audio, gaze and hand tracking — [EgoPER repo](https://github.com/robert80203/EgoPER_official)
- EgoPER has 120+ h of egocentric cooking from 11 participants, with an error taxonomy of "step omission, addition, modification, slip, correction" — [Mistake-analysis review, arXiv 2510.19292](https://arxiv.org/html/2510.19292)
- **Assembly101 mistake detection** repo (`assembly-101/assembly101-mistake-detection`): 12 stars, last push 2023-08-03. It contains annotations only, with no models, feature extractors or weights. Labels are correct / mistake (wrong order, shouldn't have happened, wrong position) / correction, over 328 sequences. Licence is CC BY-NC 4.0 — [Assembly101 mistake repo](https://github.com/assembly-101/assembly101-mistake-detection)
- Assembly101 has 362 multi-view toy-car assembly videos from 53 participants, RGB plus 3D hand pose — [review 2510.19292](https://arxiv.org/html/2510.19292)
- **CaptainCook4D** (`CaptainCook4D/CaptainCook4D`): 11 stars, last push 2025-03-30. The downloader repo is Apache-2.0 with a last push of 2026-08-28. The dataset has 384 recordings (94.5 h), 5.3K step annotations and 10K fine-grained action annotations. Error types are technique, measurement, order and preparation. Baselines cover supervised and zero-shot error recognition and multi-step localization, in separate linked repos — [CaptainCook4D repo](https://github.com/CaptainCook4D/CaptainCook4D). The review counts 24 recipes, 8 participants and "7 mistake types" — [review 2510.19292](https://arxiv.org/html/2510.19292). *The repo and the review report the error taxonomy differently (4 vs 7 types).*
- **HoloAssist** (Microsoft, ICCV 2023) has 166 h from 350 instructor-performer pairs (222 participants) across 20 object-centric manipulation tasks. It includes benchmarks on mistake detection, intervention-type prediction and hand forecasting — [arXiv 2309.17024](https://arxiv.org/abs/2309.17024), [HoloAssist site](https://holoassist.github.io/). It has 414 coarse and 1,887 fine-grained action classes. Its mistake baseline uses ViT + TimeSformer per step — [review 2510.19292](https://arxiv.org/html/2510.19292). The GitHub release repo `Ember-HoloAssist/holoassist-release` has 14 stars (GitHub search API).
- **IKEA ASM** (`IkeaASM/IKEA_ASM_Dataset`): 78 stars, last push 2026-06-10, code MIT. It has about 3M frames (about 240 GB) of furniture assembly with RGB (multi-view), depth, human pose, object segmentation and atomic action labels. It provides pretrained models for action recognition, pose and part tracking — [IKEA ASM repo](https://github.com/IkeaASM/IKEA_ASM_Dataset). *IKEA ASM is third-person, fixed-camera and full-body, which makes it the closest public analogue to ISRO's "fixed payload cameras" framing.*
- **MECCANO** (`fpv-iplab/MECCANO`): 32 stars, last push 2023-07-31 (inactive) — GitHub API.
- **Ego4D / Ego-Exo4D keystep**: `facebookresearch/Ego4d` has 651 stars, MIT, last push 2026-07-25 (GitHub API). The Ego-Exo4D keystep recognition benchmark uses 278 keysteps. Official baselines are in `EGO4D/ego-exo4d-keystep` (`fine_grained/`, `procedure_understanding/` with a "Graph-Based with Keystep Assignment" baseline) — [Ego-Exo4D keystep docs](https://docs.ego-exo4d-data.org/benchmarks/keystep/keystep_recoginition/), [ego-exo4d-keystep repo](https://github.com/EGO4D/ego-exo4d-keystep/tree/main/procedure_understanding). Ego-Exo4D is 1,286 h of ego and exo video with keysteps, procedural dependencies and 3D body/hand pose — [Ego-Exo4D paper](https://arxiv.org/pdf/2311.18259)
- More GitHub API / search metadata (2026-09-27):
  - `assembly-101/assembly101-action-recognition` ("Code and models for the Action Recognition benchmark"): 16 stars, last push 2023-03.
  - `assembly-101/assembly101-download-scripts`: 81 stars.
  - `facebookresearch/ego4d-goalstep` (Ego4D Goal-Step, NeurIPS 2023): 64 stars, MIT, last push 2024-04.
  - `Yuhan-Shen/ProTAS` ("Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos"): 40 stars. *This is online step segmentation, directly relevant to live step tracking.*
  - `z1oong/Building-Egocentric-Procedural-AI-Assistant` (survey/reading list): 56 stars.
  - Student-project clones exist, e.g. `arigato007/DAAI_PROJECT03_MISTAKE_DETECTION` (Omnivore/SlowFast features on CaptainCook4D).
- Other mistake datasets named by the review: Epic-Tent, CSV (Chemical Sequence Verification: 70 videos of 14 chemical experiments, with step addition, deletion and reordering, *the closest "lab protocol" analogue*), BRIO-TA, ATA, IndustReal (84 videos, 38 errors), EgoOops and EK-M/Ego4D-M (synthetic) — [review 2510.19292](https://arxiv.org/html/2510.19292)

**Methods with code**
- **PREGO** (CVPR 2024, `aleflabo/PREGO`): 35 stars, MIT, last push 2025-06-09. It does online mistake detection by combining online action recognition with LLM symbolic next-action prediction. A follow-up (TI-PREGO / Plini et al.) adds chain-of-thought — [PREGO repo metadata via GitHub API](https://github.com/aleflabo/PREGO); [review 2510.19292](https://arxiv.org/html/2510.19292)
- **Differentiable Task Graph Learning** (`fpv-iplab/Differentiable-Task-Graph-Learning`): 24 stars, last push 2026-09-01, "Procedural Activity Representation and Online Mistake Detection" — GitHub search API; method described in [arXiv 2502.17753](https://arxiv.org/pdf/2502.17753)
- **AEM** (ICLR 2026, "Procedural Mistake Detection via Action Effect Modeling", `WenliangGuo/AEM`): 7 stars — GitHub search API
- The review groups methods into structure-free clip classifiers (VideoMAE, X3D, Omnivore, SlowFast), step-wise models (HoloAssist ViT+TimeSformer, MistSense hand pose + RGB), graph-based models (task-graph MLE, GTG2Vid, AMNAR) and online models (PREGO). It lists open challenges including "long-range temporal reasoning to assess whether an action sequence aligns with predefined procedure" — [review 2510.19292](https://arxiv.org/html/2510.19292)

### Inferences
- None of these repos gives a team a "webcam in, step alerts out" system. All of them assume offline feature extraction on egocentric data. The skip / out-of-order check that ISRO asks for is, in practice, a hand-written FSM over a known step list, and that is trivial. The published research (task graphs, PREGO) is about learning the graph or predicting the next step, which ISRO does not require because the procedure is given.
- Because every dataset is egocentric or third-person Earth footage of kitchens and assembly, "we used EgoPER/Assembly101/HoloAssist" is a citation claim, not a technical asset. A competitor that fine-tunes on one of them still faces the same transfer gap to a fixed ISS camera.
- Of the step datasets, IKEA ASM (third-person, pose-annotated, MIT code) and CSV (chemistry protocol steps) are the closest fits to the ISRO setting.

### Gaps
- I did not verify whether HoloAssist or Ego-Exo4D keystep baselines ship downloadable checkpoints usable without the full dataset.
- The MECCANO and Ego4D Goal-Step repo contents were not fetched (the GitHub search rate limit blocked "goalstep").
- Licences for the EgoPER and CaptainCook4D main repos show "None" in the GitHub API. The dataset terms were not checked.

## Q2. Off-the-shelf pipelines for hand-object interaction, skeleton action recognition and real-time HAR demos, and the "tutorial" repos teams copy

### Takeaway
The stack a hackathon team can copy is (a) Ultralytics YOLO detect + pose (62K stars), (b) MediaPipe Pose/Hands/Holistic (37K stars), (c) an LSTM over keypoint sequences following the widely cloned sign-language tutorial pattern, or ST-GCN(++) from MMAction2/PYSKL, and (d) proximity heuristics between hand keypoints and object boxes. Several "YOLOv8-pose + ByteTrack + LSTM + Streamlit" repos already exist as templates. At least one competitor repo (ORION) already uses exactly YOLO11n-pose + ST-GCN + ByteTrack + FSM. Every component in that stack is commodity.

### Cited Findings
- **Ultralytics** (`ultralytics/ultralytics`): 62,036 stars, AGPL-3.0, pushed 2026-09-27 (GitHub API). Its prebuilt "Solutions" include Workouts Monitoring (joint-angle rep counting, "AI Gym"), Object Counting in Regions, Distance Calculation, Track Objects in Zone, Security Alarm and Live Inference with Streamlit — [Ultralytics Solutions docs](https://docs.ultralytics.com/solutions/). *Region counting, distance calculation and zone tracking are ready-made "hand/object in zone" heuristics.*
- **MediaPipe** (`google-ai-edge/mediapipe`): 37,088 stars, Apache-2.0, pushed 2026-09-25 (GitHub API).
- **MMAction2** (`open-mmlab/mmaction2`): 5,163 stars, Apache-2.0, last push 2026-03-18 (GitHub API). It hosts ST-GCN, ST-GCN++, AAGCN and PoseC3D with NTU checkpoints. ST-GCN++ is 1.4M params, 89.29% joint-only on NTU60 X-Sub — [MMAction2 skeleton model zoo](https://mmaction2.readthedocs.io/en/dev-1.x/model_zoo/skeleton.html) (already in ACTUS preliminary research).
- **MMPose** (`open-mmlab/mmpose`, home of RTMPose): 7,924 stars, Apache-2.0, last push 2025-08-04 (GitHub API). *Its commit activity has slowed.*
- **PYSKL** (`kennymckormick/pyskl`): 1,268 stars, Apache-2.0. It implements ST-GCN++, PoseConv3D, DG-STGCN, MS-G3D, CTR-GCN, AAGCN and ST-GCN with model zoos (NTU, K400, UCF101, HMDB51, FineGYM, Diving48). It includes a "real-time gesture recognition demo based on skeleton-based action recognition" that runs on CPU. The README says: "This repo is currently not maintained by the developer" — [PYSKL repo](https://github.com/kennymckormick/pyskl)
- **100DOH hand-object detector** (`ddshan/hand_object_detector`): 297 stars, last push 2023-10-12. It uses Faster R-CNN ResNet-101 and outputs hand box, left/right, contact state (none/self/other-person/portable/stationary object) and an offset vector to the object. Two models: handobj_100K at 89.8% hand AP and handobj_100K+ego at 90.4%. It needs CUDA compilation (`python setup.py build develop`) and PyTorch 1.12.1/CUDA 11.3 or legacy 1.0.1/CUDA 10.0. It publishes no speed numbers. The code licence on GitHub is MIT — [hand_object_detector repo](https://github.com/ddshan/hand_object_detector). *This conflicts with ACTUS preliminary-research.md, which calls 100DOH "non-commercially licensed". The GitHub code licence is MIT; the dataset or weights terms may differ and were not checked.*
- **Tutorial templates teams clone:**
  - `nicknochnack/ActionDetectionforSignLanguage`: 503 stars, MediaPipe Holistic keypoints → TF-Keras LSTM, the canonical "keypoints + LSTM" YouTube tutorial (GitHub API).
  - `Manthan1920/human-pose-action-recognition`: YOLOv8-Pose (17 kp) + ByteTrack + rule-based or LSTM classifier + Streamlit UI + Docker — [repo](https://github.com/Manthan1920/human-pose-action-recognition)
  - `stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input`: LSTM over 18-joint 2D pose — [repo](https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input)
  - Ultralytics issues #8496 and #8846 cover "Merge YOLOv8 with LSTM for HAR" — [#8496](https://github.com/ultralytics/ultralytics/issues/8496), [#8846](https://github.com/ultralytics/ultralytics/issues/8846)
  - Many 0 to 2 star 2025-2026 clones of "YOLOv8 + pose + LSTM" exist (padel, exercise, garbage-dumping, sign language) — GitHub search API
- **Competitor evidence of the commodity stack:**
  - `amitraghvan/Orion` ("Offline-first Edge AI Copilot for Bharatiya Antariksh Station") uses YOLO11n + YOLO11n-pose, ST-GCN, ByteTrack, an 11-state FSM for skipped/out-of-order steps, macOS `say`/pyttsx3/espeak-ng voice, FastAPI MJPEG streaming, async MP4 recording, a PySide6 GUI, SQLite + JSON logs, and MPS/CUDA/CPU support — [Orion repo](https://github.com/amitraghvan/Orion)
  - `Krishna105-ui/Astronaut-Activity-Recognition-System` (MIT, last push 2026-08-24): "validating experimental procedures in real time using computer vision, object detection, pose est[imation]" — GitHub API. Other researchers cover these repos in detail.

### Inferences
- The YOLO-pose / MediaPipe → LSTM or ST-GCN → FSM → pyttsx3 → Streamlit/PyQt pipeline is the default shape for this problem. ORION shows a team can ship all of it, with a polished GUI, without any novel ML. Judges will see this stack repeatedly.
- "Hand-object interaction" in most competitor repos will be a heuristic (wrist keypoint within N pixels of a YOLO box, or an Ultralytics region/zone count), not a learned HOI model. 100DOH is the only learned off-the-shelf HOI detector, and its build friction (CUDA compile, old PyTorch) makes it unlikely in a hackathon.
- PYSKL is unmaintained and MMAction2's last push is 2026-03. Teams that pick ST-GCN will either fight mmcv installs or reimplement a small ST-GCN in plain PyTorch, as ORION appears to.

### Gaps
- There was no reliable star count for the most-copied "YOLOv8-pose + LSTM" repo. GitHub search sorted by stars returned only low-star results, which suggests no single dominant template beyond the Nicholas Renotte sign-language tutorial.
- I did not verify whether MMAction2 ships a live webcam skeleton demo script in the current release.
- The 100DOH weight/dataset licence remains unresolved (see above).

## Q3. Orientation-agnostic / microgravity pose and 3D human mesh recovery: what exists and how hard it is on edge hardware

### Takeaway
General HMR is well served by open code (4D-Humans/HMR2.0 1.7K stars, WHAM 1.1K, TokenHMR, ROMP, NLF, MMHuman3D). All of them depend on the SMPL/SMPL-X body model, whose licence is non-commercial and registration-gated. Frame-based HMR2.0 reaches about 65 fps on an RTX 3090, but WHAM is about 4.6 fps. No HMR repo was found that targets microgravity. Microgravity-specific resources are thin: MicroG-4M (action recognition, 8 stars, PySlowFast) and a few 2025-2026 journal papers on astronaut pose (Astro-Pose, H3-Astronaut Pose Net, AstroHSP) with no public code located. Any microgravity-specific component is therefore genuinely uncommon among competitors.

### Cited Findings
- **4D-Humans / HMR2.0** (`shubham-goel/4D-Humans`): 1,695 stars, MIT, last push 2026-02-07 (GitHub API). It depends on ViTDet/Detectron2 for detection and PHALP for tracking, and needs a manual SMPL neutral-model download. Training took "8 A100 GPUs for 7 days". The repo publishes no inference speed — [4D-Humans repo](https://github.com/shubham-goel/4D-Humans)
- **Throughput (RTX 3090):** HMR2.0 65.2 fps, CameraHMR 54.5 fps, WHAM 4.6 fps, VIBE 34.4 fps, POTTER 83.3 fps, DiffMesh 25.9 fps — [FastHMR, arXiv 2510.10868](https://arxiv.org/html/2510.10868v1). WHAM is slow because of "several heavy submodules, including ViTPose ..., the HMR2.0 transformer backbone, DPVO as the SLAM module, and two RNNs" — same source.
- **WHAM** (`yohanshin/WHAM`): 1,144 stars, MIT, last push 2024-04-18 (inactive). **TokenHMR** (`saidwivedi/TokenHMR`): 348 stars, last push 2025-10-03. **SMPL-X** (`vchoutas/smplx`): 2,717 stars. **MMHuman3D**: 1,426 stars, Apache-2.0, last push 2024-11. **NLF** (`isarandi/nlf`, NeurIPS 2024): 458 stars, MIT — GitHub API.
- **ROMP/BEV** (`Arthur151/ROMP`): 1,543 stars, Apache-2.0, last push 2024-11. It is a "one-stage method for monocular multi-person 3D mesh recovery in real time", with ONNX support "with faster inference speed on CPU/GPU" and a pip package (`simple-romp`) for Linux/Windows/Mac — [ROMP repo](https://github.com/Arthur151/ROMP). *This is the most edge-friendly HMR option found. No FPS figure was published on the page fetched.*
- More HMR repos (GitHub API, 2026-09-27):
  - `facebookresearch/sam-3d-body` (SAM 3D Body): 3,561 stars, last push 2026-02, custom licence (NOASSERTION).
  - `naver/multi-hmr`: 422 stars.
  - `zju3dv/EasyMocap`: 4,882 stars.
  - `vye16/slahmr`: 535 stars, MIT.
  - `bbossam/hmr_realtime` ("Realtime HMR based on NLF"): 0 stars.
  - *SAM 3D Body is the most visible newer "foundation" HMR a 2026 competitor could pick up. Its edge feasibility was not checked.*
- **EgoHOS** (`owenzlz/EgoHOS`, ECCV 2022 hand-object segmentation): 151 stars, MIT, last push 2024-02 — GitHub API.
- Another probable SIH competitor surfaced in search: `arnab9957/realtime-work-detection` ("Autonomous, offline, on-board Artificial Intelligence assistant designed to track, guide, and deterministically..."), pushed 2026-09-27 — GitHub search API.
- **SMPL licence:** use is for "non-commercial scientific research, non-commercial education, or non-commercial artistic projects"; commercial use is prohibited and licensed via Meshcapade — [SMPL model licence](https://smpl.is.tue.mpg.de/modellicense.html)
- **MicroG-4M** (ICLR 2026, arXiv 2506.02845), repo `lei-qi-233/MicroG-4M`: 8 stars, MIT, last push 2026-07-26. The dataset has 4,759 three-second clips from real ISS, Tiangong and EVA footage plus curated films, 50 action classes, 390K+ box-annotated frames, 13K+ action labels, 1,238 captions and 7K+ QA pairs. Tasks are multi-label action recognition, captioning and VQA. Baselines are SlowFast, X3D, InternVideo and GPT-4o. Code is built on PySlowFast. The dataset is on Hugging Face (`LEI-QI-233/MicroG-4M`). The README says fine-tuned weights are documented in BENCHMARK.md — [MicroG-4M repo](https://github.com/lei-qi-233/MicroG-4M), [HAR-in-Space repo](https://github.com/LEI-QI-233/HAR-in-Space), [arXiv 2506.02845](https://arxiv.org/abs/2506.02845). *The BENCHMARK.md URL returned 404 on fetch, so checkpoint availability is still unverified, which matches the ACTUS demo-plan's "loadability" risk.*
- **Astronaut pose research:**
  - *PhysAstro-Pose* (Sensors, 2026) is semi-supervised pose estimation on an "Astro-Pose dataset", where poses "exhibit arbitrary orientations and are often affected by severe occlusion and complex background interference, while annotated in-orbit data is scarce" — [DOI 10.3390/s26113406](https://doi.org/10.3390/s26113406) (search-snippet level only; full text returned 403).
  - *H3-Astronaut Pose Net* is multi-view voxel-based 3D astronaut pose for spacecraft (Neurocomputing 2025) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225015498).
  - *AstroHSP* is hybrid-supervision monocular astronaut pose, a two-stage domain-adaptive 2D pose plus uncertainty-aware 3D lifting — [PubMed 41886897](https://pubmed.ncbi.nlm.nih.gov/41886897/) (snippet only).
- **Orientation robustness framing from the literature:** "directly applying terrestrial models to microgravity environments reduces robustness to atypical body configurations" because astronauts show "inverted floating postures, arbitrary body rotations" — search snippet of [PhysAstro-Pose](https://doi.org/10.3390/s26113406).

### Inferences
- A competitor claiming "orientation-agnostic HMR" will almost certainly run stock HMR2.0/4D-Humans or ROMP on rotated frames. That is cloneable in a day on a desktop GPU, but it is SMPL-licence-encumbered and heavy on edge hardware: Detectron2 + ViT-H backbone for HMR2.0, and an unusable 4.6 fps for WHAM even on a 3090. No existing HMR checkpoint was trained on astronaut or inverted data, so none truly solves the "no up/down" problem.
- The astronaut-pose papers (Astro-Pose, AstroHSP, H3-APN) have no public code or data found. A competitor can cite them but not fork them, so microgravity-trained pose remains uncommoditised.
- MicroG-4M is the only public, openly licensed microgravity human-action resource with code. Its low star count (8) and PySlowFast dependency mean few teams will actually get it running, so a working MicroG-4M integration is rare, though anyone who reads the ICLR paper can cite the dataset.

### Gaps
- I found no public ISS / astronaut 3D pose dataset with downloadable data, and no ISS "procedure video" dataset with step annotations.
- No HMR model has published FPS on Jetson or laptop-class GPUs. ROMP has no FPS on the page fetched.
- The PhysAstro-Pose and AstroHSP full texts were blocked (403 or cookie wall), so dataset size and code availability are unconfirmed.
- MicroG-4M checkpoint URLs are unverified (BENCHMARK.md 404).

## Q4. Open-source components for the non-ML deliverables (TTS, streaming, GUI, logging): what is trivially replicable

### Takeaway
Every non-ML deliverable in the ISRO brief maps to a mature, pip- or binary-installable open-source component: pyttsx3 or Piper for offline voice, MediaMTX or GStreamer or aiortc (or plain FastAPI MJPEG) for stream-to-IP, OpenCV VideoWriter or MediaMTX recording for local storage, Streamlit or PyQt/PySide for the GUI, and JSON/JSONL or SQLite for logs. These are pure table stakes, and ORION already ships all of them.

### Cited Findings
- **pyttsx3** (`nateshmbhat/pyttsx3`): 2,534 stars, MPL-2.0, last push 2026-07-22 — GitHub API.
- **Piper**: the original `rhasspy/piper` (11,291 stars, MIT) is **archived**. Development moved to `OHF-Voice/piper1-gpl` (5,692 stars, GPL-3.0, pushed 2026-09-17) — GitHub API. It is "a fast and local neural text-to-speech engine that embeds espeak-ng", installs with `pip install piper-tts`, runs on Raspberry Pi, and is used by Home Assistant and NVDA — [piper1-gpl repo](https://github.com/OHF-Voice/piper1-gpl). *The licence changed from MIT to GPL-3.0.*
- **Coqui TTS**: `coqui-ai/TTS` has 46,070 stars and has been dormant since 2024-08-16 (MPL-2.0). The maintained fork `idiap/coqui-ai-TTS` has 2,329 stars, last push 2026-06-10 — GitHub API.
- **MediaMTX** (`bluenviron/mediamtx`): 20,279 stars, MIT, pushed 2026-09-26. It is a zero-dependency single binary for Linux, Windows and macOS that serves and ingests SRT, WebRTC, RTSP, RTMP, HLS, MPEG-TS and RTP, and "record[s] streams to disk in fMP4 or MPEG-TS format". It accepts FFmpeg, GStreamer, OBS and Python publishers — [MediaMTX repo](https://github.com/bluenviron/mediamtx). *One binary plus an FFmpeg pipe covers both "stream to IP" and "store locally".*
- **aiortc** (`aiortc/aiortc`): 5,106 stars, BSD-3, last push 2026-07-17 — GitHub API.
- RTSP templates: `foschmitz/yolo-python-rtsp` (169 stars, MIT, YOLO + OpenCV over RTSP) and `sahilparekh/GStreamer-Python` (132 stars, RTSP via GStreamer into NumPy) — GitHub search API.
- **Streamlit**: 45,838 stars, Apache-2.0. Ultralytics ships a ready-made "Live Inference with Streamlit" solution — GitHub API; [Ultralytics Solutions](https://docs.ultralytics.com/solutions/)
- **Competitor precedent:** ORION implements voice (macOS `say`/pyttsx3/espeak-ng), FastAPI + Uvicorn MJPEG streaming, async MP4 recording, a PySide6 "dark cockpit" GUI, and SQLite + structured JSON logging with Markdown reports — [Orion repo](https://github.com/amitraghvan/Orion)

### Inferences
- None of these deliverables can differentiate a team. They can only lose points if missing. ACTUS's own architecture doc already flags stream-to-IP and local storage as "0% implemented" but "no research risk". MediaMTX + FFmpeg (or OpenCV VideoWriter plus FastAPI MJPEG, as ORION does) closes that in hours.
- Licence detail worth noting: Piper's active fork is GPL-3.0. pyttsx3/espeak is the zero-friction offline choice.

### Gaps
- GStreamer and PyQt/PySide star counts were not pulled (not GitHub-hosted in the standard way, or not queried).
- I did not benchmark TTS latency.

## Q5. ACTUS features classified as commodity vs differentiated

### Takeaway
The perception building blocks ACTUS uses are commodity: YOLO COCO detection, MediaPipe/RTMPose pose, ST-GCN++, an FSM for skip/out-of-order, and the voice/log/GUI/stream deliverables. Competitors like ORION already have near-identical versions. ACTUS's defensible differentiation is at the integration and decision level: (1) MicroG-4M as a frozen microgravity evidence stream (rare, fragile, publicly citable but almost never run), (2) the two-channel belief split (prior-free evidence for skip detection vs prior-informed posterior for next-step prompts), (3) a calibrated decision layer with an abstain/"ask the astronaut" path, and (4) an orientation test on real astronaut footage. The 4-rotation pose trick and skeleton canonicalisation are cheap and easy to copy once seen. They make a strong demo point but are not a moat.

### Cited Findings (basis for the classification)
- The commodity stack is proven in a competitor: YOLO11n-pose + ST-GCN + ByteTrack + 11-state FSM + offline TTS + MJPEG + MP4 + PySide6 + JSON/SQLite — [Orion](https://github.com/amitraghvan/Orion)
- Keypoint-sequence classifiers (LSTM, ST-GCN++, PoseC3D) with pretrained NTU weights are freely available — [PYSKL](https://github.com/kennymckormick/pyskl), [MMAction2 skeleton zoo](https://mmaction2.readthedocs.io/en/dev-1.x/model_zoo/skeleton.html), [sign-language LSTM tutorial repo, 503 stars](https://github.com/nicknochnack/ActionDetectionforSignLanguage)
- Rule-based joint-angle logic is a first-party Ultralytics feature (Workouts Monitoring / AI Gym) — [Ultralytics Solutions](https://docs.ultralytics.com/solutions/)
- Online mistake detection against a procedure exists in research code (PREGO: online recognition + LLM next-step prediction; differentiable task graphs), but only in low-star, dataset-specific repos — [review 2510.19292](https://arxiv.org/html/2510.19292), [PREGO](https://github.com/aleflabo/PREGO)
- The microgravity action resource is MicroG-4M (8 stars, PySlowFast, checkpoints unverified) — [MicroG-4M](https://github.com/lei-qi-233/MicroG-4M)
- Astronaut-pose papers have no public code found — [PhysAstro-Pose](https://doi.org/10.3390/s26113406), [H3-Astronaut Pose Net](https://www.sciencedirect.com/science/article/abs/pii/S0925231225015498)

### Inferences: classification table (ACTUS features from stage-3-architecture.md and demo-plan.md)

| ACTUS feature | Class | Why |
|---|---|---|
| Stock COCO YOLO object/person detection | Commodity | Ultralytics, 62K stars. Every competitor has it. |
| MediaPipe / RTMPose 2D pose | Commodity | 37K-star MediaPipe, MMPose. YOLO-pose used by competitors. |
| ST-GCN++ step classifier (deferred to "day 3") | Commodity | MMAction2/PYSKL checkpoints. ORION already runs ST-GCN. |
| Rule-based step logic over joint angles and wrist-object distance | Commodity | Ultralytics AI Gym / zone / distance solutions. Standard hackathon heuristic. |
| FSM sequencer: next-step prompt, skip and out-of-order alerts | Commodity | ORION's 11-state FSM. Trivial given a fixed step list. |
| Voice alerts (offline TTS) | Commodity | pyttsx3 / espeak / `say` / Piper. |
| JSONL timestamped log with outcomes | Commodity | ORION: JSON + SQLite + Markdown reports. |
| Stream to IP + local video store (not yet built in ACTUS) | Commodity, and a table-stakes gap | MediaMTX single binary, FastAPI MJPEG, OpenCV VideoWriter. |
| Monitoring GUI | Commodity | PySide6/Streamlit. ORION's "cockpit" GUI is likely more polished. |
| Frame-index-keyed cache, bounded voice queue, dedup | Engineering hygiene, not visible differentiation | Matters for demo quality ("sounds intelligent vs broken") but judges won't credit it. |
| Confidence column / per-step confidence in log | Low-level differentiation | Easy to copy, but most templates show only labels. |
| **4-rotation pose selection (0/90/180/270 by mean landmark confidence)** | Easy to copy, rarely done | It's test-time augmentation, a few lines once seen. No competitor repo surveyed advertises it. Its value is the **rotated astronaut-clip demo**. |
| **Skeleton normalisation / torso canonicalisation + retained raw torso angle ("relative to rack")** | Modest differentiation | Centring and scale-normalising are standard preprocessing. Explicitly keeping the pre-canonical torso vector as a separate "payload-relative" channel addresses ISRO's wording directly, and no repo surveyed does it. |
| **Orientation-agnostic claim without SMPL/HMR** | Differentiated positioning | Competitors going the HMR route inherit SMPL non-commercial licensing and heavy compute (WHAM 4.6 fps on a 3090). ACTUS's cheaper route can be argued as edge-appropriate. Risk: judges may reward teams that literally show a mesh. |
| **MicroG-4M frozen evidence stream (with an honest substitute fallback)** | **Most differentiated** | The only open microgravity HAR resource. 8 stars. PySlowFast friction. Checkpoints unverified. Few teams will run it; its differentiation value disappears if the substitute (Kinetics VideoMAE/TimeSformer) is what actually ships. |
| **Two-channel belief (prior-free C1 for skip detection vs prior-informed C2 for prompts)** | **Differentiated** | Addresses a real failure mode (the protocol prior masking a skipped step) that template FSMs ignore. Conceptually related to task-graph and PREGO research but not present in hackathon-grade repos. |
| **Decision layer: dwell / hysteresis / min-confidence gate / abstain → "ask" yes/no** | **Differentiated** | Human-in-the-loop confirmation is not in any template or competitor repo surveyed. It turns perception uncertainty into designed behaviour. HoloAssist's "intervention" framing is the closest academic analogue. |
| **Real ISS/astronaut clip run end-to-end + synthetic rotation test** | Differentiated demo evidence | No public astronaut procedure dataset exists. Showing real orbit footage is rare. N=1 must be disclosed. |
| Honesty/limitations list | Soft differentiator | Credibility with ISRO judges. |

- Net: roughly 60 to 70% of ACTUS's visible surface area (detector, pose, FSM, voice, log, GUI, stream) is commodity and matches what a well-executed competitor like ORION already shows. ACTUS's edge rests on the microgravity-aware reasoning layer (MicroG-4M, orientation handling on real orbit footage) and the uncertainty-aware decision logic (two channels plus abstain/ask). The pitch should lead with those, not with the pipeline.
- Risk: if MicroG-4M fails to load and the substitute is used, the most distinctive stream becomes commodity (Kinetics models via `transformers`). The decision layer and the orientation demo then carry the whole differentiation story.

### Gaps
- The classification of competitor features depends on the competitor-repo catalogue being compiled by other researchers. Only ORION and one other competitor repo were sampled here.
- I could not find a star-ranked "procedure step validation" template repo, so the absence of the two-channel or abstain pattern in open source is based on the repos and review surveyed, not an exhaustive search.
