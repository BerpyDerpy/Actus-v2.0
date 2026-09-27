# GitHub repos from competing SIH teams on ISRO PS 26174 ("AI Human Activity Recognition for On-board BAS Experiments"), found by ID and keyword search

Method (so the writer can judge coverage). Research date is 2026-09-27. The `gh` CLI was not installed and no GitHub token was available, so every search used the unauthenticated GitHub REST repository-search API (`/search/repositories`). About 95 queries ran, each in two modes: the default (name/description/topics) and `in:readme`. The query list covered SIH26174, "SIH 26174", 26174, SIH16174, 16174, SIH25174, 25174, SIH1674, SIH174, SIH2026174, 2026174, PS26174, "PS-26174", "PS 26174", "HAR BAS", "BAS HAR", "BAS-HAR", "ISRO HAR", "SIH HAR", "SIH 2026 HAR", "BAS experiment(s)", "on-board BAS", "onboard BAS", "Bharatiya Antariksh Station", Antariksh, Antriksh, "SIH 2025/2026 ISRO", "Smart India Hackathon ISRO / human activity recognition", "astronaut activity recognition", "astronaut experiment", "HAR astronaut", "BAS assistant/copilot", "astronaut assistant/copilot", "HAR for BAS", "BAS-EXP" and a few more. The raw result TSVs are in the session scratchpad. Web search (WebSearch) was used as a cross-check.

GitHub *code* search needs authentication, so it was **not** run; the README-mode repository search partly makes up for this. Every candidate repo (about 100) was shallow-cloned, and its file tree, `requirements.txt` and key source files were grepped to check README claims. "Verified in code" below means the dependency or the call was actually found in the repo. It does not mean the code was executed.

## Which public GitHub repos are tied to PS 26174 via ID/hackathon keyword searches?

### Takeaway
The problem statement exists only in SIH 2026, as **SIH26174**. It is not an older statement that has been reused. About **75 public repos** belong to competing teams (Aug 23 – Sep 27, 2026), plus about 15 more that are near-duplicates or member copies. Almost none has more than 5 stars. `SIH16174` and `SIH25174` return nothing relevant through the API. **25174 was a different ISRO problem in SIH 2025 (fish-catch identification)**, so hits on "25174" are false positives. The most productive queries were `SIH26174`/`26174` in README, "on-board BAS", "Bharatiya Antariksh Station", "astronaut activity recognition", `PS-26174` and `SIH2026174`.

### Cited Findings

**ID mapping and query yield**
- The SIH 2026 PS list gives S.No. 174 = **SIH26174**, "AI Human Activity Recognition for On-board BAS Experiments", ISRO, Software, Space Technology. The neighbours are SIH26173 (iTantra TTS/STT radio) and SIH26175 (DepthWizard). — [vedantchalke36/sih-2026-problem-statements](https://github.com/vedantchalke36/sih-2026-problem-statements/tree/main/ps_2026)
- API yields (total_count; default mode / in:readme mode): SIH26174 10/30; "SIH 26174" 10/11; 26174 39/145 (mostly noise); **SIH16174 0/0**; 16174 57/214 (all noise); **SIH25174 0/1** (the one hit is a fish-net repo); SIH1674 2/1 (SIH 2024 repos); SIH2026174 1/2; PS26174 1/1; "PS-26174" 2/17; "Bharatiya Antariksh Station" 11/85; "HAR BAS" mostly WiFi-CSI HAR, BASIC-language and Swedish "har" noise. — GitHub search API, run 2026-09-27 (scratchpad logs)
- "25174" belongs to SIH 2025's ISRO fish-detection PS: "Winning Project For SIH'25 ISRO PS: 25174" — [ashish-um/fish-sih](https://github.com/ashish-um/fish-sih); likewise [codinsayan/SIH-Fish-Detection-and-Analysis](https://github.com/codinsayan/SIH-Fish-Detection-and-Analysis), [nikhilphr/Fish-Catch-Identification-Biomass-Estimation-System](https://github.com/nikhilphr/Fish-Catch-Identification-Biomass-Estimation-System), [Aryan-Singh1729/SIH-2025-Fish](https://github.com/Aryan-Singh1729/SIH-2025-Fish)
- "SIH1674" hits are SIH 2024 projects ([v9pt/SIH1674](https://github.com/v9pt/SIH1674), [PROstudent009/SIH1674_CyberMafias](https://github.com/PROstudent009/SIH1674_CyberMafias), [Abd2301/project-NARC](https://github.com/Abd2301/project-NARC)), not this PS.
- The team name "Hangla Coders" uses the ID variant **SIH2026174**: [MakInShort/Hangla-Coders-SIH2026174](https://github.com/MakInShort/Hangla-Coders-SIH2026174).
- **Our own repo is publicly discoverable.** [BerpyDerpy/Actus](https://github.com/BerpyDerpy/Actus) (created 2026-09-20, pushed 2026-09-27, Python) comes up for the query "astronaut activity recognition in:readme". — GitHub search API

**Master catalog: competitor repos (stars/forks | created → last pushed | primary language)**. Maturity tiers: A = substantive working pipeline, B = small working prototype, C = README-only, placeholder or mock.

| Repo | ★/forks | Created → pushed | Lang | Tier | Matched via |
|---|---|---|---|---|---|
| [amitraghvan/Orion](https://github.com/amitraghvan/Orion) | 0/0 | 09-06 → 09-20 | Python | A | "Bharatiya Antariksh Station", "ISRO HAR", "HAR astronaut" |
| [arnab9957/realtime-work-detection](https://github.com/arnab9957/realtime-work-detection) | 2/2 | 09-09 → 09-27 | Python | A | "BAS experiment", "PS-26174", "26174 in:readme" |
| [Pranvkumar/AI-HAR-For-BAS](https://github.com/Pranvkumar/AI-HAR-For-BAS) | 0/0 | 08-23 → 09-07 | Python | A | "BAS-HAR", "HAR astronaut", "SIH 26174 in:readme" |
| [MakInShort/Hangla-Coders-SIH2026174](https://github.com/MakInShort/Hangla-Coders-SIH2026174) | 4/5 | 08-27 → 09-03 | Python | A | SIH2026174, "BAS experiment" |
| ↳ copy: [yash2744/AI---Human---Activity---Recognition-for-Bas-Experiment-SIH-2026-](https://github.com/yash2744/AI---Human---Activity---Recognition-for-Bas-Experiment-SIH-2026-) | 0/0 | 09-25 → 09-25 | Python | A (dup) | web search, SIH2026174 in:readme |
| ↳ copy: [aceholland/space_frontend_dashboard](https://github.com/aceholland/space_frontend_dashboard) | 0/0 | 09-01 | Python | A (dup) | "HAR astronaut in:readme" |
| [na124441/ASTRA](https://github.com/na124441/ASTRA) | 2/2 | 09-05 → 09-16 | Python | A | "SIH 26174 in:readme", "Bharatiya Antariksh Station" |
| [gsaivenkateshkumar/ASTRA-HAR-EdgeAI](https://github.com/gsaivenkateshkumar/ASTRA-HAR-EdgeAI) | 0/0 | 08-29 → 09-06 | Python | A | SIH26174, "ISRO HAR" |
| [FAILEDdokja/HAR-system](https://github.com/FAILEDdokja/HAR-system) | 1/1 | 09-02 → 09-03 | Python | A | "SIH26174 in:readme", "ISRO HAR" |
| [atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174) | 4/0 | 09-16 → 09-26 | Python | A | 26174, "BAS experiment" |
| [Shre-Sat/BASGUARD](https://github.com/Shre-Sat/BASGUARD) | 0/0 | 09-10 → 09-26 | Python | A | 26174, PS26174 in:readme |
| [krishnakankane/isro-bas-har](https://github.com/krishnakankane/isro-bas-har) | 0/0 | 09-02 → 09-11 | Python | A | "HAR BAS", "BAS-HAR", "ISRO HAR" |
| [Rahulrwt99/bas-experiment-assistant](https://github.com/Rahulrwt99/bas-experiment-assistant) | 0/0 | 09-02 | Python | A− | 26174, "Bharatiya Antariksh Station" |
| [Monishwarann/SIH](https://github.com/Monishwarann/SIH) ("ASTRA-HAR") | 0/0 | 09-09 → 09-23 | Python | A (metrics fabricated) | "BAS HAR", "PS-26174" |
| [subish345/Astra](https://github.com/subish345/Astra) ("ASTRA-EA") / [subish345/orbitademo](https://github.com/subish345/orbitademo) ("ORBITA") / [subish345/AstroActAi](https://github.com/subish345/AstroActAi) | 0/0 | 09-09 → 09-22 | Python | B (large, fake weights) | "SIH26174 in:readme" |
| [MrRouhin/liquid-mix-ai-experiment](https://github.com/MrRouhin/liquid-mix-ai-experiment) | 0/1 | 09-10 → 09-15 | Python | A− | "PS-26174", "26174 in:readme" |
| [patilbhishmesh/cv-project-26174](https://github.com/patilbhishmesh/cv-project-26174) | 0/0 | 09-17 | Python | A− | 26174 |
| [samyuthasankar07-ctrl/astra](https://github.com/samyuthasankar07-ctrl/astra) ("ASTRA-SENSE") | 0/0 | 09-20 → 09-23 | Python | B+ | "on-board BAS in:readme" |
| [Nimbus3001/KRAMA](https://github.com/Nimbus3001/KRAMA) | 0/0 | 09-24 | Python | B | "on-board BAS" |
| [Kartikey2006/Astra](https://github.com/Kartikey2006/Astra) | 0/0 | 08-28 | Python | B | "astronaut activity recognition" |
| [dnvrmhra/Hands-Object-Integration](https://github.com/dnvrmhra/Hands-Object-Integration) (+ copy [WaseemAkam/please](https://github.com/WaseemAkam/please)) and [dnvrmhra/ASTRA-HAR](https://github.com/dnvrmhra/ASTRA-HAR) (frontend) — Team UnoFlyp | 0/1 | 09-06 → 09-20 | Py/TS | B | "PS-26174", "ISRO HAR" |
| [ishwaryajuttu-gif/SIH_2026](https://github.com/ishwaryajuttu-gif/SIH_2026) ("BAS-HAR") | 0/0 | 09-26 | Python | B | "SIH 2026 HAR in:readme" |
| [Thiru-Lokesh-24/cv_module](https://github.com/Thiru-Lokesh-24/cv_module) | 0/0 | 09-18 | Python | B (CV module only) | 26174 |
| [Dynamo-SJoshi/VisionX_SIHproject](https://github.com/Dynamo-SJoshi/VisionX_SIHproject) | 0/0 | 08-27 → 09-07 | Python | B | "on-board BAS in:readme" |
| [bhaveshnerkar/SpaceAssist](https://github.com/bhaveshnerkar/SpaceAssist) | 0/0 | 09-24 → 09-25 | JS/Py | B | "26174 in:readme" |
| [h1e2l3l4o5/ORBIT-HAR](https://github.com/h1e2l3l4o5/ORBIT-HAR) | 1/1 | 09-21 | Python | B | "26174 in:readme" |
| [dhanya-s-coder/sih26174](https://github.com/dhanya-s-coder/sih26174) (+ copies [/sih](https://github.com/dhanya-s-coder/sih), [/sih2](https://github.com/dhanya-s-coder/sih2)) | 0/0 | 09-25 → 09-26 | Python | B | SIH26174 |
| [moiz2405/ASTRA-SEQ](https://github.com/moiz2405/ASTRA-SEQ) | 0/0 | 09-04 | Python | B | SIH26174 |
| [ParulDoshi/Byte_Brahmaaaas](https://github.com/ParulDoshi/Byte_Brahmaaaas) / [ParulDoshi/Byte_Brahmas](https://github.com/ParulDoshi/Byte_Brahmas) | 0/0 | 09-12; 09-26 | Python | B | SIH26174, "SIH HAR" |
| [Paru22/SIH](https://github.com/Paru22/SIH) | 0/0 | 09-04 → 09-19 | Python | B | "on-board BAS in:readme" |
| [notaditya28/isro_har_sih_project](https://github.com/notaditya28/isro_har_sih_project) | 0/0 | 09-11 | Python | B | "on-board BAS in:readme" |
| [faiisalaziz/sih_space_har](https://github.com/faiisalaziz/sih_space_har) ("AstroGuard") | 0/0 | 09-14 | Python | B | "astronaut activity recognition in:readme" |
| [aaryan2203/AI-BASED-HAS](https://github.com/aaryan2203/AI-BASED-HAS) and [Sheen11-ing/AI-based-Human-Activity-Recognition-BAS-](https://github.com/Sheen11-ing/AI-based-Human-Activity-Recognition-BAS-) (same "BAS-HAR" README template) | 1/0; 0/0 | 09-25; 09-03 | Python | B | "HAR BAS in:readme", "astronaut activity recognition" |
| [Ayush-sharma-mx/bas-ai-monitor](https://github.com/Ayush-sharma-mx/bas-ai-monitor) | 0/0 | 08-26 → 08-27 | Python | B | "on-board BAS in:readme" |
| [aayushjadhav05128-sudo/bas-experiment-copilot](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot) + [samaksh-divs/bas-experiment-copilot-frontend](https://github.com/samaksh-divs/bas-experiment-copilot-frontend) | 0/0 | 09-05; 09-14 | Py/TS | B | "HAR BAS in:readme", "BAS copilot" |
| [HarshAudichya/T-A-R-A-for-mac-with-audio-alert](https://github.com/HarshAudichya/T-A-R-A-for-mac-with-audio-alert) ("TARA") | 0/0 | 09-11 → 09-13 | Python | B | "ISRO HAR in:readme" |
| [StarCodes03/TARA-Astronaut-Assistant](https://github.com/StarCodes03/TARA-Astronaut-Assistant) | 0/0 | 09-06 | Python | B | "SIH26174 in:readme" |
| [kavyareddypallala-cell/BAS-Vision](https://github.com/kavyareddypallala-cell/BAS-Vision) ("ORBITA") | 1/0 | 09-09 | Python | B | "on-board BAS in:readme" |
| [yashkhochare-79/HAR_satellite](https://github.com/yashkhochare-79/HAR_satellite) | 0/0 | 09-26 | Python | B | "on-board BAS in:readme" |
| [PriyankShingala/Stellar](https://github.com/PriyankShingala/Stellar) | 0/0 | 09-10 → 09-11 | Python | B | "on-board BAS" |
| [chaitanyareddykarri/AI-Human-Activity-Recognition-for-On-board-BAS-](https://github.com/chaitanyareddykarri/AI-Human-Activity-Recognition-for-On-board-BAS-) ("BAS-AI") | 0/0 | 09-05 | Python | B | "SIH26174 in:readme" |
| [sangsaptak0704/KriyaSense](https://github.com/sangsaptak0704/KriyaSense) (Team Astrocrew) | 2/2 | 08-30 → 09-11 | JS/Py | B | "BAS experiment", "BAS HAR in:readme" |
| [garvit000/procsight](https://github.com/garvit000/procsight) | 0/0 | 08-25 | Python | B | "on-board BAS in:readme" |
| [Himashwetha777/SIH26174-Astronaut-HAR-System](https://github.com/Himashwetha777/SIH26174-Astronaut-HAR-System) (+ [ASTRA-HAR-System](https://github.com/Himashwetha777/ASTRA-HAR-System), empty [ASTRA-HAR-Deployment](https://github.com/Himashwetha777/ASTRA-HAR-Deployment)) | 0/1 | 09-13 | Python | B | SIH26174 |
| [swatikrishnamurthy24-dotcom/AI-ASSISTANCE-ISRO-](https://github.com/swatikrishnamurthy24-dotcom/AI-ASSISTANCE-ISRO-) | 0/0 | 09-20 | Python | B | "SIH26174 in:readme" |
| [ThunderBolt975/KRIYA-SIH-2026-PS26174-ISRO](https://github.com/ThunderBolt975/KRIYA-SIH-2026-PS26174-ISRO) (+ near-identical [lalwanisneha00/isro-on-BAS-AI-human-activity-recognisation](https://github.com/lalwanisneha00/isro-on-BAS-AI-human-activity-recognisation)) | 0/0 | 09-08 → 09-18 | Python | B | PS26174, 26174 |
| [ThouheedAhmed/DRISHTI-Edge-AI](https://github.com/ThouheedAhmed/DRISHTI-Edge-AI) | 0/0 | 09-12 → 09-13 | Python | B | "SIH26174 in:readme" |
| [k-a-v-i-n-0-0-2/hawk-bas](https://github.com/k-a-v-i-n-0-0-2/hawk-bas) ("HawkAI") | 0/0 | 08-29 | Python | B | "PS-26174", "26174 in:readme" |
| [Prithivi2005-raj/space-experiment-astra](https://github.com/Prithivi2005-raj/space-experiment-astra) | 0/0 | 09-10 → 09-13 | Py/TS | B | "26174 in:readme" |
| [madhurgitbub/ISRO](https://github.com/madhurgitbub/ISRO) | 0/0 | 09-17 → 09-20 | Python | B | "26174 in:readme" |
| [sohamjachak003/AI-BAS-Assistant_2](https://github.com/sohamjachak003/AI-BAS-Assistant_2) | 0/0 | 09-12 | Python | B | "Bharatiya Antariksh Station" |
| [David4988/Orbit-Assist-SIH26174](https://github.com/David4988/Orbit-Assist-SIH26174) | 0/0 | 08-31 → 09-01 | JS/Py | B | SIH26174 |
| [yamantaka-singh/BAS](https://github.com/yamantaka-singh/BAS) | 0/0 | 09-04 | Python | B | "on-board BAS in:readme" |
| [Sachin-Malviya/ASTRA-BAS-AI](https://github.com/Sachin-Malviya/ASTRA-BAS-AI) | 0/0 | 09-20 | Python | B− | "on-board BAS in:readme" |
| [sighkalp/EKA-Edge-Kognitive-Assistant](https://github.com/sighkalp/EKA-Edge-Kognitive-Assistant) | 0/0 | 08-28 → 09-03 | Python | B− | "astronaut activity recognition in:readme" |
| [antas06/exp-ass-bas](https://github.com/antas06/exp-ass-bas) | 0/0 | 09-06 → 09-14 | Python | B− | "on-board BAS in:readme" |
| [ArrushiTripathi2429/Space-Sight](https://github.com/ArrushiTripathi2429/Space-Sight) / [Space-Sight2](https://github.com/ArrushiTripathi2429/Space-Sight2) | 0/1; 0/0 | 09-11 → 09-14 | Python | B− | "HAR astronaut in:readme", "on-board BAS" |
| [haasitha1603/NAVIKSHA](https://github.com/haasitha1603/NAVIKSHA) | 0/0 | 09-11 → 09-17 | JS/Py | B− (HSV heuristics) | SIH26174 |
| [Shreyansh-DevHub/AI-HAR-_On-board_BAS_Experiment](https://github.com/Shreyansh-DevHub/AI-HAR-_On-board_BAS_Experiment) ("AI-HAR+") | 0/0 | 09-26 | — | B− (README+screenshot only in git) | "on-board BAS" |
| [DarpanSakpal/HUMAN-ACTIVITY-RECOGNITION](https://github.com/DarpanSakpal/HUMAN-ACTIVITY-RECOGNITION) | 0/0 | 09-02 → 09-03 | Python | B− (generic HAR) | SIH26174 |
| [yashik22-ai/BASE](https://github.com/yashik22-ai/BASE) ("ActiNexus") | 0/0 | 09-09 | JS | B− | "SIH26174 in:readme" |
| [Ozair-aly/orbital-hai](https://github.com/Ozair-aly/orbital-hai) ("ORBITAL") | 0/0 | 09-19 | TS/Py | B− (IMU, simulated) | "SIH26174 in:readme" |
| [rudra00434/HOMOSENSYS-AI-Assistant](https://github.com/rudra00434/HOMOSENSYS-AI-Assistant) (+ GitHub fork [Soubhagya-Kabiraj/HOMOSENSYS-AI-Assistant](https://github.com/Soubhagya-Kabiraj/HOMOSENSYS-AI-Assistant)) | **5/6** (most-starred) | 09-06 → 09-08 | Python | C (mock CV) | "BAS HAR in:readme", "BAS experiment" |
| [hamsadevang849-sys/AI-Human-Activity-Recognition-for-On-board-BAS-Experiments](https://github.com/hamsadevang849-sys/AI-Human-Activity-Recognition-for-On-board-BAS-Experiments) | 0/0 | 09-25 → 09-27 | HTML/Py | C (simulated) | "HAR astronaut", "Bharatiya Antariksh Station in:readme" |
| [Omkar96-18/SIH26174_BAS_AgentByXathonerTeam](https://github.com/Omkar96-18/SIH26174_BAS_AgentByXathonerTeam) | 0/0 | 09-26 | — | C (README+LICENSE only) | SIH26174 |
| [xarjunpatil/SIH26174-AI-Human-Activity-Recognition-for-On-board-BAS-Experiments](https://github.com/xarjunpatil/SIH26174-AI-Human-Activity-Recognition-for-On-board-BAS-Experiments) | 0/0 | 08-31 | HTML | C | SIH26174 |
| [Vynex-Labs/AI-Human-Activity-Recognition-for-On-Board-BAS-Experiments](https://github.com/Vynex-Labs/AI-Human-Activity-Recognition-for-On-Board-BAS-Experiments) | 0/0 | 08-31 | — | C (title-only README) | "on-board BAS" |
| [Priyanka07110/ORION](https://github.com/Priyanka07110/ORION) | 0/0 | 09-03 | — | C (README only) | "SIH26174 in:readme" |
| [Krishna105-ui/Astronaut-Activity-Recognition-System](https://github.com/Krishna105-ui/Astronaut-Activity-Recognition-System) ("AEGIS · AI-HAR") | 0/0 | 08-24 | — | C (README + zip) | "26174 in:readme" |
| [Aryasurya12/AstroFlow-AI](https://github.com/Aryasurya12/AstroFlow-AI) | 2/0 | 08-28 → 09-12 | — | C (10 .md files) | "BAS experiments", "Bharatiya Antariksh Station" |
| [dnvrmhra/ASTRA-HAR](https://github.com/dnvrmhra/ASTRA-HAR), [gurnoor88888-svg/orbit-senses](https://github.com/gurnoor88888-svg/orbit-senses), [SyroSyrup98/ORBIT-HAR](https://github.com/SyroSyrup98/ORBIT-HAR), [sohamjachak003/AI-BAS-Assistant](https://github.com/sohamjachak003/AI-BAS-Assistant), [Shrishti08raghav/Viksit-Coders-SIH2026](https://github.com/Shrishti08raghav/Viksit-Coders-SIH2026) | 0 | 08-25 → 09-26 | TS/JS | C (frontend/dashboard demos, simulated data) | various |
| Empty/stub: [Kshitij0275/BAS-Mission-Control](https://github.com/Kshitij0275/BAS-Mission-Control), [nonugit/An-AI-based-HAR-system](https://github.com/nonugit/An-AI-based-HAR-system), [nonugit/Frontend-for-AI-HUMAN-ACTIVITY](https://github.com/nonugit/Frontend-for-AI-HUMAN-ACTIVITY), [deepakbareth01/astra](https://github.com/deepakbareth01/astra), [baradaditya17-creator/AI-BAS-Assistant](https://github.com/baradaditya17-creator/AI-BAS-Assistant) (Google AI Studio stub) | 0 | 08-26 → 09-11 | — | C | "Bharatiya Antariksh Station", "on-board BAS" |

(All dates are 2026. Stars, forks and dates come from the search API on 2026-09-27; the Soubhagya fork's metadata came from `/repos` and shows fork=true, parent rudra00434.)

**False positives, checked and discarded**
- [2KRISHNAYADAV/Bharatiya-Antariksh-Hackathon-2025](https://github.com/2KRISHNAYADAV/Bharatiya-Antariksh-Hackathon-2025): Bharatiya Antariksh Hackathon 2025, Halo-CME detection on Aditya-L1 (a different hackathon). The many other "Antariksh"/"Bharatiya Antariksh Station in:readme" hits are also BAH projects (AQI, lunar ice, cloud removal, urban heat).
- MAITRI astronaut-wellbeing repos (SIH25175, a different PS): [tanubuddhibhargav-arch/SIH](https://github.com/tanubuddhibhargav-arch/SIH), [pnukadas-cloud/SIH](https://github.com/pnukadas-cloud/SIH), [RahulGodara14/MAITRI-...](https://github.com/RahulGodara14/MAITRI-Mission-Aware-Intelligent-Therapeutic-Response-Interface-).
- [Arkz-Deepak/sih2026-isro-space-tech](https://github.com/Arkz-Deepak/sih2026-isro-space-tech) (SIH26226 debris nano-satellite); [bhuvanthodeti75-ai/sih2026hackthon-isroStatement-](https://github.com/bhuvanthodeti75-ai/sih2026hackthon-isroStatement-) (satellite frame interpolation); [adot-7/BAS](https://github.com/adot-7/BAS) (BAS stowage management, 2025).
- Wearable/IMU "space-inspired" HAR with no SIH ID: [JaiPrakashVI/AI-Human-Activity-Recognition-Space](https://github.com/JaiPrakashVI/AI-Human-Activity-Recognition-Space) and its twin [idhayathulla-dev/HAR_System](https://github.com/idhayathulla-dev/HAR_System) (HMM/IMU, possibly PS-inspired but not vision-based); [Ariaeimehr/Zero-G-Space-HAR](https://github.com/Ariaeimehr/Zero-G-Space-HAR) (federated wearable transformers); [vihaanpatel2507-sudo/main](https://github.com/vihaanpatel2507-sudo/main) (generic space-station motion monitor; borderline).
- Borderline and not inspected: two late hits on "experiment protocol astronaut in:readme". [AjitKumar166/sih26-project](https://github.com/AjitKumar166/sih26-project) is an "Offline Astronaut Assistant System … Microgravity Experiment Protocol Companion (SIH-2026)" built around offline NLU/LLM, so it may be a different PS. [Yatharth-Saxena/Aethon-AI](https://github.com/Yatharth-Saxena/Aethon-AI) is "AETHON — Spaceflight AI Mission Control & Payload Assembly". Only their READMEs were skimmed via raw.githubusercontent.com.
- PS-list aggregators: [NoBugNinja/Smart-India-Hackathon-SIH-2026-Problem-Statements](https://github.com/NoBugNinja/Smart-India-Hackathon-SIH-2026-Problem-Statements) (30★), [Rajkumar-Porandla/SIH-2026-Problem-Statements](https://github.com/Rajkumar-Porandla/SIH-2026-Problem-Statements), [AryanSahu321/sih](https://github.com/AryanSahu321/sih), [subhadipmondal99/t](https://github.com/subhadipmondal99/t), [Abhinav-Prabhakar/SIH2026](https://github.com/Abhinav-Prabhakar/SIH2026) (its SIH26174 folder holds only a `.md` description).
- Profile READMEs that mention SIH26174 (team members): Omkar96-18/Omkar96-18, Soubhagya-Kabiraj/Soubhagya-Kabiraj, yamantaka-singh/yamantaka-singh.
- "HAR BAS"/"BAS HAR"/"BAS experiment"/"BAS-EXP"/"BAS copilot" also return WiFi-CSI HAR, realized-volatility HAR models, BASIC-language repos, SAP BAS tools and Swedish-language repos. None are relevant.

### Inferences
- The field is **large (~75 teams publicly) and almost entirely unstarred and recent** (created in the last ~5 weeks). GitHub stars carry no signal about quality; code inspection is the only way to judge a repo.
- Several teams keep **multiple copies** of the same project: Hangla-Coders ×3, dhanya-s-coder ×3, subish345 ×3, Himashwetha777 ×3, dnvrmhra/WaseemAkam ×2, ThunderBolt975/lalwanisneha00 ×2, HOMOSENSYS ×2, ParulDoshi ×2. The number of distinct teams is therefore closer to about 60 than 90.
- Our own public repo can be read by competitors through the same searches. If that is not intended, the team should consider making it private until the submission date.

### Gaps
- GitHub **code search** (`/search/code`) needs authentication and could not be run. Repos that mention 26174 only inside source files, with no README or description mention, may have been missed.
- The user reports that "SIH16174" and "16174" succeeded in the GitHub web UI, but the API returned 0 relevant results for SIH16174. The web UI's fuzzy matching behaves differently, and the web results could not be reproduced without a logged-in session.
- Repo search returns at most 100 items per query. For high-noise queries ("16174 in:readme", "BAS assistant", "Antariksh") only the first page was reviewed.
- Private repos, GitLab/Drive-hosted code and repos with unrelated names and empty READMEs cannot be found this way.

## Technical profile of each competitor repo (detection, pose, HOI, classifier, sequencing, TTS, logs, streaming, GUI, hardware, dataset, HMR, accuracy)

### Takeaway
The **dominant competitor design** is: webcam → YOLOv8n/YOLO11n (mostly COCO-pretrained) + MediaPipe Pose/Hands → rule-based hand-object proximity → a hand-written JSON/YAML **FSM** → `pyttsx3` voice → JSONL/CSV/SQLite logs → **Flask/FastAPI MJPEG** "streaming" → FastAPI + React or PySide6 GUI. That is roughly the same skeleton as ACTUS. Only a handful of teams trained anything:
- Orion: ST-GCN.
- Hangla Coders: XGBoost step classifier.
- Pranvkumar: YOLO + gesture model.
- na124441: MobileNetV3-LSTM + a MicroG-4M baseline.
- gsaivenkateshkumar, MrRouhin, patilbhishmesh and samyuthasankar: custom YOLO detectors.
- krishnakankane: Conv1D-BiGRU.

Only **one team (arnab9957)** wires in real 3D HMR (Multi-HMR, offline). Only **na124441** uses **MicroG-4M**, the astronaut dataset ACTUS relies on.

### Cited Findings

**Tier A: substantive pipelines (verified in code)**

- **amitraghvan/Orion ("ORION")**
  - Pipeline: YOLO11n detection + **YOLO11n-pose** (17 COCO joints) + ByteTrack with Kalman velocity. HOI contact states are APPROACH/TOUCH/MANIPULATE/RELEASE. **ST-GCN** runs over 32-frame skeletal windows (stride 8), feeding an **11-state FSM** that catches skipped, out-of-sequence and wrong-object steps.
  - Guidance and alerts: next-step guidance on the HUD and by voice; offline OS TTS with priority preemption and cooldown. — [README](https://github.com/amitraghvan/Orion/blob/HEAD/README.md)
  - Logs and outputs: SQLite + `events.json` + `timeline.log`; async MP4 recording; **HTTP MJPEG on :8080/live**; **PySide6** desktop cockpit; FastAPI headless API. Supports Apple MPS, CUDA and CPU; includes a C++20 pybind camera/frame-buffer module (`cpp/`). — [README](https://github.com/amitraghvan/Orion/blob/HEAD/README.md)
  - Orientation handling: "torso-centered, scale-normalized skeletal coordinates" for microgravity invariance, plus Shannon-entropy gating (H ≤ 1.40) and debouncing before a state transition. — [README](https://github.com/amitraghvan/Orion/blob/HEAD/README.md)
  - Dataset: **own dataset of 20 videos, 4 subjects, 5 protocols (E01–E05), 3 anomaly sequences**, shipped as ~1,374 `.npz` skeleton sequences (train/val/test). — [README](https://github.com/amitraghvan/Orion/blob/HEAD/README.md); file tree
  - Accuracy (honest): ST-GCN **95.56% train but 24.78% validation on held-out subjects**. Smoke tests report 21.2 FPS / 49.6 ms latency; "205 tests passed". — [README](https://github.com/amitraghvan/Orion/blob/HEAD/README.md)
  - Weights shipped: `stgcn_har_v1.pt` (1.78 MB), `yolo11n.pt`, `yolo11n-pose.pt`. RTSP/WebRTC/OpenVINO/TensorRT appear in code/docs (keyword grep) but the README names MJPEG as the stream.

- **arnab9957/realtime-work-detection** (explicitly "Problem Statement ID: 26174"; still being pushed on 2026-09-27)
  - Architecture: an "8-Agent Blackboard" multi-agent system (Perception YOLOv8n + 3D HMR, IMU agent with a 128 Hz ZUPT filter, Fusion, HAR, Digital Twin, deterministic-FSM Validation, Reasoning for next-step guidance, Monitoring for GUI + offline TTS + "RTSP" + JSONL). Claims 80+ FPS. — [README](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/README.md)
  - **Real HMR code:** `src/core/mesh_recovery.py` masks the person with **YOLO-World (yolov8s-worldv2)** and then runs a vendored **Multi-HMR** (`src/multihmr2/`, checkpoint `models/hmr2_checkpoints/multihmr2.pt`) over the video as an offline post-process. — [mesh_recovery.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/core/mesh_recovery.py). This work was contributed through PRs titled "HMR2"/"Hmr" by Ocean-101 — [PR #4](https://github.com/arnab9957/realtime-work-detection/pull/4)
  - Streaming: the "RTSP" claim is actually a Python `ThreadingHTTPServer` serving **MJPEG** at `/stream`, alongside local MP4. — [video_pipeline.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/streaming/video_pipeline.py)
  - Other components:
    - TTS: `pyttsx3` fallback — [offline_tts.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/audio/offline_tts.py)
    - LLM: **Ollama** used for post-hoc audit reports, with a deterministic fallback — [offline_llm_analyzer.py](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/src/llm/offline_llm_analyzer.py)
    - FSM: JSON FSM configs matching the PS's **red/yellow box** experiment (`BAS-EXP-26174`, 15-frame debounce, contact threshold 0.1 m, lid-open angle 45°, forbidden events) — [experiment_fsm.json](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/configs/experiment_fsm.json)
    - Dataset: a domain-randomized **synthetic YOLO dataset generator** plus a webcam annotator; models and datasets are on Google Drive — [README](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/README.md)

- **Pranvkumar/AI-HAR-For-BAS**
  - Target hardware: Lenovo Legion with an RTX 4060 (NVENC).
  - Detection and tracking: Ultralytics ByteTrack plus a custom YOLO for the PS's outer container and inner boxes. Weights shipped: `models/exported/best.pt` 5.35 MB and `models/objects/objects.onnx` 10.1 MB. A MediaPipe **gesture_recognizer.task** handles hands.
  - Outputs: annotated MP4, **JSONL telemetry**, and an **MJPEG stream at :8000/stream**, with a **Streamlit** dashboard. — [README](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/README.md)
  - Dependencies include pyttsx3, **faster-whisper**, SpeechRecognition and onnxruntime (so it takes voice input too); code greps also show piper/espeak, RTSP/UDP and TensorRT paths. — [requirements.txt](https://github.com/Pranvkumar/AI-HAR-For-BAS/blob/HEAD/requirements.txt)
  - Dataset: its own `data/sih_box_experiment` and `data/tool_detection` YOLO datasets (195 jpg). Docs report **"held-out accuracy 84.7% (operator op3)"**, a leave-one-operator-out result. — repo docs (grep of `docs/*.md`)

- **MakInShort/Hangla-Coders-SIH2026174** (Team Hangla Coders; 4★/5 forks; copies at yash2744/… and aceholland/…)
  - Approach: "protocol-aware step recognition using **MediaPipe hand landmarks + YOLO object features + XGBoost classifier**", a deterministic FSM with skip/repeat/out-of-sequence detection, and a **PySide6** mission-control dashboard plus Streamlit/**streamlit-webrtc**. Logging is JSONL. — [yash2744 README](https://github.com/yash2744/AI---Human---Activity---Recognition-for-Bas-Experiment-SIH-2026-/blob/HEAD/README.md)
  - They chose a **different experiment**: a wireless-microphone protocol (M0–M6) with G1/G2 LED verification gates, and trained a YOLO26 microphone detector (`runs/detect/.../microphone_yolo/baseline-2/weights/best.pt`, 19.35 MB). — file tree
  - Accuracy: the step-classifier docs report **Accuracy 75.62% / Macro-F1 62.34% / Balanced acc 63.23%**, and a second run at 71.07% / 67.44% / 76.60%. — repo docs (grep)

- **na124441/ASTRA ("Autonomous Space Task Recognition & Assistance")**
  - Models and deployment: a **MobileNetV3 + 2-layer LSTM** temporal action model on real RGB clips; FastAPI + React portal; claims Jetson AGX Orin TensorRT FP16/INT8 deployment. A native **Android multi-camera collector app** (Kotlin) records synchronized experiment runs. — [README](https://github.com/na124441/ASTRA/blob/HEAD/README.md)
  - **MicroG-4M:** it includes a **MicroG-4M temporal baseline** (`train_microg.py`, a CausalMicroGLSTM, a MicroG taxonomy/splitter with "fail-closed" integrity checks). — [README](https://github.com/na124441/ASTRA/blob/HEAD/README.md); tests `tests/unit/training/test_microg_training.py`
  - Accuracy: metrics files show `baseline_test_accuracy 0.888` and `lstm_test_accuracy 0.911`, plus a separate "Validation Accuracy 57.50% / Macro-F1 56.89%". Weights shipped: `temporal_model.pt` (0.27 MB) and `baseline_mlp.pt`. — repo metric JSON/md (grep)

- **gsaivenkateshkumar/ASTRA-HAR-EdgeAI**
  - Stack: YOLO11n + MediaPipe pose_landmarker_lite + hand_landmarker; FastAPI + React; pyttsx3/gTTS; JSONL event logs.
  - Dataset pipeline: raw recordings → frame extraction → **CVAT annotation** → a YOLO dataset (`astra_har_detection_v1..v8`, `red_blue_box_extraction` runs). **About 10,667 jpg and 2,420 label txt files are committed.** — [README](https://github.com/gsaivenkateshkumar/ASTRA-HAR-EdgeAI/blob/HEAD/README.md); file tree
  - Weights are stored in **Git LFS** (e.g. best.pt, 5.4 MB pointer). Training results show **mAP50 0.966 / mAP50-95 0.563** and 0.945/0.571. — results JSON (grep)

- **FAILEDdokja/HAR-system**
  - Stack: YOLO11n + **YOLO11n-pose**, an 8-step protocol FSM, `pyttsx3` with SAPI5/**espeak-ng**, `events.jsonl`, and a **Flask MJPEG GUI/stream at :8080**. It is **Docker Compose**-packaged with a camera profile.
  - Testing: a `tools/make_synthetic_video.py` gate test renders a synthetic run and self-verifies "8 COMPLETED in order". The sample-vial transfer step is dropped from the live demo. — [README](https://github.com/FAILEDdokja/HAR-system/blob/HEAD/README.md)

- **atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174 ("A.T.L.A.S.")**
  - Perception: YOLOv8 detection, YOLOv8-Pose body/hand tracking, **ORB feature matching** for object state, **HSV color segmentation**, MediaPipe hand_landmarker, plus a "multiple evidence signals" verifier that feeds an FSM. — [README](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174/blob/HEAD/README.md); WebSearch summary
  - Operator identity: **face_recognition biometrics** (`encodings_cache.pkl`). — [engine.py](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174/blob/HEAD/bas-apg-backend/app/core/engine.py)
  - Tool orientation: a **PnP solver for tool pitch/yaw/roll "in microgravity"** — `bas-apg-backend/app/engines/pnp_solver.py`
  - Stack: FastAPI + WebSocket + React HUD; `pyttsx3`/macOS `say`; albumentations "microgravity rotation augmentation" (requirements comment). Dependencies include transformers/accelerate and grpcio-tools. — [requirements.txt](https://github.com/atulharshvardhan2006/ATLAS-FULL-STACK-SIH-26174/blob/HEAD/bas-apg-backend/requirements.txt)

- **Shre-Sat/BASGUARD**
  - Approach: "deterministic **Petri-net** logic" for protocol enforcement, a "Mission" GUI in **PyQt6** + pyqtgraph, and a React frontend. TTS is **Piper** (`en_US-lessac-medium.onnx`) with a pyttsx3 fallback. Code greps show RTSP via **MediaMTX**/GStreamer, and the stack includes YOLOv8 + MediaPipe. — [README](https://github.com/Shre-Sat/BASGUARD/blob/HEAD/README.md); [tts_manager.py](https://github.com/Shre-Sat/BASGUARD/blob/HEAD/src/alerting/tts_manager.py); [requirements.txt](https://github.com/Shre-Sat/BASGUARD/blob/HEAD/requirements.txt)
  - The description says it is "orientation-…" (truncated) for BAS/ISS payloads. — search API description

- **krishnakankane/isro-bas-har**
  - Stack: custom-trained YOLOv8n for space apparatus, MediaPipe pose/hands, a **Conv1D-BiGRU** HAR model (`conv1d_bigru_best.pt`, 2.6 MB), a 10-step "Synthetic Demonstration Protocol", pyttsx3/SAPI5 with Piper, JSONL + SQLite + CSV, a PyQt6 GUI, RTSP input, and a "zero-g frame orientation normalizer" (a camera rotation setting of 0/90/180/270). Target: **GTX 1650 4 GB**. — [README](https://github.com/krishnakankane/isro-bas-har/blob/HEAD/README.md); [settings.py](https://github.com/krishnakankane/isro-bas-har/blob/HEAD/config/settings.py)
  - Accuracy: docs list *targets* (mAP@50 ≥ 0.88) and a metrics JSON with `accuracy: 1.0, macro_f1: 1.0` (likely synthetic or overfit). — repo docs (grep)

- **Rahulrwt99/bas-experiment-assistant ("BAS-EXA")**
  - Rack-relative approach: it calibrates a **rack reference frame with AprilTags** (`pupil_apriltags`) — a lightweight take on the PS's "relative to the payload rack" requirement. — [rack_calibration.py](https://github.com/Rahulrwt99/bas-experiment-assistant/blob/HEAD/src/perception/rack_calibration.py)
  - Stack: PySide6 GUI, YOLOv8 + MediaPipe, pyttsx3, JSONL.
  - Honesty: the README says "Mission-specific dataset collection and model training remain necessary", and activity comes from a simulator (`src/activity/simulator.py`). The target is macro-F1 ≥ 0.90 on unseen participants (a target, not a result). — [README](https://github.com/Rahulrwt99/bas-experiment-assistant/blob/HEAD/README.md)

- **Monishwarann/SIH ("ASTRA-HAR")**
  - Claims: 16 BAS action classes with a 3D CNN/MoViNet in the diagram, MediaPipe, ONNX YOLO, an FSM, pyttsx3, SQLite/PDF/CSV/JSON reports, and a **Flutter desktop app plus a React dashboard**, pretrained on UCF101 and then fine-tuned. — [README](https://github.com/Monishwarann/SIH/blob/HEAD/README.md)
  - Headline claim: **94.61% accuracy, 93.61% F1**.
  - **These numbers are generated randomly:** `training/evaluate.py` sets each class accuracy to `round(0.91 + np.random.uniform(0.0, 0.07), 4)` and derives precision/recall by subtracting constants. — [evaluate.py](https://github.com/Monishwarann/SIH/blob/HEAD/training/evaluate.py)
  - **Placeholder weights:** `models/activity_model.pt` and `.onnx` are 34- and 31-byte text files ("ASTRA_HAR_PYTORCH_OFFLINE_MODEL_V1"). Only `best_bilstm_model.keras` (33 MB) is a real model.

- **subish345/Astra ("ASTRA-EA") / orbitademo ("ORBITA") / AstroActAi**
  - Scale and claims: about 1,280 files and 580 .py files, with a PyQt GUI, YOLOv8, MediaPipe 33-point pose, pyttsx3 and 48 Jetson mentions. The README claims **mAP@0.5 92.4%**, up from a 64.7% baseline.
  - **The "weights" are fake:** `ASTRA_OBJECT_DETECTOR_v0.1.0.pt` is a ~1.2 KB **JSON file** holding metrics such as `val_mAP50 0.87` and `epoch_duration_sec 0.146`. It is not a model. — [README](https://github.com/subish345/Astra/blob/HEAD/README.md); file blob `models/checkpoints/ASTRA_OBJECT_DETECTOR_v0.1.0.pt`
  - AstroActAi uses MediaPipe efficientdet_lite0.tflite + pose_landmarker. — [README](https://github.com/subish345/AstroActAi/blob/HEAD/README.md)

- **MrRouhin/liquid-mix-ai-experiment**
  - A different experiment: 6-step liquid mixing, where the tube colour must read as the mixed colour.
  - Stack: **YOLOv8n trained on a self-collected dataset** (`runs_liquid_mix/v1/weights/best.pt`, 5.4 MB; about 450 images) + HSV, a sequence engine, `logs/run.jsonl`, FastAPI MJPEG + `/status`, a Streamlit dashboard, and pyttsx3/espeak/`say`. — [README](https://github.com/MrRouhin/liquid-mix-ai-experiment/blob/HEAD/README.md)

- **patilbhishmesh/cv-project-26174**: YOLO11/YOLO26 carton and hand detection with a careful ablation ("TEST +2.0 pts mAP50 … hand precision regressed"), plus an FSM, JSONL/CSV and UDP/MJPEG. The README is empty ("# cv-project"). — repo docs (grep)

- **samyuthasankar07-ctrl/astra ("ASTRA-SENSE")**: a custom-trained YOLOv8 `astronaut.pt` (6 MB) with about 4,531 images plus labels committed, YOLOv8-pose, a Tkinter GUI, pyttsx3 and RTSP. The README is essentially a demo video link. — [repo](https://github.com/samyuthasankar07-ctrl/astra)

**Tier B: smaller working prototypes (verified dependencies)**

- **Nimbus3001/KRAMA** (pushed 09-24): uses **RTMPose-m SimCC ONNX** pose (`models/pose/rtmpose-m_simcc.onnx` via its own onnx_runtime wrapper) with FastAPI + React, RTSP and JSONL. It is the only competitor found with RTMPose, which is our survey's recommended pose model. — [pose/config.py](https://github.com/Nimbus3001/KRAMA/blob/HEAD/KRAMA/ai/pose/config.py)
- **Kartikey2006/Astra**: **zero-shot HAR with Microsoft X-CLIP** (`microsoft/xclip-base-patch16-zero-shot`), YOLOv8-pose, pyttsx3, FastAPI + React. Its frontend types carry `gemini_status/gemini_model` fields, which hints at a Gemini path (unverified). — [xclip_har.py](https://github.com/Kartikey2006/Astra/blob/HEAD/backend/app/har/xclip_har.py); [verify_xclip.py](https://github.com/Kartikey2006/Astra/blob/HEAD/backend/verify_xclip.py)
- **dnvrmhra/Hands-Object-Integration** (Team UnoFlyp; copy at WaseemAkam/please): a "YOLO Vision Pipeline" for "ISS Columbus Science Module" equipment (glovebox, centrifuge, sample tubes). It ships trained `best.pt`/`multilight_best.pt`/`multilight_v2_best.pt` (~6 MB each), a synthetic dataset, and a React frontend with `useMockDetections`. — [README](https://github.com/dnvrmhra/Hands-Object-Integration/blob/HEAD/README.md). Its sibling [dnvrmhra/ASTRA-HAR](https://github.com/dnvrmhra/ASTRA-HAR) is a React-only simulation (`src/state/sim.ts`).
- **k-a-v-i-n-0-0-2/hawk-bas ("HawkAI")**: MediaPipe + FSM + pyttsx3 + FastAPI/React. It uses a **Gemini API (gemini-3.5-flash) with an Ollama fallback**, a cloud dependency that conflicts with the offline requirement. The README is 2,346 lines. — [gemini_provider.py](https://github.com/k-a-v-i-n-0-0-2/hawk-bas/blob/HEAD/llm/gemini/gemini_provider.py)
- **ishwaryajuttu-gif/SIH_2026 ("BAS-HAR")**: live webcam → OpenCV → YOLO11n → MediaPipe → HOI → temporal reasoning → state machine → workflow and safety engines → FastAPI → React. Uses pyttsx3 and Web Speech. It is explicitly honest: "Nothing … trained on ISRO, BAS or astronaut data" and "**No accuracy figures**". — [README](https://github.com/ishwaryajuttu-gif/SIH_2026/blob/HEAD/README.md)
- **Thiru-Lokesh-24/cv_module**: a CV module only. YOLO (person + objects) + MediaPipe Pose + Hands produce per-frame structured features for a separate activity module; it "does not classify activities itself". — [README](https://github.com/Thiru-Lokesh-24/cv_module/blob/HEAD/README.md)
- **Dynamo-SJoshi/VisionX_SIHproject**: a JSON graph FSM (SKIPPED/WRONG_ORDER/UNEXPECTED/COMPLETED), USB/**RTSP** input with a synthetic fallback, pyttsx3, `events.jsonl` + `summary.csv`, FastAPI + **Streamlit**. **COCO-pretrained YOLOv8 classes are mapped onto BAS items**, and there are mock detector/action modules. — [README](https://github.com/Dynamo-SJoshi/VisionX_SIHproject/blob/HEAD/README.md)
- **bhaveshnerkar/SpaceAssist**: FastAPI + SQLite + React, a real OpenCV webcam with MediaPipe pose/hand, and a "Sample Processing Experiment" FSM (Pick Container → Open Container …). It admits a "genuine (honestly untrained) activity-recognition model interface" and includes JUDGE_DEMO.md. — [README](https://github.com/bhaveshnerkar/SpaceAssist/blob/HEAD/README.md)
- **moiz2405/ASTRA-SEQ**: HSV RED/YELLOW detection behind a detector interface, hand tracking, a procedure FSM, pyttsx3, JSONL, and **UDP chunked-JPEG streaming** (`socket.SOCK_DGRAM`). — [README](https://github.com/moiz2405/ASTRA-SEQ/blob/HEAD/README.md); `astra_seq/streaming.py`
- **ParulDoshi/Byte_Brahmas**: webcam → MediaPipe pose → a "**rule-based classifier instead of a trained LSTM**" → sequence validation, Tkinter GUI, pyttsx3/espeak, JSONL/CSV, local recording + IP streaming. The earlier Byte_Brahmaaaas used scikit-learn RandomForest. — [README](https://github.com/ParulDoshi/Byte_Brahmas/blob/HEAD/README.md); [Byte_Brahmaaaas README](https://github.com/ParulDoshi/Byte_Brahmaaaas/blob/HEAD/README.md)
- **Paru22/SIH**: YOLO11n + YOLO11n-pose + RandomForest `activity_model.pkl` (4.2 MB), with its own recorded mp4/csv data. — [repo](https://github.com/Paru22/SIH)
- **notaditya28/isro_har_sih_project**: a different reference experiment, a **four-cube colour-order stacking task**. Uses YOLO11 (custom `best.pt`, 5.2 MB) + MediaPipe + TCN, and pyttsx3. — [README](https://github.com/notaditya28/isro_har_sih_project/blob/HEAD/README.md)
- **faiisalaziz/sih_space_har ("AstroGuard")**: a "Payload Sample Transfer" experiment built on YOLOv8 + MediaPipe + FastAPI + pyttsx3 + JSONL. — [README](https://github.com/faiisalaziz/sih_space_har/blob/HEAD/README.md)
- **aaryan2203/AI-BASED-HAS** and **Sheen11-ing/…BAS-** (same README template): YOLO11n-pose + an **LSTM/GRU** activity model (`activity_lstm.pth`, 1 MB, in aaryan2203), PySide6/Tkinter, pyttsx3. There is an `hmr_interface.py` with a `StubHMRProvider`, i.e. **HMR planned but stubbed**. — [Sheen11 hmr_interface.py](https://github.com/Sheen11-ing/AI-based-Human-Activity-Recognition-BAS-/blob/HEAD/BAS/src/hmr_interface.py)
- **sangsaptak0704/KriyaSense** (Team Astrocrew): runs **MediaPipe Tasks in the browser** (PoseLandmarker + ObjectDetector efficientdet_lite0.tflite via WASM) with a FastAPI/WebSocket backend and Web Speech voice. "LSTM/Transformer" appears only as UI text. — [real_detection.js](https://github.com/sangsaptak0704/KriyaSense/blob/HEAD/frontend/js/real_detection.js)
- **StarCodes03/TARA-Astronaut-Assistant**: YOLOv8n-pose + FastAPI MJPEG/WebSocket and "hierarchical temporal action" claims, with a telemetry/fault-injection simulator. — [README](https://github.com/StarCodes03/TARA-Astronaut-Assistant/blob/HEAD/README.md)
- **Others, all MediaPipe (± YOLOv8) + rule FSM + pyttsx3/Web Speech + JSONL/CSV + Flask/FastAPI/Streamlit/PyQt:**
  - [Ayush-sharma-mx/bas-ai-monitor](https://github.com/Ayush-sharma-mx/bas-ai-monitor) (PyQt, ONNX export, RTSP)
  - [aayushjadhav05128-sudo/bas-experiment-copilot](https://github.com/aayushjadhav05128-sudo/bas-experiment-copilot) (GRU + LLM mention, RTSP)
  - [HarshAudichya/T-A-R-A…](https://github.com/HarshAudichya/T-A-R-A-for-mac-with-audio-alert) (Streamlit + WebRTC, macOS `say`)
  - [kavyareddypallala-cell/BAS-Vision](https://github.com/kavyareddypallala-cell/BAS-Vision)
  - [yashkhochare-79/HAR_satellite](https://github.com/yashkhochare-79/HAR_satellite)
  - [PriyankShingala/Stellar](https://github.com/PriyankShingala/Stellar) (PySide6, ONNX)
  - [chaitanyareddykarri/…BAS-](https://github.com/chaitanyareddykarri/AI-Human-Activity-Recognition-for-On-board-BAS-) (PySide6; labels the person "ASTRONAUT SIMULATION")
  - [garvit000/procsight](https://github.com/garvit000/procsight) (RandomForest `step_classifier.pkl` trained on a **mock-generated** dataset, `generate_mock_dataset.py`)
  - [Himashwetha777/SIH26174-Astronaut-HAR-System](https://github.com/Himashwetha777/SIH26174-Astronaut-HAR-System) (Streamlit + Flask)
  - [swatikrishnamurthy24-dotcom/AI-ASSISTANCE-ISRO-](https://github.com/swatikrishnamurthy24-dotcom/AI-ASSISTANCE-ISRO-) (Streamlit)
  - [dhanya-s-coder/sih26174](https://github.com/dhanya-s-coder/sih26174) (RTSP/MJPEG/ffmpeg, albumentations)
  - [h1e2l3l4o5/ORBIT-HAR](https://github.com/h1e2l3l4o5/ORBIT-HAR) (YOLO11 + MediaPipe, React)
  - [madhurgitbub/ISRO](https://github.com/madhurgitbub/ISRO) (Tkinter + FastAPI)
  - [Prithivi2005-raj/space-experiment-astra](https://github.com/Prithivi2005-raj/space-experiment-astra) (Piper/Vosk/LLM mentions, React)
  - [sohamjachak003/AI-BAS-Assistant_2](https://github.com/sohamjachak003/AI-BAS-Assistant_2)
  - [David4988/Orbit-Assist-SIH26174](https://github.com/David4988/Orbit-Assist-SIH26174) (Vosk voice commands)
  - [yamantaka-singh/BAS](https://github.com/yamantaka-singh/BAS) (JSONL telemetry focus)
  - [Sachin-Malviya/ASTRA-BAS-AI](https://github.com/Sachin-Malviya/ASTRA-BAS-AI) ("natural language explanations")
  - [sighkalp/EKA…](https://github.com/sighkalp/EKA-Edge-Kognitive-Assistant)
  - [antas06/exp-ass-bas](https://github.com/antas06/exp-ass-bas)
  - [ArrushiTripathi2429/Space-Sight](https://github.com/ArrushiTripathi2429/Space-Sight)
  - — keyword/dependency grep across cloned repos (requirements.txt and source)
- **Teams that misread or re-scoped the PS:**
  - [ThunderBolt975/KRIYA…PS26174](https://github.com/ThunderBolt975/KRIYA-SIH-2026-PS26174-ISRO) (and its lalwanisneha00 twin) classifies **crew daily activities** (exercising, eating, maintenance, idle, motionless) instead of experiment steps — [README](https://github.com/ThunderBolt975/KRIYA-SIH-2026-PS26174-ISRO/blob/HEAD/README.md)
  - [ThouheedAhmed/DRISHTI-Edge-AI](https://github.com/ThouheedAhmed/DRISHTI-Edge-AI) does **posture/distress** classification (claims 97.25% held-out, 91.99% live) — [README](https://github.com/ThouheedAhmed/DRISHTI-Edge-AI/blob/HEAD/README.md)
  - [DarpanSakpal/HUMAN-ACTIVITY-RECOGNITION](https://github.com/DarpanSakpal/HUMAN-ACTIVITY-RECOGNITION) does walking/running/falling/idle on sensor CSV ("3-Hour MVP")
  - [Ozair-aly/orbital-hai](https://github.com/Ozair-aly/orbital-hai) runs a RandomForest on **simulated 6-DoF IMU** signals
  - [yashik22-ai/BASE](https://github.com/yashik22-ai/BASE) ("ActiNexus") trains a scikit-learn activity model
- **HSV/colour-heuristic only:** [haasitha1603/NAVIKSHA](https://github.com/haasitha1603/NAVIKSHA) (`inference_service.py` uses `cv2.inRange` masks; its `ml/inference/detector.py` and `train_detector.py` are 0-byte files; it has a Protocol Designer and SQLite/JSON/JSONL/CSV export; about 71 WhatsApp mp4 clips committed as its dataset). [Shreyansh-DevHub/AI-HAR-…](https://github.com/Shreyansh-DevHub/AI-HAR-_On-board_BAS_Experiment) is a browser Canvas HSV red/yellow detector with a 4-phase FSM, a 5-second hold rule and Web Speech; only the README and a screenshot are in git. — [NAVIKSHA inference_service.py](https://github.com/haasitha1603/NAVIKSHA/blob/HEAD/backend/app/services/inference_service.py); [Shreyansh README](https://github.com/Shreyansh-DevHub/AI-HAR-_On-board_BAS_Experiment/blob/HEAD/README.md)

**Tier C: mock, simulated or README-only**
- **rudra00434/HOMOSENSYS-AI-Assistant** (the most-starred at 5★/6 forks): a React/FastAPI/WebSocket "full-stack MVP" whose CV is **`MockDetector`, `MockPoseEstimator` and `MockTracker`** over a zero array. Activity labels come from scripted scenarios with `random.uniform` confidences; the backend requirements contain no CV library. It has a "Grounded Guidance Assistant" (a `TemplateLLM`) and a real deterministic FSM. — [monitoring_service.py](https://github.com/rudra00434/HOMOSENSYS-AI-Assistant/blob/HEAD/backend/services/monitoring_service.py); [scenarios.py](https://github.com/rudra00434/HOMOSENSYS-AI-Assistant/blob/HEAD/backend/services/scenarios.py)
- **hamsadevang849-sys/…**: `EdgeInferenceEngine` hard-codes `active_model = "ViTPose-Edge + SpatioTemporalTransformer"`, `device = "Rad-Hardened Edge TPU"` and `latency_ms = 11.4`, then simulates events. Its README claims HMR, NVMe storage and IP streaming, and it adds a WebGL station vista. — [edge_inference.py](https://github.com/hamsadevang849-sys/AI-Human-Activity-Recognition-for-On-board-BAS-Experiments/blob/HEAD/backend/edge_inference.py)
- **Omkar96-18/SIH26174_BAS_AgentByXathonerTeam** (Xathoner Team; created 09-26): **README + LICENSE only**, but it holds the most ambitious written spec:
  - MediaPipe multi-task + MediaPipe Object Detection fine-tuning
  - **MS-TCN/MS-TCN++ temporal action segmentation**, ONNX INT8 (claimed ~2.8 ms)
  - rack-fiducial-anchored "orientation-agnostic HMR"
  - TFLite + ONNX Runtime
  - FFmpeg H.264 local recording + **RTSP + WebRTC (<15 ms)**
  - PySide6 GUI, SQLite + JSONL logs, `uv` packaging
  - — [README](https://github.com/Omkar96-18/SIH26174_BAS_AgentByXathonerTeam/blob/HEAD/README.md)
- README-only or empty: Vynex-Labs (title only), Priyanka07110/ORION (a 264-line README), Krishna105-ui ("AEGIS · AI-HAR", README + zip), Aryasurya12/AstroFlow-AI (10 markdown files), nonugit ×2, Kshitij0275, deepakbareth01, baradaditya17-creator (AI Studio boilerplate), xarjunpatil (a 2-file FastAPI stub). Frontend-only simulations: dnvrmhra/ASTRA-HAR, gurnoor88888-svg/orbit-senses ("not space-qualified… demo"), SyroSyrup98/ORBIT-HAR, sohamjachak003/AI-BAS-Assistant, Shrishti08raghav/Viksit-Coders-SIH2026 (a committed `node_modules` dump plus a yolov8n.pt). — file trees

**Cross-cutting counts** (keyword/dependency grep over about 75 cloned competitor repos; approximate, and a hit means the string appears, not that the feature works):
- **Pose:** MediaPipe is by far the most common, in roughly 45 repos. YOLO-pose (v8/11) appears in about 10. **RTMPose in 1** (KRAMA) and a doc mention in Orion. OpenPose and MoveNet in 0.
- **Detection:** mostly YOLOv8n or YOLO11n; YOLO26 in 3 (Hangla, moiz2405, patilbhishmesh); YOLO-World in 1 (arnab9957); MediaPipe EfficientDet-Lite0 in 2 (KriyaSense, AstroActAi).
- **Temporal classifier:**
  - Rule-based FSM or heuristics dominate.
  - Trained models: ST-GCN (Orion); XGBoost (Hangla); RandomForest (ParulDoshi v1, Paru22, garvit000, Ozair, DRISHTI); LSTM/GRU (na124441, aaryan2203, Sheen11, krishnakankane BiGRU, Monishwarann BiLSTM); TCN (notaditya28).
  - Zero-shot X-CLIP (Kartikey2006). MS-TCN appears only in Omkar's README.
- **Sequencing:** a deterministic JSON/YAML FSM almost everywhere. Variants are a Petri net (BASGUARD), entropy-gated FSM transitions (Orion), and an "8-agent blackboard" (arnab9957). LLMs appear only as auxiliaries: Ollama reports (arnab9957), Gemini with an Ollama fallback (hawk-bas), a template "LLM" (HOMOSENSYS).
- **TTS:** `pyttsx3` in about 35 repos; browser Web Speech in about 12; **Piper** in about 5 (BASGUARD, krishnakankane, Pranvkumar, Prithivi2005, aayushjadhav); espeak-ng in a few; macOS `say` in a few.
- **Streaming "to IP":** in nearly every repo this means **HTTP MJPEG** (`multipart/x-mixed-replace`) from Flask/FastAPI/`http.server`. RTSP appears in about 10 (mostly as an *input* source or a README claim; MediaMTX in BASGUARD). UDP chunked JPEG in moiz2405. WebRTC in streamlit-webrtc (Hangla) and T-A-R-A.
- **Logs:** JSONL is most common, then CSV and SQLite. Some teams add PDF reports (Monishwarann).
- **GUI:** FastAPI + React is most common; PySide6/PyQt6 in about 12; Streamlit in about 8; Tkinter in about 6; Flutter in 1 (Monishwarann).
- **Hardware:** mostly laptops (RTX 4060, GTX 1650, Apple MPS). Jetson is claimed in about 8 READMEs (na124441 Orin, subish345, ishwaryajuttu, Monishwarann) but no Jetson-specific build scripts were verified. Raspberry Pi appears only in na124441 docs.
- **Orientation/HMR:** real HMR only in arnab9957 (Multi-HMR, offline post-process). Torso-centred skeleton normalization in Orion. AprilTag rack frame in Rahulrwt99. PnP tool pose in ATLAS. HMR stubs in aaryan2203/Sheen11. Camera-rotation setting in krishnakankane. README-only claims in Omkar96-18, hamsadevang and others. No SMPL-based WHAM, 4DHumans or HMR2.0 was found running in a live loop.

### Inferences
- **Where ACTUS differs.**
  - *What is common:* ACTUS's skeleton → classifier → FSM spine, voice alerts, JSONL logs and MJPEG-style streaming are table stakes (~40 teams have that shape).
  - *Pose and classifier:* only Orion does a **skeleton-GCN**, and it reports 24.78% held-out validation. Only KRAMA uses RTMPose.
  - *Microgravity:* only na124441 touches **MicroG-4M**, and only arnab9957 runs **real 3D HMR**. The microgravity/astronaut-footage angle is still sparsely contested.
  - *Belief channels:* none of the inspected repos has anything like ACTUS's two-channel belief split, the abstain/ask path, or rule-based step logic over pretrained action evidence feeding the FSM as a sensor.
- **Closest competitors to watch:**
  - **Orion**: ST-GCN, honest metrics, own 20-video dataset, PySide6, C++ capture.
  - **arnab9957**: Multi-HMR, synthetic data generator, red/yellow-box FSM, still pushing on 09-27.
  - **na124441**: MicroG-4M baseline, Android multi-camera collector, MobileNetV3-LSTM at 91% test.
  - **Pranvkumar**: trained box detector, leave-one-operator-out 84.7%.
  - **Hangla Coders**: XGBoost step classifier, most forks.
  - **gsaivenkateshkumar**: large CVAT-annotated detection set with mAP50 0.966.
- **Differentiation to press in the demo/pitch:** show *held-out* numbers and be candid about limits (as Orion and ishwaryajuttu do). Show MicroG-4M on real astronaut footage. Make the skeleton representation's Earth→orbit argument explicitly. Given how many teams stop at COCO-mapped YOLO + MediaPipe + rules, demonstrating any trained, evaluated component on held-out subjects is a meaningful separator.
- **Streaming:** almost everyone equates "stream to specific IP" with an MJPEG HTTP endpoint. A true RTSP/UDP push to a configured IP, as the PS literally asks, would be a small but concrete differentiator.

### Gaps
- No repo was *executed*. Every "verified" finding is based on dependency files, source greps and reading specific files. The per-feature counts are approximate string matches.
- Several READMEs are 400–2,300 lines long and were only partly read. Some features (for example Jetson builds, WebRTC paths, next-step suggestion quality) were not traced end to end.
- Google Drive assets (arnab9957) and Git LFS assets (gsaivenkateshkumar) were not downloaded, so their sizes and contents are unconfirmed.
- Pranvkumar's 84.7% could not be tied to a specific model file (the gesture classifier vs. a GRU) without deeper reading. The same applies to na124441's two sets of numbers (91% vs 57.5%, which may be different tasks).

## How mature is each repo, and which publish trained weights or datasets (with accuracy-claim credibility)?

### Takeaway
About 15 repos are substantive working pipelines, about 45 are small working prototypes (pretrained YOLO/MediaPipe + rules), and about 20 are README-only, empty, frontend-only or mock. **Custom trained weights are published by roughly 15 teams**, almost all small YOLO detectors or tiny temporal models. At least **two teams publish fabricated metrics or fake weight files** (Monishwarann/SIH, subish345/Astra), and a third (HOMOSENSYS, the most-starred) runs on mock CV.

### Cited Findings
- **Real custom weights in git:**
  - Orion `stgcn_har_v1.pt` 1.78 MB — [Orion](https://github.com/amitraghvan/Orion)
  - Pranvkumar `best.pt` 5.35 MB + `objects.onnx` 10.1 MB — [Pranvkumar](https://github.com/Pranvkumar/AI-HAR-For-BAS)
  - Hangla Coders YOLO26 microphone `best.pt` 19.35 MB + XGBoost `step_classifier_fused.json` — [MakInShort](https://github.com/MakInShort/Hangla-Coders-SIH2026174); [yash2744](https://github.com/yash2744/AI---Human---Activity---Recognition-for-Bas-Experiment-SIH-2026-)
  - na124441 `temporal_model.pt` / `baseline_mlp.pt` — [na124441](https://github.com/na124441/ASTRA)
  - krishnakankane `conv1d_bigru_best.pt` 2.6 MB — [krishnakankane](https://github.com/krishnakankane/isro-bas-har)
  - MrRouhin `best.pt` 5.36 MB — [MrRouhin](https://github.com/MrRouhin/liquid-mix-ai-experiment)
  - notaditya28 `best.pt` 5.2 MB — [notaditya28](https://github.com/notaditya28/isro_har_sih_project)
  - samyuthasankar07 `astronaut.pt` 6 MB — [samyuthasankar07-ctrl/astra](https://github.com/samyuthasankar07-ctrl/astra)
  - dnvrmhra/WaseemAkam three ~6 MB YOLO checkpoints — [dnvrmhra](https://github.com/dnvrmhra/Hands-Object-Integration)
  - aaryan2203 `activity_lstm.pth` 1 MB — [aaryan2203](https://github.com/aaryan2203/AI-BASED-HAS)
  - Paru22 RandomForest 4.2 MB — [Paru22](https://github.com/Paru22/SIH)
  - Monishwarann `best_bilstm_model.keras` 33 MB — [Monishwarann](https://github.com/Monishwarann/SIH)
  - DRISHTI `posture_classifier.pkl` — [ThouheedAhmed](https://github.com/ThouheedAhmed/DRISHTI-Edge-AI)
  - garvit000 `step_classifier.pkl` (trained on mock data) — [garvit000](https://github.com/garvit000/procsight)
- **Weights in LFS or external storage:** gsaivenkateshkumar keeps YOLO runs v1–v8 as LFS pointers (e.g. 5,446,490 bytes) — [gsaivenkateshkumar](https://github.com/gsaivenkateshkumar/ASTRA-HAR-EdgeAI). arnab9957 keeps "Models & Datasets" on Google Drive — [README](https://github.com/arnab9957/realtime-work-detection/blob/HEAD/README.md)
- **Datasets committed:**
  - gsaivenkateshkumar: ~10.7k images + labels, CVAT-derived
  - samyuthasankar07: ~4.5k images + labels
  - MrRouhin: ~450 images
  - Pranvkumar: ~195 images across box/tool sets
  - patilbhishmesh: ~45 images
  - Orion: ~1.4k `.npz` skeleton sequences from 20 of its own videos
  - na124441: processed `.npz` runs plus an Android collection app
  - NAVIKSHA: about 71 WhatsApp mp4 clips
  - — file trees of the respective repos
- **Accuracy claims and their credibility:**
  - Orion: 95.56% train / **24.78% held-out val** (credible; self-critical) — [README](https://github.com/amitraghvan/Orion/blob/HEAD/README.md)
  - Pranvkumar: 84.7% held-out operator
  - Hangla: 71–76% acc, macro-F1 62–67%
  - na124441: 88.8% baseline / 91.1% LSTM test, and a separate 57.5% val
  - gsaivenkateshkumar: mAP50 0.966 / 0.945 (Ultralytics results)
  - DRISHTI: 97.25% held-out (posture, not steps)
  - krishnakankane: accuracy 1.0 (suspicious)
  - subish345: mAP@0.5 92.4% (**backed by a JSON "weights" file, not a model**)
  - Monishwarann: 94.61% (**random-number generator in evaluate.py**) — [evaluate.py](https://github.com/Monishwarann/SIH/blob/HEAD/training/evaluate.py)
  - ishwaryajuttu explicitly publishes "No accuracy figures" — [README](https://github.com/ishwaryajuttu-gif/SIH_2026/blob/HEAD/README.md)
- **Recency:** most active in the final week are arnab9957 (pushed 09-27), hamsadevang (09-27), ATLAS (09-26), BASGUARD (09-26), dhanya-s-coder (09-26), ishwaryajuttu (09-26), ParulDoshi/Byte_Brahmas (09-26), aaryan2203 (09-26), Omkar96-18 (09-26 creation), KRAMA (09-24), Monishwarann (09-23). — search API `pushed_at`

### Inferences
- The credible bar among competitors is low. Honest held-out evaluation, as Orion does even with a bad number, is rare, and fabricated numbers exist. If judges probe, a clearly reproducible held-out metric from ACTUS (even a modest one) combined with a real astronaut-clip demo would likely stand out.
- Most "trained AI model" deliverables are small YOLO detectors fine-tuned on webcam footage of the PS's red/yellow boxes or a team-chosen object. Very few have trained a *step or action* model evaluated across subjects.

### Gaps
- Commit counts and contributor counts were not collected (shallow clones; the API core rate limit is 60/hr unauthenticated).
- No weights were loaded to confirm they are valid models. Only file sizes and headers were checked, which is how the fake JSON "weights" were caught.
- External artifacts (Drive, LFS, YouTube demo links in READMEs) were not reviewed.
