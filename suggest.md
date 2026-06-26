wget -O /home/si/PycharmProjects/CV_Project/data/raw/video_002.avi "https://s3.master.siblamer.net/cvproject/video_002.avi" 


```python
   import cv2                                                                                                                                                                                                      
   cap = cv2.VideoCapture('data/raw/video_002.avi')
   print('Width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
   print('Height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
   print('FPS:', cap.get(cv2.CAP_PROP_FPS))
   print('Frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
   print('Duration:', round(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS), 1), 's')
   cap.release()
```

  ---
  Project: People Counting & Flow                                                                                                                                                                                          
                                                                                                                                                                                                                           
  Pipeline: image → enhance → segment → clean → detect → decision                                                                                                                                                          
                                                                                                                                                                                                                           
  ---             
  Project Structure (recommended)                                                                                                                                                                                          
                                                                                                                                                                                                                           
  CV_Project/
  ├── data/                                                                                                                                                                                                                
  │   ├── raw/          # input videos/frames
  │   └── output/       # per-stage output images                                                                                                                                                                          
  ├── src/
  │   ├── enhance.py    # Александр: CLAHE, gamma correction                                                                                                                                                               
  │   ├── segment.py    # Александр: MOG2 background subtraction                                                                                                                                                           
  │   ├── clean.py      # Богдан: morphological ops                                                                                                                                                                        
  │   ├── detect.py     # Витя: contour finding, centroids                                                                                                                                                                 
  │   ├── decision.py   # Витя: centroid tracking, line crossing (In/Out)                                                                                                                                                  
  │   └── visualize.py  # Богдан: draw boxes, counters on frames                                                                                                                                                           
  ├── pipeline.py       # Витя: assembles full pipeline end-to-end
  ├── main.py           # entry point (video path → output)                                                                                                                                                                
  └── report/           # Богдан: PDF report assets                                                                                                                                                                        
                                                                                                                                                                                                                           
  ---                                                                                                                                                                                                                      
  Phases                                                                                                                                                                                                                   
                  
  Phase 1 — Setup (all, 1 day)
                                                                                                                                                                                                                           
  - Init git repo, add all 3 as collaborators                                                                                                                                                                              
  - Add OpenCV to deps: uv add opencv-python numpy                                                                                                                                                                         
  - Agree on test video (e.g. MOT dataset or PETS benchmark)                                                                                                                                                               
  - Define the counting line Y coordinate                                                                                                                                                                                  
                                                                                                                                                                                                                           
  Phase 2 — Core modules (parallel, 3–5 days)                                                                                                                                                                              
                                                                                                                                                                                                                           
  Александр (enhance.py, segment.py):                                                                                                                                                                                      
  - enhance_clahe(frame) — apply CLAHE to equalize lighting
  - enhance_gamma(frame, gamma) — gamma correction                                                                                                                                                                         
  - segment_mog2(frame, subtractor) → binary mask 
                                                                                                                                                                                                                           
  Богдан (clean.py, visualize.py):                                                                                                                                                                                         
  - clean_mask(mask, kernel_size) — opening + closing                                                                                                                                                                      
  - visualize(frame, boxes, count_in, count_out) → annotated frame                                                                                                                                                         
                                                                  
  Витя (detect.py, decision.py, pipeline.py):                                                                                                                                                                              
  - detect_people(mask, min_area) → list of (cx, cy, bbox)                                                                                                                                                                 
  - track_centroids(prev, curr, max_dist) → tracked IDs                                                                                                                                                                    
  - update_counters(tracks, line_y) → in_count, out_count                                                                                                                                                                  
  - Assemble pipeline.py to chain all stages                                                                                                                                                                               
                                                                                                                                                                                                                           
  Phase 3 — Integration (Витя, 2 days)                                                                                                                                                                                     
                                                                                                                                                                                                                           
  - Wire all modules in pipeline.py and main.py                                                                                                                                                                            
  - Save 6 output images per frame: original, enhanced, raw mask, clean mask, detection, final                                                                                                                             
  - Run on 2–3 test videos                                                                                                                                                                                                 
   
  Phase 4 — Report & Demo (Богдан leads, 2 days)                                                                                                                                                                           
                  
  - PDF: problem → pipeline diagram → methods → per-stage results → failure cases                                                                                                                                          
  - Prepare live demo video
  - Each person writes their section                                                                                                                                                                                       
                                                                                                                                                                                                                           
  ---
  Missing dependencies to add                                                                                                                                                                                              
                             
  The current pyproject.toml is missing OpenCV. Run:
                                                                                                                                                                                                                           
  uv add opencv-python numpy scipy
                                                                                                                                                                                                                           
  ---             
  Key implementation choices to decide as a team                                                                                                                                                                           
                                                
  ┌─────────────────────┬─────────────────────────────────────────────────────────────────┐
  │      Decision       │                             Options                             │                                                                                                                                
  ├─────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ Test dataset        │ PETS 2009, MOT17, or your own camera recording                  │                                                                                                                                
  ├─────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ Counting line       │ Fixed Y (e.g. frame_height // 2) or configurable                │                                                                                                                                
  ├─────────────────────┼─────────────────────────────────────────────────────────────────┤                                                                                                                                
  │ Tracker             │ Simple centroid distance (as spec'd) or add Hungarian algorithm │                                                                                                                                
  ├─────────────────────┼─────────────────────────────────────────────────────────────────┤                                                                                                                                
  │ Enhancement default │ CLAHE vs gamma — pick one as primary                            │
  └─────────────────────┴─────────────────────────────────────────────────────────────────┘                                                                                                                                
                  
  ---                                                                                                                                                                                                                      
  Deliverables checklist
                        
  - GitHub repo with commits from all 3 members
  - python main.py --video input.mp4 runs end-to-end                                                                                                                                                                       
  - 6 saved output images per test video                                                                                                                                                                                   
  - PDF report (5–8 pages)                                                                                                                                                                                                 
  - 10-min presentation with live demo                                                                                                                                                                                     
  - Contribution Statement (1 page, signed)                                                                                                                                                                                
                                                                                                                                                                                                                           
  ---                                                                                                                                                                                                                      
  Want me to scaffold the file structure and write starter code for any specific module?  