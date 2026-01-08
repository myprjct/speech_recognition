# speech_recognition
●	Voice-Based Interaction:
 Speech-driven controls were incorporated to allow users to interact with the application using voice commands. This minimizes dependency on traditional graphical interfaces and improves accessibility.

●	Frame-Level Video Processing:
 Video input is analyzed sequentially on a frame-by-frame basis without skipping frames. This design ensures that transient or fast-moving objects are not missed, which is crucial for assistive applications where incomplete detection may compromise user safety.

●	Efficient Object Detection:
 A YOLOv8-based object detection model was used to identify objects within each video frame. The model selection focused on achieving a balance between detection accuracy and computational efficiency.
